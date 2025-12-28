from tinygrad.nn.state import torch_load
from dataclasses import dataclass
from tinygrad import Tensor, Variable, nn, Context, dtypes, TinyJit, UOp
from tinygrad.helpers import DEBUG, JIT, getenv, fetch
from typing import Optional, Union
import tiktoken
import math

MAX_CONTEXT = getenv("MAX_CONTEXT", 128)
HALF = getenv("HALF")

class MultiHeadAttention:
    def __init__(self, dim, n_heads):
        self.c_attn = nn.Linear(dim, 3*dim) # x3 (Q,K,V)
        self.c_proj = nn.Linear(dim, dim)
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads

    def __call__(self, x: Tensor, start_pos:UOp, mask:Optional[Tensor]) -> Tensor:
        # x is (B, T, dim)
        if mask is not None or start_pos.val == 0:
            # no symbolic shape qkv when consuming prompts, (prefill?)
            start_pos = start_pos.val  # ty:ignore[invalid-assignment]
        
        if HALF: x = x.half() # use float16

        # (B, T, dim) @ (dim, 3*dim) -> (B, T, 3*dim) (calc all q,k,v for all heads)
        # (B, T, 3*dim) reshaped -> (B, T, 3, n_head, head_dim) (3 = [Q, K, V]
        xqkv = self.c_attn(x).reshape(None, None, 3, self.n_heads, self.head_dim)
        xq, xk, xv = [xqkv[:, :, i, :, :] for i in range(3)]
        bs, seqlen, _, _ = xq.shape

        if not Tensor.training:
          if not hasattr(self, "cache_kv"):
             self.cache_kv = Tensor.zeros(2, bs, MAX_CONTEXT, self.n_heads, self.head_dim, dtype=x.dtype).contiguous().realize()

          # update the cache
          cache_kv_slice = self.cache_kv[:, :, start_pos:start_pos+seqlen, :, :]
          xkv_new = Tensor.stack(xk, xv)
          cache_kv_slice.assign(xkv_new).realize()

          # use the cache if prefill already happened
          if start_pos > 0:
            keys = self.cache_kv[0][:, :start_pos+seqlen, :, :]
            values = self.cache_kv[1][:, :start_pos+seqlen, :, :]
          else:
            keys = xk
            values = xv

        else:
          keys = xk
          values = xv

        # tranpose to get seqlen dim at -2 and head_dim at -1
        xq, keys, values = xq.transpose(1, 2), keys.transpose(1,2), values.transpose(1,2)
        # perform attn and tranpose back and reshape -1 to `self.dim`, effectively concatenating back
        scale = 1.0 / math.sqrt(xq.size(-1))  # ty:ignore[invalid-argument-type]
        attn = xq @ keys.transpose(-2, -1) * scale

        if mask is None:
           mask = attn.ones_like(requires_grad=False, device=attn.device, dtype=dtypes.bool).where(0, -float("inf"))
           # where seems to not inherit dtypes.half, so we need to explicitly cast here
           if HALF:
             mask = mask.cast(dtypes.half)

        attn += mask

        attn = attn.softmax(-1) @ values
        xcat = attn.transpose(1, 2).reshape(bs, seqlen, self.dim)
        # final projection, yay!
        return self.c_proj(xcat)

class MLP():
  def __init__(self, dim, hidden_dim):
    self.c_fc = nn.Linear(dim, hidden_dim)
    self.c_proj = nn.Linear(hidden_dim, dim)

  def __call__(self, x: Tensor) -> Tensor:
    x = self.c_fc(x).gelu() # i dont think tinygrad has tanh approx
    return self.c_proj(x)

class Block():
  def __init__(self, dim, n_heads):
    self.ln_1 = nn.LayerNorm(dim)
    self.attn = MultiHeadAttention(dim, n_heads)
    self.ln_2 = nn.LayerNorm(dim)
    self.mlp = MLP(dim, 4*dim) # TODO: read art on why 4 is a good number here

  def __call__(self, x: Tensor, start_pos:UOp, mask:Optional[Tensor]) -> Tensor:
    x = x + self.attn(self.ln_1(x), start_pos, mask)
    x = x + self.mlp(self.ln_2(x)).contiguous()
    return x

        
@dataclass
class GPTConfig:
  block_size: int = 1024
  vocab_size: int = 50257
  n_layer: int = 12
  n_head: int = 12
  n_embd: int = 768

class GPT():
  def __init__(self, config: GPTConfig):
    self.config = config
    self.wte = nn.Embedding(config.vocab_size, config.n_embd)
    self.wpe = nn.Embedding(config.block_size, config.n_embd)
    self.h = [Block(config.n_embd, config.n_head) for _ in range(config.n_layer)]
    self.ln_f = nn.LayerNorm(config.n_embd)
    self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
    self.forward_jit = TinyJit(self.forward)


  def forward(self, tokens:Union[Tensor,UOp], start_pos:UOp):
    if not hasattr(self, 'allpos'): self.allpos = Tensor.arange(0, MAX_CONTEXT).reshape(1, -1).realize()
    if isinstance(tokens, UOp):
      seqlen = 1
      tok_emb = self.wte.weight.shrink(((tokens, tokens+1), None))
    else:
      seqlen = tokens.shape[1]
      tok_emb = self.wte(tokens)

    # not symbolic when consuming the prompt
    selected_pos = (0, seqlen) if start_pos.val == 0 else (start_pos, start_pos+1)
    pos_emb = self.wpe(self.allpos.shrink((None, selected_pos)))

    h = tok_emb + pos_emb

    if HALF: h = h.half()

    mask = Tensor.full((1, 1, seqlen, start_pos.val+seqlen), float("-inf"), dtype=h.dtype).triu(start_pos.val+1) if seqlen > 1 else None

    for hi in self.h: h = hi(h, start_pos, mask)

    logits = self.lm_head(self.ln_f(h))
    return logits


  def __call__(self, tokens:Union[Tensor,UOp], start_pos:UOp) -> Tensor:
    forward = (self.forward_jit if JIT else self.forward)
    return forward(tokens, start_pos)


  @classmethod
  def from_pretrained(cls, model_type):
    """Loads pretrained GPT-2 model weights from huggingface"""
    assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
    print("loading weights from pretrained gpt: %s" % model_type)

    # n_layer, n_head and n_embd are determined from model_type
    config_args = {
      'gpt2': dict(n_layer=12, n_head=12, n_embd=768),
      'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),
      'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),
      'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),
    }[model_type]

    config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
    config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints

    # create a from-scratch initialized nanoGPT model
    config = GPTConfig(**config_args)
    model = GPT(config)
    sd = nn.state.get_state_dict(model)

    # init a hf transformer
    weights = torch_load(fetch(f'https://huggingface.co/{model_type}/resolve/main/pytorch_model.bin'))
    transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

    for k in weights:
      if any(k.endswith(w) for w in transposed):
        # special treatment for the conv1d weights we need to transpose
        assert weights[k].shape[::-1] == sd[k].shape
        weights[k] = weights[k].T

    weights['lm_head.weight'] = weights['wte.weight']

    nn.state.load_state_dict(model, weights)
    if HALF:
      for l in nn.state.get_state_dict(model).values():
        l.replace(l.half().realize())
    return model

  def train_step(self, x: Tensor, y: Tensor):
    start_pos = Variable("start_pos", 0, MAX_CONTEXT-1).bind(0)
    logits = self.forward(x, start_pos)
    loss = logits.reshape(-1, self.config.vocab_size).cross_entropy(y.reshape(-1))
    return logits, loss

  def generate(self, prompt: str, max_length, batch_size):
    enc = tiktoken.get_encoding('gpt2')
    prompt_tokens = enc.encode(prompt)
    toks = [prompt_tokens[:] for _ in range(batch_size)] # (5, 8) 
    start_pos = 0
    while len(toks[0]) < max_length:
        print(len(toks[0]))
        tokens = Tensor([x[start_pos:] for x in toks])
        # forward the model to get the logits
        logits = model(tokens, Variable("start_pos", 1 if start_pos else 0, MAX_CONTEXT-1).bind(start_pos))
        # take the logits at the last position
        logits = logits[:, -1, :] # (B, vocab_size)
        # get the probabilities
        probs = logits.softmax()
        # do top-k sampling of 50 (huggingface pipeline default)
        # topk_probs here becomes (5, 50), topk_indices is (5, 50)
        topk_probs, topk_indices = probs.topk(50, dim=-1)
        # select a token from the top-k probabilities
        # note: multinomial does not demand the input to sum to 1
        ix = topk_probs.multinomial(1) # (B, 1)
        # gather the corresponding indices
        tok = topk_indices.gather(-1, ix) # (B, 1)
        start_pos = len(toks[0])
        # append to sequence
        for i,t in enumerate(tok.flatten().tolist()): toks[i].append(t)

    for tokens in toks:
      decoded = enc.decode(tokens)
      print(">", decoded)
    


if __name__ == "__main__":
  num_return_sequences = 5
  max_length = 30

  Tensor.manual_seed(42)
  #model = GPT.from_pretrained('gpt2')
  with Context(BEAM=2):
    #model.generate("Hello, I'm a language model,", 30, 5)
    pass

  enc = tiktoken.get_encoding('gpt2')
  with open('input.txt', 'r') as f:
    text = f.read()
  text = text[:1000]
  tokens = enc.encode(text)
  B, T = 4, 32
  buf = Tensor(tokens[:B*T+1])
  x = buf[:-1].view(B, T)
  y = buf[1:].view(B, T)
  
  model = GPT(GPTConfig())
  params = nn.state.get_parameters(model)
  optim = nn.optim.AdamW(params, lr=3e-4)

  @TinyJit
  def step():
    Tensor.training = True
    optim.zero_grad()
    _, loss = model.train_step(x, y)
    loss.backward()
    optim.step()
    return loss

  with Context(BEAM=2):
      for i in range(50):
        loss = step()
        print(f"step {i}, loss: {loss.item()}")

