from dataclasses import dataclass
from tinygrad import Tensor, dtypes, nn, Variable, TinyJit, Device
import math

@dataclass
class GPTConfig:
  block_size: int = 256
  vocab_size: int = 65
  n_layer: int = 6
  n_head: int = 6
  n_embd: int = 384


class CausalSelfAttention():
  def __init__(self, config: GPTConfig):
    assert config.n_embd % config.n_head == 0
    # key, query, value projections for all heads
    self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
    # output projection
    self.c_proj = nn.Linear(config.n_embd, config.n_embd)
    # regularization
    self.n_head = config.n_head
    self.n_embd = config.n_embd

    # this is the attn mask but gpt2 called it "bias"
    self.bias = Tensor.ones(config.block_size, config.block_size, requires_grad=False).tril() \
                      .view(1, 1, config.block_size, config.block_size)

    # kv cache
    self.k_total = None
    self.v_total = None

  def __call__(self, x: Tensor) -> Tensor:
    B,T,C = x.size() # batch size, seq len, n_embd
    assert C == self.n_embd

    if Tensor.training:
      qkv = self.c_attn(x) # (B, T, 3*n_embd)
      q, k, v = qkv.split(self.n_embd, dim=2) # (3, B, T, n_embd)
      q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, n_head, T, head_size)
      k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, n_head, T, head_size)
      v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, n_head, T, head_size)
    
      att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (B, n_head, T, T)
      att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf')) # (B, n_head, T, T)
      att = att.softmax() # (B, n_head, T, T)
      y = att @ v # (B, n_head, T, head_size)
      y = y.transpose(1, 2) # (B, T, n_head, head_size)
      y = y.view(B, T, C) # (B, T, C) since C = n_head * head_size = n_embd
      # output projection
      y = self.c_proj(y)
      
      return y

    else:
      qkv = self.c_attn(x) # (B, T, 3*n_embd)
      q, k_new, v_new = qkv.split(self.n_embd, dim=2) # (3, B, T, n_embd)
      q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, n_head, T, head_size)
      k_new = k_new.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, n_head, T, head_size)
      v_new = v_new.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, n_head, T, head_size)

      # q is (B, n_head, 1, head_size)
      # k.T is (B, n_head, head_size, 1) -> we want it to be T (of the entire sequence)
      # desired: q @ k.T -> 1xT (instead of 1x1, or TxT)

      # Initialize KV Cache if empty (prefill)
      if self.k_total is None or self.v_total is None:
        self.k_total = k_new # (B, n_head, T, head_size)
        self.v_total = v_new # (B, n_head, T, head_size)
      else:  # decode (single new token)
        self.k_total = self.k_total.cat(k_new, dim=2) # (B, n_head, T+1, head_size)
        self.v_total = self.v_total.cat(v_new, dim=2) # (B, n_head, T+1, head_size)
      T_total = self.k_total.size(2)

      att = (q @ self.k_total.transpose(-2, -1)) * (1.0 / math.sqrt(self.k_total.size(-1))) # (B, n_head, T, T)
      if T > 1:
        att = att.masked_fill(self.bias[:,:,:T,:T_total] == 0, float('-inf')) # (B, n_head, T, T) 
      att = att.softmax() # (B, n_head, T, T)
      y = att @ self.v_total # (B, n_head, T, head_size)
      y = y.transpose(1, 2) # (B, T, n_head, head_size)
      y = y.view(B, T, C) # (B, T, C) since C = n_head * head_size = n_embd
      # output projection
      y = self.c_proj(y)
      return y

class MLP():
  def __init__(self, config: GPTConfig):
    self.config = config
    self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
    self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

  def __call__(self, x: Tensor) -> Tensor:
    x = self.c_fc(x)
    x = x.gelu() # gelu approximation used in gpt2 paper
    x = self.c_proj(x)
    return x

class Block():
  def __init__(self, config: GPTConfig):
    self.conifg = config
    self.ln_1 = nn.LayerNorm(config.n_embd)
    self.attn = CausalSelfAttention(config)
    self.ln_2 = nn.LayerNorm(config.n_embd)
    self.mlp = MLP(config)

  def __call__(self, x: Tensor) -> Tensor:
    x = x + self.attn(self.ln_1(x))
    x = x + self.mlp(self.ln_2(x))
    return x

class GPT():
  def __init__(self, config: GPTConfig):
    self.config = config
    self.transformer = dict(
      wte=nn.Embedding(config.vocab_size, config.n_embd),
      wpe=nn.Embedding(config.block_size, config.n_embd),
      h=[Block(config) for _ in range(config.n_layer)],
      ln_f=nn.LayerNorm(config.n_embd),
    )
    self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
    self.last_pos = None # last token pos (when using kv cache)

  def __call__(self, idx):
    # idx is of shape (B, T)
    B, T = idx.size()
    assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is {self.config.block_size}"

    # forward the token and position embeddings
    if Tensor.training:
      pos = Tensor.arange(0, T, dtype=dtypes.long, device=idx.device) # shape (T)
    elif self.last_pos is None: # prefill
      pos = Tensor.arange(0, T, dtype=dtypes.long, device=idx.device) # shape (T)
      self.last_pos = Tensor(T-1, dtype=dtypes.long)
    else:
      pos = self.last_pos + 1
      self.last_pos.assign(pos).realize()
    pos_emb = self.transformer["wpe"](pos)
    tok_emb = self.transformer["wte"](idx)
    x = pos_emb + tok_emb
    # forward the blocks of the transformer
    for block in self.transformer['h']:
      x = block(x)
    # forward the final layernorm and the classifier
    x = self.transformer['ln_f'](x)
    logits = self.lm_head(x) # (B, T, vocab_size)
    return logits


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
    sd_keys = sd.keys()
    sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard the mask buffer

    # init a hf transformer
    from transformers import GPT2LMHeadModel
    model_hf = GPT2LMHeadModel.from_pretrained(model_type)
    sd_hf = model_hf.state_dict()

    sd_keys_hf = sd_hf.keys()
    sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore the mask buffer
    sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same
    transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

    assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
    for k in sd_keys_hf:
      if any(k.endswith(w) for w in transposed):
        # special treatment for the conv1d weights we need to transpose
        assert sd_hf[k].shape[::-1] == sd[k].shape
        sd_hf[k] = sd_hf[k].t()

      sd[k] = Tensor(sd_hf[k].numpy())

    nn.state.load_state_dict(model, sd)
    return model

  
  def generate(self, x: Tensor) -> Tensor:
    t = Tensor.training
    Tensor.training = False
    Tensor.manual_seed(42)
    """Prefill + Decode using KV Cache"""
    B, T = x.size()
    num_return_sequences = 5
    max_length = 30
    def sample(logits: Tensor):
      # we only care about the newest predicted token
      logits = logits[:,-1,:] # (B, 1, vocab_size)
      probs = logits.softmax(axis=-1)
      # do top-k sampling of 50 (hugginface pipeline default)
      # topk_probs here becomes (5, 50), topk_indices is (5, 50)
      topk_probs, topk_indices = probs.topk(50, dim=-1)
      # select a token from the top-k probabilities
      ix = topk_probs.multinomial(1)
      # gather the corresponding indices
      xcol = topk_indices.gather(-1, ix)
      return xcol

    # prefill
    logits = self(x) # (B, T, vocab_size)
    new_tok = sample(logits)
    x = x.cat(new_tok, dim=1).realize()
    #@TinyJit
    def net(new_tok: Tensor):
      logits = model(new_tok)
      new_tok = sample(logits).realize()
      return new_tok
    # append to the sequence
    while x.size(1) < max_length:
      print('before decode', x.size(1))
      new_tok = net(new_tok)
      x = x.cat(new_tok, dim=1).realize()

    Tensor.training = t
    for i in range(num_return_sequences):
      tokens = x[i, :max_length].tolist()
      decoded = enc.decode(tokens)
      print(">", decoded)


if __name__ == "__main__":
  num_return_sequences = 5
  max_length = 30

  model = GPT.from_pretrained('gpt2')
  import tiktoken
  enc = tiktoken.get_encoding('gpt2')
  tokens = enc.encode("Hello, I'm a language model,")
  x = Tensor(tokens, dtype=dtypes.long) # (8,)
  x = x.unsqueeze(0).repeat(num_return_sequences, 1) # (5, 8)
  model.generate(x)
  import sys
  sys.exit(0)

