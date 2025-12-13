from tinygrad import Tensor, dtypes
from transformers import GPT2LMHeadModel
from model import GPT
import tiktoken
import torch
import numpy as np

model = GPT.from_pretrained("gpt2")
model_hf = GPT2LMHeadModel.from_pretrained("gpt2")

enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm a language model,")
x_hf = torch.tensor(tokens, dtype=torch.long)
x_hf = x_hf.unsqueeze(0)
y_hf = model_hf(x_hf)

x = Tensor(tokens, dtype=dtypes.long) # (8,)
x = x.unsqueeze(0)
y = model(x)

np.testing.assert_allclose(y_hf.logits.detach().numpy(), y.detach().numpy(), atol=1e-5, rtol=1e-5)
print("Test passed!")