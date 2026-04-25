import torch
from tokenizers import Tokenizer
from model.gpt import GPT
from config import *

# =========================
# LOAD TOKENIZER
# =========================
tokenizer = Tokenizer.from_file("tokenizer.json")

def encode(s):
    return tokenizer.encode(s).ids

def decode(ids):
    return tokenizer.decode(ids)

vocab_size = tokenizer.get_vocab_size()

# =========================
# DEVICE
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# LOAD MODEL
# =========================
model = GPT(vocab_size).to(device)
model.load_state_dict(torch.load("output/run_20260424_232122/model.pt", map_location=device))
model.eval()

# =========================
# GENERATE FUNCTION
# =========================
@torch.no_grad()
def generate(prompt="[BOS]", max_new_tokens=100):
    context = torch.tensor(encode(prompt), dtype=torch.long).unsqueeze(0).to(device)

    for _ in range(max_new_tokens):
        logits, _ = model(context)
        logits = logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)
        context = torch.cat((context, next_token), dim=1)

        context = context[:, -block_size:]

    return decode(context[0].tolist())

# =========================
# TEST
# =========================
while True:
    prompt = input("\nPrompt: ")
    if prompt == "exit":
        break

    result = generate(prompt)
    print("\nOutput:")
    print(result)