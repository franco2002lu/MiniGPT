import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size = 12
block_size = 12
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

torch.manual_seed(1337)

# read training file
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# get the characters in the file and define encoding/decoding functions
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
iots = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]  # outputs list of integers
decode = lambda lst: ''.join([iots[i] for i in lst])  # outputs string

# defining training and validation data with a 90/10 split
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
validation_data = data[n:]


# gets a batch of data and separating them into train and validation data, with x predicting y
def get_batch(split):
    batch_data = train_data if split == 'train' else validation_data
    ix = torch.randint(len(batch_data) - block_size, (batch_size,))
    x = torch.stack([batch_data[i: i + block_size] for i in ix])
    y = torch.stack([batch_data[i + 1: i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


# estimates the loss function and trains the model
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


model = BigramLanguageModel(vocab_size)
m = model.to(device)

# pytorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: training loss {losses['train']:.4f}, value loss {losses['val']:.4f}")

    # taking a batch of data

    x_batch, y_batch = get_batch('train')
    logits, loss = model(x_batch, y_batch)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))