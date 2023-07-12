import torch
import torch.nn as nn
from torch.nn import functional as F

# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

with open('input.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# Hyperparameters
batch_size = 32 # 19 for 19 gpu cores? see if i can further optimize this
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu' # see if there's a gpu option
eval_iters = 200

torch.manual_seed(1337)

# Set of unique characters in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# Mapping of integers
stoi = { ch: i for i, ch in enumerate(chars)} # list of characters and corresponding integers 
itos = { i: ch for i, ch in enumerate(chars)} # list of integers and corresponding characters 
encode = lambda s: [stoi[c] for c in s] # take string, output list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # take list of integers, output string

# For training
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n] # train dataset, all data before nth index
val_data = data[n:] # test dataset, all data after nth index

# data loading
def get_batch(split):
    # generates inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # (batch size) no. random offsets between block size and produces 4 random sequences from the data
    # min val block size because uses data[:block_size]
    x = torch.stack([data[i:i+block_size] for i in ix]) # makes a 4 by 8 tensor (rowsxcolumns). 4 blocks
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # offset by one
    x, y = x.to(device), y.to(device)

    return x, y

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

    def forward(self, idx, targets= None):
        # idx and targets are (B, T) tensors of integers
        logits = self.token_embedding_table(idx)
        if targets == None:
            loss = None
        else:
            # Reshaping logits and targets into format that pytorch wants, i don't know how to read documentation so this is copying andrej
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
        
            # evaluating loss 
            loss = F.cross_entropy(logits, targets) # evaluates how close the logits are to the targets

        return logits, loss
    
    # generate
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in current context
        for _ in range(max_new_tokens):
            logits, loss = self(idx)# get predictions
            logits = logits[:, -1, :] # focus only on last time step
            probs = F.softmax(logits, dim=-1) # softmax to get probabilities
            idx_next = torch.multinomial(probs, num_samples=1) # sample from distribution
            idx = torch.cat((idx, idx_next), dim=1) # append sampled index to running sequence
        return idx

model = BigramLanguageModel(vocab_size)
m = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f'Step {iter}: train loss {losses["train"]:.4f}, val loss {losses["val"]:.4f}')

    # get batch
    xb, yb = get_batch('train')

    # evaluate loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))