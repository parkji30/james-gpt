import torch
from torch import optim

from model import GPT
from config import GPT_CONFIG


def cross_entropy(logits, targets):
    logits_scaled = logits - torch.max(logits, dim=-1, keepdim=True).values # for softmax stab since exp(0) = 1
    probs = torch.softmax(logits_scaled, dim=-1)
    target_probs = probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1) # we want a (B, T) vector
    return -torch.log(target_probs).mean()


def train_test_split(train_portion = 0.7):
    with open("tokenized_text.txt", "r", encoding="utf-8") as f:
        data = torch.tensor([int(token) for token in f.read().split()], dtype=torch.long)

    data_size = len(data)

    train_set = data[:int(data_size*train_portion)]
    test_set = data[int(data_size*train_portion):]

    return train_set, test_set


def get_batch(text_slice, batch_size, context_window):
    max_start = len(text_slice) - context_window
    start_indices = torch.randint(0, max_start, (batch_size, )) # return 32, 1

    x_idx = start_indices[:, None] + torch.arange(context_window)
    y_idx = x_idx + 1 # shift by 1 to right.

    x = text_slice[x_idx]
    y = text_slice[y_idx]

    # x = torch.stack([
    #     text_slice[i:i+context_window] for i in start_indices.tolist()
    # ])

    # y = torch.stack([
    #     text_slice[i+1: i+context_window + 1] for i in start_indices.tolist()
    # ])

    return start_indices, x, y


train_set, test_set = train_test_split()

EPOCHS = 1000
STEPS = 10000
BATCH_SIZE = 32
CONTEXT_WINDOW = GPT_CONFIG["context_length"]

model = GPT(**GPT_CONFIG)
optimizer = optim.AdamW(model.parameters())

for ep in range(EPOCHS):
    for step in range(STEPS):
        batch_indices, train_batch, train_targets = get_batch(
            train_set,
            BATCH_SIZE,
            CONTEXT_WINDOW,
        )

        # nuke old gradients
        optimizer.zero_grad()

        # forward pass
        logits = model(train_batch) # return raw logits before softmax output
        
        # calculate loss
        loss = cross_entropy(logits=logits, targets=train_targets)

        # calculate gradients
        loss.backward()

        # update weights
        optimizer.step()