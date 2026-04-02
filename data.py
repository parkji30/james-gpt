import datasets
from transformers import GPT2Tokenizer
import torch

dataset = datasets.load_dataset(
    "mlfoundations/dclm-baseline-1.0", 
    split='train', 
    streaming=True
)

TOKENIZER = GPT2Tokenizer.from_pretrained("openai-community/gpt2")

def stream_fixed_context_length(context_length:int =256):
    token_buffer = []

    for text in dataset:        
        token_ids = TOKENIZER.encode(text, add_special_tokens=False)

        # If there's nothing to tokenize, we just continue
        if not token_ids:
            continue 

        token_buffer.extend(token_ids)
        token_buffer.append(TOKENIZER.eos_token_id)   

        while len(token_buffer) >= context_length + 1:
            block = token_buffer[: context_length + 1]

            del token_buffer[: context_length + 1]

        x = torch.tensor(block[:-1], dtype=torch.long)