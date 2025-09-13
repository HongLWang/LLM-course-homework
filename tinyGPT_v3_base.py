import torch
import os
import json
from Configuration import Config3
from tinyGPT_v2_base import set_tokenizer,save_model_and_tokenizer, train_model
from utility import load_dataset_from_HuggingFace
from model import GPTv3Base

'''
for gpt3-base, only pretrain, no evaluation on the language2sql task because it does not perform well.
'''

def encode(s,tokenizer):
    """Encoder using GPT-2 tokenizer"""
    tokens = tokenizer.encode(s, add_special_tokens=True)
    return tokens


def decode(tokens, tokenizer):
    """Decoder using GPT-2 tokenizer"""
    text = tokenizer.decode(tokens, skip_special_tokens=False)
    return text.strip()




def main(config):
    torch.manual_seed(1337)

    # init tokenizer, print tokenizer information
    tokenizer = set_tokenizer(config)
    vocab_size = len(tokenizer)

    # load dataset from hugging face
    combined_text = load_dataset_from_HuggingFace(config)


    # Process the text data, th4n # Train and validation splits
    print("Encoding text data...")
    data = torch.tensor(encode(combined_text,tokenizer), dtype=torch.long)

    if not config.debug_mode:
        n = int(0.9 * len(data))
    else:
        n = 128

    train_data = data[:n]
    val_data = data[n:]

    print(f"Training data size: {len(train_data)} tokens")
    print(f"Validation data size: {len(val_data)} tokens")


    # --- Training Loop ---
    model = GPTv3Base(
        vocab_size=vocab_size,
        n_embd=config.n_embd,
        n_head=config.n_head,
        n_layer=config.n_layer,
        block_size=config.block_size,
        dropout=config.dropout,
        pad_token_id=config.PAD_TOKEN,
        eos_token_id=config.EOS_TOKEN
    )

    m = model.to(config.device)
    print(f"{sum(p.numel() for p in m.parameters()) / 1e6:.2f}M parameters")

    train_model(config, model, train_data, val_data)
    save_model_and_tokenizer(model, tokenizer, config)


if __name__ == '__main__':
    config = Config3(debug_mode=False, fine_tune_mode=False)
    config.tokenizer_init = True
    main(config)