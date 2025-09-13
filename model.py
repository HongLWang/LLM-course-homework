import torch
import torch.nn as nn
from torch.nn import functional as F
import re
import os
import json
from torch.utils.data import Dataset

'''
The transformer model (Head+MultiheadAttention+FeedForward+block) is implemented by Karpathy at repo https://github.com/karpathy/nanoGPT
All models are based on the v1base model.
'''



class Head(nn.Module):
    """
    One head of self-attention, the key part is the QKV matrix multiplication
    This is decoder only causual self-attention. i.e., a token only access itself and tokens before it, to avoid information leaking in the training steps.
    """
    def __init__(self, n_embd, head_size, block_size, dropout):

        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape # B=batch size, T = token length, C=token embedding dimension
        k = self.key(x) # Key, K
        q = self.query(x) # Query,Q
        wei = q @ k.transpose(-2, -1) * C ** -0.5 # Value, V
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # masking tokens after current token
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    """ Multiple heads of self-attention  """

    def __init__(self, n_embd, num_heads, head_size, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(n_embd, head_size, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # parallelzation, concat
        # print('dimension of out after concatnation is ', out.shape)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """ linear + ReLU + liner + dropout """

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer blocks"""

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_embd, n_head, head_size, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):  # residual connection
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x




class GPTLanguageModel(nn.Module):  # for V1base
    """ The full GPT-style Language Model with special token support """

    def __init__(self, vocab_size=64, n_embd=320, n_head=5, n_layer=8, block_size=256, dropout=0.2,
                 pad_token_id=2, eos_token_id=1):
        super().__init__()
        self.block_size = block_size

        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head, block_size=block_size, dropout=dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def calculate_loss(self, logits, targets):
        # Apply different loss weights for special tokens
        loss_weights = torch.ones_like(targets, dtype=torch.float)
        # Give higher weight to EOS token to encourage proper stopping
        loss_weights[targets == self.eos_token_id] = 2.0
        # Lower weight for padding tokens
        loss_weights[targets == self.pad_token_id] = 0.1
        loss = F.cross_entropy(logits, targets, reduction='none')
        loss = (loss * loss_weights).mean()

        return loss


    def stop_condition(self, stop_at_eos, idx_next, eos_token):
        return stop_at_eos and idx_next.item() == eos_token

    def forward(self, idx, targets=None):
        B, T = idx.shape
        device = idx.device

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)

            loss = self.calculate_loss(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, stop_at_eos=True,
                 eos_token=1, block_size=None):

        if block_size is None:
            block_size = self.block_size

        for _ in range(max_new_tokens):

            # print('size of input tensor is ', idx.shape)
            # print('block_size is', block_size)
            # Crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]

            # Get predictions
            logits, loss = self(idx_cond)

            # Focus only on the last time step
            logits = logits[:, -1, :] / temperature

            # Apply top-k filtering if specified
            if top_k is not None:
                top_k = min(top_k, logits.size(-1))
                # Remove all tokens with a probability less than the top-k tokens
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')

            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)

            # Stop if we hit EOS token
            if self.stop_condition(stop_at_eos, idx_next, eos_token):
                break

        return idx



class GPTv23(GPTLanguageModel):  # for v2-fine tune and v3-finetune, they share the same struture (different parameters)

    def __init__(self, vocab_size=64, n_embd=320, n_head=5, n_layer=8, block_size=256, dropout=0.2,
                 pad_token_id=2, eos_token_id=1, SQL_TOKEN=None,EOS_TOKEN=None,QUESTION_TOKEN=None,CONTEXT_TOKEN=None, PAD_TOKEN = None):
        super().__init__(vocab_size, n_embd, n_head, n_layer, block_size, dropout,
                 pad_token_id, eos_token_id)
        self.SQL_TOKEN = SQL_TOKEN
        self.EOS_TOKEN = EOS_TOKEN
        self.QUESTION_TOKEN = QUESTION_TOKEN
        self.PAD_TOKEN = PAD_TOKEN
        self.CONTEXT_TOKEN = CONTEXT_TOKEN



    def calculate_loss(self, logits, targets):
        # Apply loss masking for fine-tuning
        loss_weights = torch.ones_like(targets, dtype=torch.float)
        # Don't compute loss on padding tokens
        loss_weights[targets == self.PAD_TOKEN] = 0.0
        # Higher weight for SQL tokens during fine-tuning
        loss_weights[targets == self.SQL_TOKEN] = 2.0
        loss_weights[targets == self.EOS_TOKEN] = 2.0

        loss = F.cross_entropy(logits, targets, reduction='none')
        loss = (loss * loss_weights).sum() / loss_weights.sum()

        return loss

    def stop_condition(self, stop_at_eos, idx_next, eos_token):

        sql_started = False

        # Check if we've started generating SQL
        if idx_next.item() == self.SQL_TOKEN:
            sql_started = True

        # Stop generation if we hit certain tokens after SQL has started
        if stop_at_eos and sql_started:
            if (idx_next.item() == self.EOS_TOKEN or
                    idx_next.item() == self.CONTEXT_TOKEN or
                    idx_next.item() == self.QUESTION_TOKEN):
                return True

        return False


class GPTv2Base(nn.Module):  # for v2base

    def __init__(self, vocab_size, n_embd, block_size, n_head, n_layer, dropout,
                 SQL_TOKEN, EOS_TOKEN, QUESTION_TOKEN, CONTEXT_TOKEN, PAD_TOKEN):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # Store special tokens
        self.SQL_TOKEN = SQL_TOKEN
        self.EOS_TOKEN = EOS_TOKEN
        self.QUESTION_TOKEN = QUESTION_TOKEN
        self.CONTEXT_TOKEN = CONTEXT_TOKEN
        self.PAD_TOKEN = PAD_TOKEN

        # Initialize weights properly for the new vocabulary size
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)

            # Apply different loss weights for special tokens (same as original)
            loss_weights = torch.ones_like(targets, dtype=torch.float)
            loss_weights[targets == self.CONTEXT_TOKEN] = 1.5
            loss_weights[targets == self.QUESTION_TOKEN] = 1.5
            loss_weights[targets == self.SQL_TOKEN] = 1.5
            loss_weights[targets == self.EOS_TOKEN] = 2.0
            loss_weights[targets == self.PAD_TOKEN] = 0.1

            loss = F.cross_entropy(logits, targets, reduction='none')
            loss = (loss * loss_weights).mean()

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, stop_at_eos=True):
        """Enhanced generation with temperature, top-k sampling, and special token stopping"""
        for _ in range(max_new_tokens):
            # Crop idx to the last block_size tokens
            idx_cond = idx[:, -self.position_embedding_table.num_embeddings:]

            # Get predictions
            logits, loss = self(idx_cond)

            # Focus only on the last time step
            logits = logits[:, -1, :] / temperature

            # Apply top-k filtering if specified
            if top_k is not None:
                top_k = min(top_k, logits.size(-1))
                # Remove all tokens with a probability less than the top-k tokens
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')

            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)

            # Stop if we hit EOS token or context token (which indicates new example)
            if stop_at_eos and (idx_next.item() == self.EOS_TOKEN or idx_next.item() == self.CONTEXT_TOKEN):
                break

        return idx


class GPTv3Base(GPTLanguageModel):
    """ The full GPT-style Language Model with GPT-2 tokenizer support """

    def calculate_loss(self,logits, targets):
        loss = F.cross_entropy(logits, targets)
        return loss

    def stop_condition(self, stop_at_eos, idx_next, eos_token):
        return False



class GPTLanguageModel4Finetune(GPTLanguageModel): # for v1 base

    def forward(self, idx, targets=None, loss_mask=None):
        B, T = idx.shape
        device = idx.device

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)

            # Calculate loss
            loss = F.cross_entropy(logits_flat, targets_flat, reduction='none')

            # Apply loss mask if provided (only calculate loss on output tokens)
            if loss_mask is not None:
                loss_mask_flat = loss_mask.view(B * T)
                loss = loss * loss_mask_flat
                # Average only over non-masked tokens
                loss = loss.sum() / (loss_mask_flat.sum() + 1e-8)
            else:
                loss = loss.mean()

        return logits, loss

    def load_pretrained_weights(self, weights_path):
        """Load pretrained weights if available"""
        if os.path.exists(weights_path):
            self.load_state_dict(torch.load(weights_path))
            print("Loaded pretrained weights successfully!")
            return True
        else:
            print("No pretrained weights found. Training from scratch...")
            return False

    def set_eos_token(self, eos_token):
        """Set the EOS token for generation stopping"""
        self.eos_token = eos_token