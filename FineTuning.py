import re

# --- Data Processing for Fine-tuning ---
def encode(s, tokenizer):
    """Encoder using GPT-2 tokenizer with special token handling"""
    s = s.replace('[context]', '<CONTEXT>')
    s = s.replace('[question]', '<QUESTION>')
    s = s.replace('[SQL]', '<SQL>')
    tokens = tokenizer.encode(s, add_special_tokens=False)
    return tokens


def decode(tokens,tokenizer):
    """Decoder using GPT-2 tokenizer that handles special tokens"""
    text = tokenizer.decode(tokens, skip_special_tokens=False)
    text = text.replace('<CONTEXT>', '[context]')
    text = text.replace('<QUESTION>', '[question]')
    text = text.replace('<SQL>', '[SQL]')
    text = text.replace('<|endoftext|>', '')
    text = text.replace('<|startoftext|>', '')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()