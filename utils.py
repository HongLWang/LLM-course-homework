import torch
import re
import os
import json
from torch.utils.data import Dataset

# --- Special Tokens ---
SPECIAL_TOKENS = {
    '<BOS>': 0,    # Beginning of sequence
    '<EOS>': 1,    # End of sequence
    '<PAD>': 2,    # Padding token
    '<UNK>': 3,    # Unknown token
    '<CONTEXT>': 4,  # Context section marker
    '<QUESTION>': 5, # Question section marker
    '<SQL>': 6,      # SQL section marker
}

class Tokenizer:
    """Character-level tokenizer with special tokens for SQL finetuning"""

    def __init__(self, tokenizer_path=None, text_data=None):
        self.SPECIAL_TOKENS = SPECIAL_TOKENS

        if tokenizer_path and os.path.exists(tokenizer_path):
            self._load_tokenizer(tokenizer_path)
        else:
            if text_data is None:
                raise ValueError("Either tokenizer_path or text_data must be provided")
            self._create_tokenizer(text_data)

        # Set special token IDs
        self.BOS_TOKEN = self.stoi['<BOS>']
        self.EOS_TOKEN = self.stoi['<EOS>']
        self.PAD_TOKEN = self.stoi['<PAD>']
        self.UNK_TOKEN = self.stoi['<UNK>']
        self.CONTEXT_TOKEN = self.stoi['<CONTEXT>']
        self.QUESTION_TOKEN = self.stoi['<QUESTION>']
        self.SQL_TOKEN = self.stoi['<SQL>']

    def _load_tokenizer(self, tokenizer_path):
        """Load pretrained tokenizer"""
        tokenizer_data = torch.load(tokenizer_path)
        self.stoi = tokenizer_data['stoi']
        self.itos = tokenizer_data['itos']
        self.vocab_size = tokenizer_data['vocab_size']
        self.SPECIAL_TOKENS = tokenizer_data['special_tokens']
        print(f"Loaded pretrained tokenizer with vocab size: {self.vocab_size}")

    def _create_tokenizer(self, text_data):
        """Create new tokenizer from text data"""
        print("Creating new tokenizer...")
        chars = sorted(list(set(text_data)))

        # Remove special token strings if they exist in the text
        for token in self.SPECIAL_TOKENS.keys():
            if token in chars:
                chars.remove(token)

        # Add special tokens at the beginning
        special_token_chars = list(self.SPECIAL_TOKENS.keys())
        vocab_chars = special_token_chars + chars
        self.vocab_size = len(vocab_chars)

        # Create mappings
        self.stoi = {ch: i for i, ch in enumerate(vocab_chars)}
        self.itos = {i: ch for i, ch in enumerate(vocab_chars)}

    def encode(self, s):
        """Encoder with special token handling"""
        result = []

        # Replace section markers with special tokens
        s = s.replace('[context]', '<CONTEXT>')
        s = s.replace('[question]', '<QUESTION>')
        s = s.replace('[SQL]', '<SQL>')

        for c in s:
            if c in self.stoi:
                result.append(self.stoi[c])
            else:
                result.append(self.UNK_TOKEN)

        return result

    def decode(self, l):
        """Decoder that handles special tokens"""
        result = []
        for i in l:
            if i < len(self.itos):
                token = self.itos[i]
                if token == '<BOS>':
                    continue
                elif token == '<EOS>':
                    break
                elif token == '<PAD>':
                    continue
                elif token == '<UNK>':
                    result.append('?')
                elif token == '<CONTEXT>':
                    result.append('[context]')
                elif token == '<QUESTION>':
                    result.append('[question]')
                elif token == '<SQL>':
                    result.append('[SQL]')
                else:
                    result.append(token)
        return ''.join(result)

    def save(self, path):
        """Save tokenizer to file"""
        torch.save({
            'stoi': self.stoi,
            'itos': self.itos,
            'vocab_size': self.vocab_size,
            'special_tokens': self.SPECIAL_TOKENS
        }, path)



def load_data(data_path):
    """Load training and test data from files"""
    train_file = os.path.join(data_path, "train.txt")
    test_file = os.path.join(data_path, "test.txt")

    with open(train_file, 'r', encoding='utf-8') as f1:
        text = f1.read()

    with open(test_file, 'r', encoding='utf-8') as f2:
        text_test = f2.read()

    return text, text_test




class SQLDataset(Dataset):
    """Dataset class for SQL finetuning"""

    def __init__(self, examples, block_size, tokenizer):
        self.examples = examples
        self.block_size = block_size
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        # Encode input and output
        input_ids = [self.tokenizer.BOS_TOKEN] + self.tokenizer.encode(example['input'])
        output_ids = self.tokenizer.encode(example['output']) + [self.tokenizer.EOS_TOKEN]

        # Combine for full sequence
        full_sequence = input_ids + output_ids

        # Truncate if necessary
        if len(full_sequence) > self.block_size:
            full_sequence = full_sequence[:self.block_size]

        input_seq = full_sequence[:-1]
        target_seq = full_sequence[1:]

        # Pad sequences
        input_length = len(input_seq)
        target_length = len(target_seq)

        if input_length < self.block_size - 1:
            input_seq.extend([self.tokenizer.PAD_TOKEN] * (self.block_size - 1 - input_length))
        if target_length < self.block_size - 1:
            target_seq.extend([self.tokenizer.PAD_TOKEN] * (self.block_size - 1 - target_length))

        # Create attention mask
        attention_mask = [1] * input_length + [0] * (self.block_size - 1 - input_length)

        # Create loss mask
        loss_mask = [0] * len(input_ids) + [1] * len(output_ids)
        if len(loss_mask) > self.block_size - 1:
            loss_mask = loss_mask[:self.block_size - 1]
        if len(loss_mask) < self.block_size - 1:
            loss_mask.extend([0] * (self.block_size - 1 - len(loss_mask)))

        return {
            'input_ids': torch.tensor(input_seq, dtype=torch.long),
            'targets': torch.tensor(target_seq, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.float),
            'loss_mask': torch.tensor(loss_mask, dtype=torch.float)
        }




def process_training_data(text):
    """Process data for pretraining"""
    # Split text into individual examples
    examples = []

    # Split by [context] to get individual entries
    parts = text.split('[context]')[1:]  # Skip first empty part

    for part in parts:
        try:
            # Extract context
            context_match = re.search(r'^(.*?)\[question\]', part, re.DOTALL)
            if not context_match:
                continue
            context = context_match.group(1).strip()

            # Extract question
            question_match = re.search(r'\[question\](.*?)\[SQL\]', part, re.DOTALL)
            if not question_match:
                continue
            question = question_match.group(1).strip()

            # Extract SQL
            sql_match = re.search(r'\[SQL\](.*?)(?=\[context\]|$)', part, re.DOTALL)
            if not sql_match:
                continue
            sql = sql_match.group(1).strip()

            # Reconstruct the example with proper formatting
            example = f"[context] {context} [question] {question} [SQL] {sql}"
            examples.append(example)

        except Exception as e:
            print(f"Error processing training example: {e}")
            continue

    return ' '.join(examples)


def process_training_data_debug(text, chunk_size):
    """Process data for pretraining"""
    # Split text into individual examples
    examples = []

    # Split by [context] to get individual entries
    parts = text.split('[context]')[1:]  # Skip first empty part

    cnt = 0

    for part in parts:
        try:

            if cnt >= chunk_size:
                break

            # Extract context
            context_match = re.search(r'^(.*?)\[question\]', part, re.DOTALL)
            if not context_match:
                continue
            context = context_match.group(1).strip()

            # Extract question
            question_match = re.search(r'\[question\](.*?)\[SQL\]', part, re.DOTALL)
            if not question_match:
                continue
            question = question_match.group(1).strip()

            # Extract SQL
            sql_match = re.search(r'\[SQL\](.*?)(?=\[context\]|$)', part, re.DOTALL)
            if not sql_match:
                continue
            sql = sql_match.group(1).strip()

            # Reconstruct the example with proper formatting
            example = f"[context] {context} [question] {question} [SQL] {sql}"
            examples.append(example)

            cnt += 1


        except Exception as e:
            print(f"Error processing training example: {e}")
            continue



    return ' '.join(examples)




def parse_training_data(text):

    """Parse training data into context+question and SQL pairs"""
    examples = []

    # Split by [context] to get individual entries
    parts = text.split('[context]')[1:]

    for part in parts:
        try:
            # Extract context
            context_match = re.search(r'^(.*?)\[question\]', part, re.DOTALL)
            if not context_match:
                continue
            context = context_match.group(1).strip()

            # Extract question
            question_match = re.search(r'\[question\](.*?)\[SQL\]', part, re.DOTALL)
            if not question_match:
                continue
            question = question_match.group(1).strip()

            # Extract SQL
            sql_match = re.search(r'\[SQL\](.*?)(?=\[context\]|$)', part, re.DOTALL)
            if not sql_match:
                continue
            sql = sql_match.group(1).strip()

            # Create input and output
            input_text = f"[context] {context} [question] {question} [SQL]"
            output_text = f" {sql}"

            examples.append({
                'input': input_text,
                'output': output_text,
                'context': context,
                'question': question,
                'sql': sql
            })

        except Exception as e:
            print(f"Error processing training example: {e}")
            continue

    return examples


def parse_training_data_debug(text, chunk_size):

    """Parse training data into context+question and SQL pairs"""
    examples = []

    # Split by [context] to get individual entries
    parts = text.split('[context]')[1:]

    cnt = 0
    for part in parts:
        try:
            if cnt> chunk_size:
                break
            # Extract context
            context_match = re.search(r'^(.*?)\[question\]', part, re.DOTALL)
            if not context_match:
                continue
            context = context_match.group(1).strip()

            # Extract question
            question_match = re.search(r'\[question\](.*?)\[SQL\]', part, re.DOTALL)
            if not question_match:
                continue
            question = question_match.group(1).strip()

            # Extract SQL
            sql_match = re.search(r'\[SQL\](.*?)(?=\[context\]|$)', part, re.DOTALL)
            if not sql_match:
                continue
            sql = sql_match.group(1).strip()

            # Create input and output
            input_text = f"[context] {context} [question] {question} [SQL]"
            output_text = f" {sql}"

            examples.append({
                'input': input_text,
                'output': output_text,
                'context': context,
                'question': question,
                'sql': sql
            })

            cnt += 1


        except Exception as e:
            print(f"Error processing training example: {e}")
            continue

    return examples



@torch.no_grad()
def estimate_loss(config, model, train_data, val_data):
    """Estimate training and validation loss"""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        data = train_data if split == 'train' else val_data
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, Y = get_batch(config,data)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# Data loading function
def get_batch(config, data):
    ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
    x = torch.stack([data[i:i + config.block_size] for i in ix])
    y = torch.stack([data[i + 1:i + config.block_size + 1] for i in ix])
    x, y = x.to(config.device), y.to(config.device)
    return x, y

def parse_test_data(config, test_file):

    flag = 0
    if config.debug_mode:
        flag = 1
    """Parse test.txt file with [context], [question], [SQL] format"""
    with open(test_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by entries and parse each one
    entries = []
    # Split by [context] to get individual entries
    parts = content.split('[context]')[1:]  # Skip first empty part

    cnt = 0
    for part in parts:
        try:
            if flag:
                if cnt >= config.debug_dataset_size:
                    break

            # Extract context
            context_match = re.search(r'^(.*?)\[question\]', part, re.DOTALL)
            if not context_match:
                continue
            context = context_match.group(1).strip()

            # Extract question
            question_match = re.search(r'\[question\](.*?)\[SQL\]', part, re.DOTALL)
            if not question_match:
                continue
            question = question_match.group(1).strip()

            # Extract SQL
            sql_match = re.search(r'\[SQL\](.*?)(?=\[context\]|$)', part, re.DOTALL)
            if not sql_match:
                continue
            sql = sql_match.group(1).strip()

            entries.append({
                'context': context,
                'question': question,
                'sql': sql
            })

            cnt += 1

        except Exception as e:
            print(f"Error parsing entry: {e}")
            continue

    return entries



def parse_test_data_debug(test_file, chunk_size):
    """Parse test.txt file with [context], [question], [SQL] format"""
    with open(test_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by entries and parse each one
    entries = []
    # Split by [context] to get individual entries
    parts = content.split('[context]')[1:]  # Skip first empty part

    cnt = 0
    for part in parts:
        try:
            if cnt>=chunk_size:
                break
            # Extract context
            context_match = re.search(r'^(.*?)\[question\]', part, re.DOTALL)
            if not context_match:
                continue
            context = context_match.group(1).strip()

            # Extract question
            question_match = re.search(r'\[question\](.*?)\[SQL\]', part, re.DOTALL)
            if not question_match:
                continue
            question = question_match.group(1).strip()

            # Extract SQL
            sql_match = re.search(r'\[SQL\](.*?)(?=\[context\]|$)', part, re.DOTALL)
            if not sql_match:
                continue
            sql = sql_match.group(1).strip()

            entries.append({
                'context': context,
                'question': question,
                'sql': sql
            })

            cnt += 1
        except Exception as e:
            print(f"Error parsing entry: {e}")
            continue

    return entries



def normalize_sql(sql):
    """Normalize SQL query for comparison"""
    sql = sql.strip().upper()
    # Remove extra whitespace
    sql = re.sub(r'\s+', ' ', sql)
    # Remove trailing semicolon if present
    sql = sql.rstrip(';')
    return sql


def calculate_exact_match(predicted, target):
    """Calculate exact match accuracy"""
    return normalize_sql(predicted) == normalize_sql(target)


def calculate_bleu_score(predicted, target):
    """Simple BLEU-like score based on n-gram overlap"""
    pred_tokens = predicted.lower().split()
    target_tokens = target.lower().split()

    if len(pred_tokens) == 0 or len(target_tokens) == 0:
        return 0.0

    # Calculate unigram precision
    pred_unigrams = set(pred_tokens)
    target_unigrams = set(target_tokens)

    intersection = pred_unigrams.intersection(target_unigrams)
    if len(pred_unigrams) == 0:
        return 0.0

    precision = len(intersection) / len(pred_unigrams)
    recall = len(intersection) / len(target_unigrams) if len(target_unigrams) > 0 else 0

    # F1 score
    if precision + recall == 0:
        return 0.0

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1




def generate_sql_from_input_BPE(model, context, question, max_tokens=100):
    """Generate SQL from context and question using GPT tokenizer"""

    # Create input prompt with special tokens
    input_text = f"[context] {context} [question] {question} [SQL] "

    # Encode input using GPT-2 tokenizer
    input_ids = encode(input_text,tokenizer)

    # Convert to tensor
    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(config.device)

    # Generate with enhanced parameters
    with torch.no_grad():
        generated = model.generate(
            input_tensor,
            max_new_tokens=max_tokens,
            temperature=0.7,  # Lower temperature for more deterministic output
            top_k=50,
            stop_at_eos=True
        )

    # Decode generated text
    generated_text = decode(generated[0].tolist(),tokenizer)

    # Extract SQL part - look for [SQL] pattern
    sql_start = generated_text.find('[SQL]')

    if sql_start != -1:
        # Get everything after [SQL]
        sql_part = generated_text[sql_start + 5:].strip()  # 5 is length of '[SQL]'

        # Stop at next section marker or end
        stop_markers = ['[context]', '[question]', '[SQL]']

        for marker in stop_markers:
            if marker in sql_part:
                sql_part = sql_part[:sql_part.find(marker)].strip()
                break

        # Clean up any remaining artifacts
        sql_part = sql_part.strip()

        # Remove any trailing punctuation that might be artifacts
        while sql_part and sql_part[-1] in '.,;':
            sql_part = sql_part[:-1].strip()

        return sql_part

    return "ERROR: No SQL found in generation"





def generate_sql_from_input_CL(model, tokenizer, context, question, device, max_tokens=100, block_size=None):
    """Generate SQL from context and question using special tokens"""
    # Create input prompt with special tokens
    input_text = f"[context] {context} [question] {question} [SQL] "

    # Encode input (this will add BOS token automatically)
    input_ids = tokenizer.encode(input_text)

    # Remove the EOS token that encode() adds since we want to continue generating
    if input_ids[-1] == tokenizer.EOS_TOKEN:
        input_ids = input_ids[:-1]

    # Convert to tensor
    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)

    # Generate with enhanced parameters
    with torch.no_grad():
        generated = model.generate(
            input_tensor,
            max_new_tokens=max_tokens,
            temperature=0.8,
            top_k=40,
            stop_at_eos=True,
            eos_token=tokenizer.EOS_TOKEN,
            block_size=block_size
        )

    # Decode generated text
    generated_text = tokenizer.decode(generated[0].tolist())

    # Extract SQL part - look for both [SQL] and <SQL> patterns
    sql_patterns = ['[SQL]', '<SQL>']
    sql_start = -1
    sql_marker = None

    for pattern in sql_patterns:
        pos = generated_text.find(pattern)
        if pos != -1:
            sql_start = pos
            sql_marker = pattern
            break

    if sql_start != -1:
        sql_part = generated_text[sql_start + len(sql_marker):].strip()

        # Stop at next section marker (both original and special token formats)
        stop_markers = ['[context]', '[question]', '[SQL]', '<CONTEXT>', '<QUESTION>', '<SQL>']

        for marker in stop_markers:
            if marker in sql_part:
                sql_part = sql_part[:sql_part.find(marker)].strip()
                break

        return sql_part

    # If no SQL marker found, return the part after the input
    input_without_bos_eos = input_text.strip()
    if input_without_bos_eos in generated_text:
        after_input = generated_text[generated_text.find(input_without_bos_eos) + len(input_without_bos_eos):].strip()

        # Stop at any section marker
        stop_markers = ['[context]', '[question]', '[SQL]', '<CONTEXT>', '<QUESTION>', '<SQL>']
        for marker in stop_markers:
            if marker in after_input:
                after_input = after_input[:after_input.find(marker)].strip()
                break

        return after_input

    return generated_text

def save_evaluation_results(results, output_dir):
    """Save evaluation results to JSON file"""
    with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)


def extract_sql_from_generated(generated_text):
    """Extract only the SQL part from generated text"""
    # Find the [SQL] marker
    if '[SQL]' in generated_text:
        # Split at [SQL] and take everything after it
        sql_part = generated_text.split('[SQL]', 1)[1]

        # Stop at any of these markers that might appear after SQL
        stop_markers = ['[context]', '[question]', '[SQL]']

        for marker in stop_markers:
            if marker in sql_part:
                sql_part = sql_part.split(marker)[0]

        # Clean up the SQL
        sql_part = sql_part.strip()

        # Return with [SQL] prefix
        return f"[SQL] {sql_part}" if sql_part else "[SQL]"

    return generated_text.strip()