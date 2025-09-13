import torch
import os, re
from transformers import GPT2Tokenizer
from torch.utils.data import Dataset, DataLoader
import random
from FineTuning import encode, decode
from model import GPTv23 as GPTLanguageModel
from Configuration import Config2
from utils import extract_sql_from_generated,calculate_exact_match,calculate_bleu_score
from tinyGPT_v2_base import set_tokenizer


def print_sample_result(test_input,expected,model,tokenizer, config):
    input_tokens = encode(test_input, tokenizer)
    input_tensor = torch.tensor(input_tokens, dtype=torch.long).unsqueeze(0).to(config.device)

    with torch.no_grad():
        generated = model.generate(
            input_tensor,
            max_new_tokens=100,
            temperature=0.3,  # Low temperature for more deterministic output
            top_k=40,
            stop_at_eos=True
        )  # 100 0.3 40; 50 0.3 20

    generated_text = decode(generated[0].tolist(), tokenizer)
    # Extract just the SQL part
    if '[SQL]' in generated_text:
        generated_text = generated_text.split('[SQL]')[-1].strip()

    expected_sql = expected[6:].strip()

    exact_match = calculate_exact_match(generated_text, expected_sql)
    bleu_score = calculate_bleu_score(generated_text, expected_sql)

    print(f"Input: {test_input}")
    print(f"Expected: {expected_sql}")
    print(f"Predicted_sql: {generated_text}")
    print(f"Exact match: {exact_match}")
    print(f"BLEU-score: {bleu_score}")

# --- Fine-tuning Dataset ---
class SQLDataset(Dataset):
    def __init__(self, config,data_samples, tokenizer, block_size):
        self.data_samples = data_samples
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.config = config

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, idx):
        sample = self.data_samples[idx]

        # Use the full text if available, otherwise format the parts
        if 'full_text' in sample:
            formatted_text = sample['full_text']
        else:
            formatted_text = f"{sample['context']} {sample['question']} {sample['sql']}"

        # Encode the text
        tokens = encode(formatted_text,self.tokenizer)

        # Pad or truncate to config.block_size
        if len(tokens) > self.block_size:
            tokens = tokens[:self.block_size]
        else:
            tokens = tokens + [self.config.PAD_TOKEN] * (self.block_size - len(tokens))

        # Create input (x) and target (y) tensors
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)

        return x, y


def load_data_from_file(file_path):


    """Load training data from text file where each line contains context, question, and SQL"""
    training_samples = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                # Parse the line to extract context, question, and SQL
                # Expected format: [context] ... [question] ... [SQL] ...
                try:
                    # Find the positions of the markers
                    context_start = line.find('[context]')
                    question_start = line.find('[question]')
                    sql_start = line.find('[SQL]')

                    if context_start == -1 or question_start == -1 or sql_start == -1:
                        print(f"Warning: Line {line_num} doesn't contain all required markers, skipping")
                        continue

                    # Extract each part
                    context_part = line[context_start:question_start].strip()
                    question_part = line[question_start:sql_start].strip()
                    sql_part = line[sql_start:].strip()

                    sample = {
                        'context': context_part,
                        'question': question_part,
                        'sql': sql_part,
                        'full_text': line  # Keep the original line for reference
                    }

                    training_samples.append(sample)

                except Exception as e:
                    print(f"Error parsing line {line_num}: {e}")
                    print(f"Line content: {line}")
                    continue

        print(f"Successfully loaded {len(training_samples)} samples from {file_path}")

    except FileNotFoundError:
        print(f"Error: File {file_path} not found!")
        return []
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return []

    return training_samples

@torch.no_grad()
def estimate_loss(model, config, data_loader, eval_iters):
    """Estimate loss on evaluation data"""
    model.eval()
    losses = torch.zeros(eval_iters)

    for k, (xb, yb) in enumerate(data_loader):
        if k >= eval_iters:
            break
        xb, yb = xb.to(config.device), yb.to(config.device)
        logits, loss = model(xb, yb)
        losses[k] = loss.item()

    model.train()
    return losses.mean()


def load_datasets(config, tokenizer):

    if config.debug_mode:
        load_limit = 64
    else:
        load_limit = -1

    # Create training data
    print("Loading training data from train.txt...")
    training_data = load_data_from_file('./sql_data/train.txt')[:load_limit]

    if not training_data:
        print("No training data loaded! Please check that train.txt exists and is properly formatted.")
        exit(1)

    # Load test data if available
    print("Loading test data from test.txt...")
    test_data = load_data_from_file('./sql_data/test.txt')[:load_limit]

    if not test_data:
        print("Warning: No test data loaded from test.txt. Using training data for validation.")
        test_data = training_data[:min(50, len(training_data))]  # Use first 50 samples for testing

    # Create dataset and data loader
    train_dataset = SQLDataset(config, training_data, tokenizer, config.block_size)
    train_loader = DataLoader( train_dataset, batch_size=config.batch_size, shuffle=True)

    # Create validation dataset from test data
    val_dataset = SQLDataset(config,test_data, tokenizer, config.block_size)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    return  train_loader, val_loader,training_data, test_data


def print_example(model, config, iter_num, tokenizer, test_data):
    # Test generation with a sample from test data
    if test_data:
        test_sample = test_data[iter_num]
        # Create test input by removing the SQL part
        test_input = f"{test_sample['context']} {test_sample['question']}"
        expected_sql = test_sample['sql']
    else:
        test_input = "[context] CREATE TABLE farm (Cows INTEGER) [question] What are the maximum and minimum number of cows across all farms."
        expected_sql = "[SQL] SELECT MAX(Cows), MIN(Cows) FROM farm"

    input_tokens = encode(test_input, tokenizer)
    input_tensor = torch.tensor(input_tokens, dtype=torch.long).unsqueeze(0).to(config.device)

    with torch.no_grad():
        generated = model.generate(
            input_tensor,
            max_new_tokens=100,
            temperature=0.3,
            top_k=40,
            stop_at_eos=True
        )

    generated_text = decode(generated[0].tolist(), tokenizer)
    clean_sql = extract_sql_from_generated(generated_text)

    print(f"Input: {test_input}")
    print(f"Generated: {clean_sql}")
    if test_data:
        print(f"Expected: {expected_sql}")
    print("-" * 80)


def save_model_n_tokenizer(model, tokenizer, config):
    # Save fine-tuned model
    print("Saving fine-tuned model...")
    finetuned_model_path = config.output_dir
    # os.makedirs(finetuned_model_path, exist_ok=True)

    # Save model weights
    torch.save(model.state_dict(), os.path.join(finetuned_model_path, 'finetuned_model.pth'))

    # Save tokenizer
    tokenizer.save_pretrained(finetuned_model_path)

    print(f"Fine-tuned model saved to {finetuned_model_path}")


def _testing_n_print(test_cases, model, tokenizer,config):

    # Final test

    print("---------------------------------------------")
    print("FINAL TEST")
    print("-------------------------------------" )


    model.eval()

    exact_matches = 0
    bleu_scores = []

    for i, (test_input, expected) in enumerate(test_cases):

        input_tokens = encode(test_input, tokenizer)
        input_tensor = torch.tensor(input_tokens, dtype=torch.long).unsqueeze(0).to(config.device)

        with torch.no_grad():
            generated = model.generate(
                input_tensor,
                max_new_tokens=100,
                temperature=0.3,  # Low temperature for more deterministic output
                top_k=40,
                stop_at_eos=True
            )  # 100 0.3 40; 50 0.3 20

        generated_text = decode(generated[0].tolist(), tokenizer)
        # Extract just the SQL part
        if '[SQL]' in generated_text:
            generated_text = generated_text.split('[SQL]')[-1].strip()

        expected_sql = expected[6:].strip()

        exact_match = calculate_exact_match(generated_text, expected_sql)
        bleu_score = calculate_bleu_score(generated_text, expected_sql)

        print(f"\nTest Case {i + 1}:")
        print(f"Input: {test_input}")
        print(f"Expected: {expected_sql}")
        print(f"Predicted_sql: {generated_text}")
        print(f"Exact match: {exact_match}")
        print(f"BLEU-score: {bleu_score}")

        if calculate_exact_match(generated_text, expected_sql):
            exact_matches += 1
        bleu_scores.append(bleu_score)

        # Calculate metrics for all test entries

    # Print final results
    print("\n--- Final Results ---")

    print(f"Exact match accuracy: {exact_matches}/{len(test_cases)} = {exact_matches / len(test_cases) * 100:.2f}%")
    print(f"Average BLEU score: {sum(bleu_scores) / len(test_cases):.4f}")
    print(f"Median BLEU score: {sorted(bleu_scores)[len(test_cases) // 2]:.4f}")

    # Additional statistics
    high_bleu = sum(1 for score in bleu_scores if score > 0.5)
    print(f"Examples with BLEU > 0.5: {high_bleu}/{len(test_cases)} = {high_bleu / len(test_cases) * 100:.2f}%")

    # Save results
    results = {
        'exact_match_accuracy': exact_matches / len(test_cases),
        'average_bleu': sum(bleu_scores) / len(bleu_scores),
        'median_bleu': sorted(bleu_scores)[len(bleu_scores) // 2],
        'high_bleu_count': high_bleu,
        'total_examples': len(test_cases)
    }

    import json


    with open(os.path.join(config.output_dir, 'finetune_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nEvaluation results saved to {config.output_dir}/finetune_results.json")




def main(config):

    # load tokenizer
    tokenizer = set_tokenizer(config)
    vocab_size = len(tokenizer)
    print(f"Loaded tokenizer with vocab size: {vocab_size}")
    print(f"Special token IDs - CONTEXT: {config.CONTEXT_TOKEN}, QUESTION: {config.QUESTION_TOKEN}, SQL: {config.SQL_TOKEN}")

    # Set random seeds for reproducibility
    torch.manual_seed(1337)
    random.seed(1337)


    # Load pre-trained model
    print("Loading pre-trained model...")
    model = GPTLanguageModel(vocab_size=vocab_size, n_embd= config.n_embd,
                             n_head=config.n_head, n_layer=config.n_layer, block_size=config.block_size,
                             dropout=config.dropout, SQL_TOKEN=config.SQL_TOKEN,EOS_TOKEN=config.EOS_TOKEN,
                             QUESTION_TOKEN=config.QUESTION_TOKEN, CONTEXT_TOKEN= config.CONTEXT_TOKEN, PAD_TOKEN=config.PAD_TOKEN)

    model.load_state_dict(torch.load(os.path.join(config.output_dir, 'model_weights.pth')))
    model = model.to(config.device)

    train_loader, val_loader,training_data, test_data = load_datasets(config,tokenizer)

    test_cases = []
    for i in range(len(test_data)):
        sample = test_data[i]
        test_input = f"{sample['context']} {sample['question']}"
        expected_sql = sample['sql']
        test_cases.append((test_input, expected_sql))

    # Set up optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    print(f"Starting fine-tuning with {len(training_data)} samples...")
    print(f"config.device: {config.device}")

    need_train = True
    if need_train:
        # Fine-tuning loop
        model.train()
        for iter_num in range(config.max_iters):
            # Training step
            train_step_cnt = 0
            for xb, yb in train_loader:
                train_step_cnt += 1
                if train_step_cnt % 50 == 0:
                    print(f'training step {train_step_cnt} in iter {iter_num}')
                xb, yb = xb.to(config.device), yb.to(config.device)
                logits, loss = model(xb, yb)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            if iter_num % 5 == 0:
                print(f"Step {iter_num}: train loss {loss.item():.4f}")
                test_sample_ids = torch.randint(0, len(test_cases), (2,))
                for id in test_sample_ids:
                    textt, expected = test_cases[id]
                    print_sample_result(textt, expected, model, tokenizer, config)
        save_model_n_tokenizer(model, tokenizer, config)
    else:
        model = GPTLanguageModel(vocab_size=vocab_size, n_embd=config.n_embd,
                                 n_head=config.n_head, n_layer=config.n_layer, block_size=config.block_size,
                                 dropout=config.dropout, SQL_TOKEN=config.SQL_TOKEN, EOS_TOKEN=config.EOS_TOKEN,
                                 QUESTION_TOKEN=config.QUESTION_TOKEN, CONTEXT_TOKEN=config.CONTEXT_TOKEN,
                                 PAD_TOKEN=config.PAD_TOKEN)

        model.load_state_dict(torch.load(os.path.join(config.output_dir, 'finetuned_model.pth')))
        model = model.to(config.device)

    _testing_n_print(test_cases, model, tokenizer,config)


def main_test(config):

    # load tokenizer
    tokenizer = set_tokenizer(config)
    vocab_size = len(tokenizer)
    print(f"Loaded tokenizer with vocab size: {vocab_size}")
    print(f"Special token IDs - CONTEXT: {config.CONTEXT_TOKEN}, QUESTION: {config.QUESTION_TOKEN}, SQL: {config.SQL_TOKEN}")

    # Set random seeds for reproducibility
    torch.manual_seed(1337)
    random.seed(1337)


    # Load pre-trained model
    print("Loading pre-trained model...")
    model = GPTLanguageModel(vocab_size=vocab_size, n_embd= config.n_embd,
                             n_head=config.n_head, n_layer=config.n_layer, block_size=config.block_size,
                             dropout=config.dropout, SQL_TOKEN=config.SQL_TOKEN,EOS_TOKEN=config.EOS_TOKEN,
                             QUESTION_TOKEN=config.QUESTION_TOKEN, CONTEXT_TOKEN= config.CONTEXT_TOKEN, PAD_TOKEN=config.PAD_TOKEN)

    model.load_state_dict(torch.load(os.path.join(config.output_dir, 'model_weights.pth')))
    model = model.to(config.device)

    train_loader, val_loader,training_data, test_data = load_datasets(config,tokenizer)

    test_cases = []
    for i in range(len(test_data)):
        sample = test_data[i]
        test_input = f"{sample['context']} {sample['question']}"
        expected_sql = sample['sql']
        test_cases.append((test_input, expected_sql))


    model = GPTLanguageModel(vocab_size=vocab_size, n_embd=config.n_embd,
                             n_head=config.n_head, n_layer=config.n_layer, block_size=config.block_size,
                             dropout=config.dropout, SQL_TOKEN=config.SQL_TOKEN, EOS_TOKEN=config.EOS_TOKEN,
                             QUESTION_TOKEN=config.QUESTION_TOKEN, CONTEXT_TOKEN=config.CONTEXT_TOKEN,
                             PAD_TOKEN=config.PAD_TOKEN)

    checkpoint_path = os.path.join(config.output_dir, 'finetuned_model.pth')
    state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)

    _testing_n_print(test_cases, model, tokenizer,config)



def Natrual_language_chat(config):

    # load tokenizer
    tokenizer = set_tokenizer(config)
    vocab_size = len(tokenizer)
    print(f"Loaded tokenizer with vocab size: {vocab_size}")
    print(f"Special token IDs - CONTEXT: {config.CONTEXT_TOKEN}, QUESTION: {config.QUESTION_TOKEN}, SQL: {config.SQL_TOKEN}")

    # Set random seeds for reproducibility
    torch.manual_seed(1337)
    random.seed(1337)

    model = GPTLanguageModel(vocab_size=vocab_size, n_embd=config.n_embd,
                             n_head=config.n_head, n_layer=config.n_layer, block_size=config.block_size,
                             dropout=config.dropout, SQL_TOKEN=config.SQL_TOKEN, EOS_TOKEN=config.EOS_TOKEN,
                             QUESTION_TOKEN=config.QUESTION_TOKEN, CONTEXT_TOKEN=config.CONTEXT_TOKEN,
                             PAD_TOKEN=config.PAD_TOKEN)


    checkpoint_path = os.path.join(config.output_dir, 'finetuned_model.pth')
    state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)

    interactive_chat(config, model, tokenizer)




def interactive_chat(config, model, tokenizer):

    # --- Interactive Testing ---
    print("\n--- Interactive Testing ---")
    print("You can now test the model with custom inputs!")
    print("Format: [context] <your context> [question] <your question>")
    print("Type 'quit' to exit")

    while True:
        try:
            user_input = input("\nEnter your test input: ")
            if user_input.lower() == 'quit':
                break

            test_input = parse_user_input(user_input)

            input_tokens = encode(test_input, tokenizer)
            input_tensor = torch.tensor(input_tokens, dtype=torch.long).unsqueeze(0).to(config.device)

            with torch.no_grad():
                generated = model.generate(
                    input_tensor,
                    max_new_tokens=100,
                    temperature=0.3,  # Low temperature for more deterministic output
                    top_k=40,
                    stop_at_eos=True
                )  # 100 0.3 40; 50 0.3 20

            generated_text_ori = decode(generated[0].tolist(), tokenizer)
            # Extract just the SQL part
            if '[SQL]' in generated_text_ori:
                generated_text = generated_text_ori.split('[SQL]')[-1].strip()
                print(generated_text)
                print('original generated: ',generated_text_ori )
            else:
                print(generated_text_ori)


        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

    print("\nThanks for testing!")





def parse_user_input(user_input):
    try:
        # Find the positions of the markers
        context_start = user_input.find('[context]')
        question_start = user_input.find('[question]')

        if context_start == -1 or question_start == -1:
            return user_input

        # Extract each part
        context_part = user_input[context_start:question_start].strip()
        question_part = user_input[question_start:].strip()

        return f'{context_part} {question_part}'


    except Exception as e:
        return user_input


if __name__ == '__main__':
    config = Config2(debug_mode=False, fine_tune_mode=True)
    config.tokenizer_init = False  # need to load exisiting tokenizer from pretrained version
    # main(config)
    Natrual_language_chat(config)
    # main_test(config)
