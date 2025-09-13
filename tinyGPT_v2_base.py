import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import re
import json
from collections import defaultdict
from transformers import GPT2Tokenizer

from data_processing import process_and_save_sql_data
from model import GPTv2Base
from Configuration import Config2
from utils import *



def encode(s,tokenizer):
    """Encoder using GPT-2 tokenizer with special token handling - FIXED"""
    # Replace section markers with special tokens
    s = s.replace('[context]', '<CONTEXT>')
    s = s.replace('[question]', '<QUESTION>')
    s = s.replace('[SQL]', '<SQL>')

    # Encode using GPT-2 tokenizer with proper special token handling
    tokens = tokenizer.encode(s, add_special_tokens=False)  # Don't add BOS/EOS automatically

    return tokens


def decode(tokens,tokenizer):
    """Decoder using GPT-2 tokenizer that handles special tokens - FIXED"""
    # Decode using GPT-2 tokenizer
    text = tokenizer.decode(tokens, skip_special_tokens=False)

    # Replace special tokens back to original format
    text = text.replace('<CONTEXT>', '[context]')
    text = text.replace('<QUESTION>', '[question]')
    text = text.replace('<SQL>', '[SQL]')

    # Clean up any remaining special tokens
    text = text.replace('<|endoftext|>', '')
    text = text.replace('<|startoftext|>', '')

    return text.strip()




def save_model_and_tokenizer(model, tokenizer, config):
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    # Save model and tokenizer
    torch.save(model.state_dict(), os.path.join(config.output_dir, 'model_weights.pth'))

    # Save tokenizer configuration
    tokenizer_config = {
        'vocab_size': config.vocab_size,
        'special_tokens': {
            'config.CONTEXT_TOKEN': config.CONTEXT_TOKEN,
            'config.QUESTION_TOKEN': config.QUESTION_TOKEN,
            'config.SQL_TOKEN': config.SQL_TOKEN,
            'config.BOS_TOKEN': config.BOS_TOKEN,
            'config.EOS_TOKEN': config.EOS_TOKEN,
            'config.PAD_TOKEN': config.PAD_TOKEN,
        }
    }

    config.tokenizer_config_fp = os.path.join(config.output_dir, 'tokenizer_config.json')
    with open(config.tokenizer_config_fp, 'w') as f:
        json.dump(tokenizer_config, f, indent=2)

    config.tokenizer_fp = os.path.join(config.output_dir, 'saved_tokenizer.json')
    # Save the actual tokenizer
    tokenizer.save_pretrained(config.tokenizer_fp)









def load_model_and_tokenizer(config):

    print("\n--- Loading Model and tokenizer for Evaluation ---")
    model_eval = GPTv2Base(
        vocab_size=config.vocab_size,
        n_embd=config.n_embd,
        block_size=config.block_size,
        n_head=config.n_head,
        n_layer=config.n_layer,
        dropout=config.dropout,
        SQL_TOKEN=config.SQL_TOKEN,
        EOS_TOKEN=config.EOS_TOKEN,
        QUESTION_TOKEN=config.QUESTION_TOKEN,
        CONTEXT_TOKEN=config.CONTEXT_TOKEN,
        PAD_TOKEN=config.PAD_TOKEN
    )
    model_eval.load_state_dict(torch.load(os.path.join(config.output_dir, 'model_weights.pth')))
    model_eval = model_eval.to(config.device)
    model_eval.eval()

    # Load tokenizer configuration
    # loaded_tokenizer = GPT2Tokenizer.from_pretrained(config.tokenizer_fp)

    return model_eval





def generate_sql_from_input(config, model, context, question, tokenizer, max_tokens=100):

    #Generate SQL from context and question using special tokens
    # if the input is standard context+question, process it as f"[context] {context} [question] {question} [SQL] "
    # otherwise just encode it to get a tensor

    input_text = f"[context] {context} [question] {question} [SQL] "

    config.all_natural_language_mode = False

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
    generated_text = decode(generated[0].tolist(), tokenizer)

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

    else:
        return generated_text




def train_model(config, model,train_data, val_data):
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    print("\n-------------------- Starting Training --------------------")
    for iter in range(config.max_iters):
        if iter % config.eval_interval == 0 or iter == config.max_iters - 1:
            losses = estimate_loss(config, model, train_data, val_data)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch(config, train_data)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    print("------------------- Training Complete -------------------")


def _test_model(config, model_eval,tokenizer):
    # --- Run Evaluation ---
    print("\n--- Evaluating on Test Data ---")

    # Load test data
    test_entries = parse_test_data(config, config.test_file)
    print(f"Loaded {len(test_entries)} test entries")

    if len(test_entries) == 0:
        print("No test entries found. Please check the test file format.")
    else:
        # Test tokenization first
        print("\n--- Testing Tokenization ---")
        test_input = "[context] CREATE TABLE farm (Cows INTEGER) [question] What are the maximum and minimum number of cows across all farms. [SQL] SELECT MAX(Cows), MIN(Cows) FROM farm"
        tokens = encode(test_input, tokenizer)
        decoded = decode(tokens, tokenizer)
        print(f"Original: {test_input}")
        print(f"Decoded: {decoded}")
        print(f"Match: {test_input == decoded}")

        # Evaluate model
        exact_matches = 0
        bleu_scores = []

        print("\n--- Sample Predictions ---")
        for i, entry in enumerate(test_entries[:5]):  # Show first 5 examples
            context = entry['context']
            question = entry['question']
            target_sql = entry['sql']

            # Generate prediction
            predicted_sql = generate_sql_from_input(config, model_eval, context, question,tokenizer)

            # Calculate metrics
            exact_match = calculate_exact_match(predicted_sql, target_sql)
            bleu_score = calculate_bleu_score(predicted_sql, target_sql)

            print(f"\nExample {i + 1}:")
            print(f"Context: {context[:100]}...")
            print(f"Question: {question}")
            print(f"Target SQL: {target_sql}")
            print(f"Predicted SQL: {predicted_sql}")
            print(f"Exact Match: {exact_match}")
            print(f"BLEU Score: {bleu_score:.4f}")
            print("-" * 50)

        # Calculate metrics for all test entries
        print("\n--- Computing Final Metrics ---")
        for i, entry in enumerate(test_entries):
            if i % 10 == 0:
                print(f"Processing {i + 1}/{len(test_entries)} examples...")

            context = entry['context']
            question = entry['question']
            target_sql = entry['sql']

            # Generate prediction
            predicted_sql = generate_sql_from_input(config, model_eval, context, question, tokenizer)

            # Calculate metrics
            if calculate_exact_match(predicted_sql, target_sql):
                exact_matches += 1

            bleu_score = calculate_bleu_score(predicted_sql, target_sql)
            bleu_scores.append(bleu_score)

        # Print final results
        print("\n--- Final Results ---")
        print(f"Total test examples: {len(test_entries)}")
        print(
            f"Exact match accuracy: {exact_matches}/{len(test_entries)} = {exact_matches / len(test_entries) * 100:.2f}%")
        print(f"Average BLEU score: {sum(bleu_scores) / len(bleu_scores):.4f}")
        print(f"Median BLEU score: {sorted(bleu_scores)[len(bleu_scores) // 2]:.4f}")

        # Additional statistics
        high_bleu = sum(1 for score in bleu_scores if score > 0.5)
        print(f"Examples with BLEU > 0.5: {high_bleu}/{len(test_entries)} = {high_bleu / len(test_entries) * 100:.2f}%")

        # Save results
        results = {
            'exact_match_accuracy': exact_matches / len(test_entries),
            'average_bleu': sum(bleu_scores) / len(bleu_scores),
            'median_bleu': sorted(bleu_scores)[len(bleu_scores) // 2],
            'high_bleu_count': high_bleu,
            'total_examples': len(test_entries)
        }

        config.fine_tune_mode = False

        if config.fine_tune_mode:
            fp_name = 'finetune_results.json'
        else:
            fp_name = 'evaluation_results.json'

        with open(os.path.join(config.output_dir, fp_name), 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nEvaluation results saved to {config.output_dir}/{fp_name}")




def set_tokenizer(config):
    # init GPT-2 BPE Tokenizer

    if config.tokenizer_init:
        print("Loading GPT-2 tokenizer...")
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
    else:  # tokenizer_load
        tokenizer_fp = config.output_dir
        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_fp)

    # Add special tokens for our SQL task - FIXED METHOD
    special_tokens_dict = {
        'additional_special_tokens': ['<CONTEXT>', '<QUESTION>', '<SQL>']
    }
    tokenizer.add_special_tokens(special_tokens_dict)

    # Get token IDs for special tokens - FIXED METHOD
    config.CONTEXT_TOKEN = tokenizer.convert_tokens_to_ids('<CONTEXT>')
    config.QUESTION_TOKEN = tokenizer.convert_tokens_to_ids('<QUESTION>')
    config.SQL_TOKEN = tokenizer.convert_tokens_to_ids('<SQL>')

    # GPT-2 already has these tokens
    config.BOS_TOKEN = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    config.EOS_TOKEN = tokenizer.eos_token_id
    config.PAD_TOKEN = tokenizer.pad_token_id

    vocab_size = len(tokenizer)
    config.vocab_size = vocab_size



    print(f"Vocabulary size: {vocab_size}")
    print(f"BOS token ID: {config.BOS_TOKEN}")
    print(f"EOS token ID: {config.EOS_TOKEN}")
    print(f"PAD token ID: {config.PAD_TOKEN}")
    print(f"CONTEXT token ID: {config.CONTEXT_TOKEN}")
    print(f"QUESTION token ID: {config.QUESTION_TOKEN}")
    print(f"SQL token ID: {config.SQL_TOKEN}")


    return tokenizer

def load_n_tokenize_data(config,tokenizer):
    # --- Data Loading ---
    data_path = config.data_path
    train_file = os.path.join(data_path, "train.txt")
    test_file = os.path.join(data_path, "test.txt")

    with open(train_file, 'r', encoding='utf-8') as f1:
        text = f1.read()

    with open(test_file, 'r', encoding='utf-8') as f2:
        text_test = f2.read()

    # Process the training text
    processed_text = process_training_data(text)
    print(f"Processed {len(processed_text.split('[context]')) - 1} training examples")

    cache_file = os.path.join(data_path, f"tokenized_train_data.pt")

    # Try to load from cache first
    if os.path.exists(cache_file):
        print(f"Loading tokenized data from cache: {cache_file}")
        try:
            data = torch.load(cache_file)
            print("Successfully loaded tokenized data from cache")
        except Exception as e:
            print(f"Failed to load from cache: {e}")
            print("Regenerating tokenized data...")
            data = torch.tensor(tokenizer.encode(processed_text), dtype=torch.long)
            # Save to cache
            torch.save(data, cache_file)
            print(f"Saved tokenized data to cache: {cache_file}")
    else:
        print("Cache not found. Tokenizing data...")
        # Encode data (this is the slow operation)
        data = torch.tensor(tokenizer.encode(processed_text), dtype=torch.long)

        # Save to cache
        try:
            torch.save(data, cache_file)
            print(f"Saved tokenized data to cache: {cache_file}")
        except Exception as e:
            print(f"Failed to save to cache: {e}")

    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    print(f"Training data size: {len(train_data)}")
    print(f"Validation data size: {len(val_data)}")

    return train_data, val_data

def print_info(config):
    print(f"---------------------printing infos for GPT BPE tokenizer---------------------")
    print(f"Vocabulary size: {config.vocab_size}")
    print(f"BOS token ID: {config.BOS_TOKEN}")
    print(f"EOS token ID: {config.EOS_TOKEN}")
    print(f"PAD token ID: {config.PAD_TOKEN}")
    print(f"CONTEXT token ID: {config.CONTEXT_TOKEN}")
    print(f"QUESTION token ID: {config.QUESTION_TOKEN}")
    print(f"SQL token ID: {config.SQL_TOKEN}")



def main(config):
    # for reproducibility
    torch.manual_seed(1337)



    tokenizer = set_tokenizer(config)  # init tokenizer, set padding token + add special token + get special token ids and save to config

    # Update vocab size to include new special tokens
    vocab_size = len(tokenizer)
    config.vocab_size = vocab_size

    # print tokenier info
    print_info(config)

    # load data, tokenize data, split to train and validation set
    train_data, val_data = load_n_tokenize_data(config,tokenizer)


    model = GPTv2Base(
        vocab_size=config.vocab_size,
        n_embd=config.n_embd,
        block_size=config.block_size,
        n_head=config.n_head,
        n_layer=config.n_layer,
        dropout=config.dropout,
        SQL_TOKEN=config.SQL_TOKEN,
        EOS_TOKEN=config.EOS_TOKEN,
        QUESTION_TOKEN=config.QUESTION_TOKEN,
        CONTEXT_TOKEN=config.CONTEXT_TOKEN,
        PAD_TOKEN=config.PAD_TOKEN
    )



    m = model.to(config.device)
    print(f"{sum(p.numel() for p in m.parameters()) / 1e6:.2f}M parameters")

    trainagain = 1
    if trainagain:
    # --- Training ---
        train_model(config, model, train_data, val_data)

        # --- Save the model ---
        save_model_and_tokenizer(model, tokenizer, config)
        print(f"\nModel and tokenizer saved to {config.output_dir}")
    else:
        config.tokenizer_fp = os.path.join(config.output_dir, 'tokenizer_config.json')
        pass # directly load trained model

    # --- Load the model for evaluation ---
    model_eval = load_model_and_tokenizer(config)

    # test model
    _test_model(config, model_eval, tokenizer)






if __name__ == '__main__':

    config = Config2(debug_mode=True, fine_tune_mode=False)

    config.tokenizer_init = True # need to init

    ## If you want to train and evaluate the model
    main(config)

