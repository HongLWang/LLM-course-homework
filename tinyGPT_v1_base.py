#!/usr/bin/env python3


from model import GPTLanguageModel
from utils import *
from Configuration import Config

# Data configuration


# print('global block size is ', config.block_size)

def setup_device():
    """Setup device for training"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    return device


def prepare_data(config, tokenizer):
    """Load and prepare training data"""
    # Load raw data
    text, text_test = load_data(config.data_path)

    # Process training text, if for debugging, use small portion of data
    if not config.debug_mode:
        processed_text = process_training_data(text)
    else:
        processed_text = process_training_data_debug(text, config.debug_dataset_size)
    print(f"Processed {len(processed_text.split('[context]')) - 1} training examples")

    # Encode data
    data = torch.tensor(tokenizer.encode(processed_text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Training data size: {len(train_data)}")
    print(f"Validation data size: {len(val_data)}")

    return train_data, val_data, text_test



def train_model(config, model, train_data, val_data, tokenizer):
    """Train the model"""
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    print("\n---------------------- Starting Training ----------------------")
    for iter in range(config.max_iters):
        if iter % config.eval_interval == 0 or iter == config.max_iters - 1:
            losses = estimate_loss(config, model, train_data, val_data)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        data_batch, truth_batch = get_batch(config, train_data)
        logits, loss = model(data_batch, truth_batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print("--------------------- Training Complete ----------------------")


def save_model_and_tokenizer(model, tokenizer, config):
    """Save model weights and tokenizer"""
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    # Save model weights
    torch.save(model.state_dict(), os.path.join(config.output_dir, 'model_weights.pth'))

    # Save tokenizer
    tokenizer.save(os.path.join(config.output_dir, 'tokenizer.pth'))

    print(f"\nModel and tokenizer saved to {config.output_dir}")


def load_model_for_evaluation(tokenizer, config):
    """Load trained model for evaluation"""
    print("\n--------------------- loading saved model ---------------------")
    model_eval = GPTLanguageModel(
        vocab_size=tokenizer.vocab_size,
        n_embd=config.n_embd,
        n_head=config.n_head,
        n_layer=config.n_layer,
        block_size=config.block_size,
        dropout=config.dropout,
        pad_token_id=tokenizer.PAD_TOKEN,
        eos_token_id=tokenizer.EOS_TOKEN
    )
    model_eval.load_state_dict(torch.load(os.path.join(config.output_dir, 'model_weights.pth')))
    model_eval = model_eval.to(config.device)
    model_eval.eval()

    # print('after training , and after loading , the model block size is ', model_eval.config.block_size)
    return model_eval


def evaluate_model(model, tokenizer, config):
    """Evaluate model on test data"""
    print("\n--------------------- evaluating on test data ---------------------")

    # Load test data
    test_file = os.path.join(config.data_path, "test.txt")
    if not config.debug_mode:
        test_entries = parse_test_data(config,test_file)
    else:
        test_entries = parse_test_data_debug(test_file, config.debug_dataset_size)

    print(f"Loaded {len(test_entries)} test entries")

    if len(test_entries) == 0:
        print("No test entries found. Please check the test file format.")
        return None

    # Show sample predictions
    exact_matches = 0
    bleu_scores = []

    print("\n------------ Sample Predictions ------------")
    for i, entry in enumerate(test_entries[:config.num_example_2_print]):
        context = entry['context']
        question = entry['question']
        target_sql = entry['sql']

        # Generate prediction
        predicted_sql = generate_sql_from_input_CL(model, tokenizer, context, question, config.device, block_size = config.block_size)

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
    print("\n---------- Computing Final Metrics ----------")
    for i, entry in enumerate(test_entries):
        if i % 10 == 0:
            print(f"Processing percentage: {i + 1}/{len(test_entries)}")

        context = entry['context']
        question = entry['question']
        target_sql = entry['sql']

        # Generate prediction
        predicted_sql = generate_sql_from_input_CL(model, tokenizer, context, question, config.device, block_size = config.block_size)

        # Calculate metrics
        if calculate_exact_match(predicted_sql, target_sql):
            exact_matches += 1

        bleu_score = calculate_bleu_score(predicted_sql, target_sql)
        bleu_scores.append(bleu_score)

    # Print final results
    print("\n--------------------GPT2SQL task on GPTv1Base Model Final Testing Results --------------------")
    print(f"Total test examples: {len(test_entries)}")
    print(f"Exact match accuracy: {exact_matches}/{len(test_entries)} = {exact_matches / len(test_entries) * 100:.2f}%")
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

    return results




def main():

    config = Config(debug_mode=False, fine_tune_mode=False)
    """Main training and evaluation pipeline"""
    # Set random seed for reproducibility
    torch.manual_seed(1337)

    device = setup_device()

    text, text_test = load_data(config.data_path)
    alltext = list(set(text)) + list(set(text_test))

    tokenizer = Tokenizer(text_data = alltext)

    train_data, val_data, _ = prepare_data(config, tokenizer)

    model = GPTLanguageModel(
        vocab_size=tokenizer.vocab_size,
        n_embd=config.n_embd,
        n_head=config.n_head,
        n_layer=config.n_layer,
        block_size=config.block_size,
        dropout=config.dropout,
        pad_token_id=tokenizer.PAD_TOKEN,
        eos_token_id=tokenizer.EOS_TOKEN
    )
    model = model.to(device)
    # print('before training, after init, the model block size is ', model.config.block_size)

    train_model(config, model, train_data, val_data, tokenizer)

    save_model_and_tokenizer(model, tokenizer, config)

    # Load model for evaluation
    model_eval = load_model_for_evaluation(tokenizer, config)

    # Evaluate model
    results = evaluate_model(model_eval, tokenizer, config)

    if results:
        # Save evaluation results
        save_evaluation_results(results, config.output_dir)
        print(f"\nEvaluation results saved to {config.output_dir}/evaluation_results.json")



if __name__ == "__main__":
    main()
