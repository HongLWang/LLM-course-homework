
from torch.utils.data import DataLoader
from model import GPTLanguageModel4Finetune
from utils import *
from Configuration import Config


def setup_tokenizer(config, train_text, test_text):
    """Setup tokenizer from pretrained or create new"""
    tokenizer_path = os.path.join(config.pretrained_dir, 'tokenizer.pth')

    if os.path.exists(tokenizer_path):
        tokenizer = Tokenizer(tokenizer_path=tokenizer_path)
    else:
        # Combine all unique characters from both datasets
        all_text = list(set(train_text)) + list(set(test_text))
        tokenizer = Tokenizer(text_data=all_text)
        # Save tokenizer for future use
        os.makedirs(config.pretrained_dir, exist_ok=True)
        tokenizer.save(tokenizer_path)

    return tokenizer


def create_datasets(train_text, test_text, tokenizer, config):
    """Create training and test datasets"""

    print("Parsing training data...")
    if config.debug_mode:
        train_examples = parse_training_data_debug(train_text, config.debug_dataset_size)
    else:
        train_examples = parse_training_data(train_text)
    print(f"Total {len(train_examples)} training examples")

    print("Parsing test data...")
    if config.debug_mode:
        test_examples = parse_training_data_debug(test_text, config.debug_dataset_size)
    else:
        test_examples = parse_training_data(test_text)
    print(f"Total {len(test_examples)} test examples")

    # Create datasets
    train_dataset = SQLDataset(train_examples, config.block_size, tokenizer)
    test_dataset = SQLDataset(test_examples, config.block_size, tokenizer)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    print(f"Training batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")

    return train_loader, test_loader, train_examples, test_examples


def load_model(config, tokenizer):
    """Load or create model"""
    model = GPTLanguageModel4Finetune(
        vocab_size=tokenizer.vocab_size,
        n_embd=config.n_embd,
        n_head=config.n_head,
        n_layer=config.n_layer,
        block_size=config.block_size,
        dropout=config.dropout,
    )

    # Set EOS token for generation
    model.set_eos_token(tokenizer.EOS_TOKEN)

    # Load pretrained weights if available
    pretrained_weights_path = os.path.join(config.pretrained_dir, 'model_weights.pth')
    if os.path.exists(pretrained_weights_path):
        model.load_state_dict(torch.load(pretrained_weights_path))
        print("Loaded pretrained weights successfully!")
    else:
        print("No pretrained weights found. Training from scratch...")

    model = model.to(config.device)
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")

    return model


@torch.no_grad()
def evaluate_model(model, test_loader, config):
    """Evaluate model on test set"""
    model.eval()
    total_loss = 0
    total_samples = 0

    for batch in test_loader:
        input_ids = batch['input_ids'].to(config.device)
        targets = batch['targets'].to(config.device)
        loss_mask = batch['loss_mask'].to(config.device)

        logits, loss = model(input_ids, targets, loss_mask)

        total_loss += loss.item() * input_ids.size(0)
        total_samples += input_ids.size(0)

    model.train()
    return total_loss / total_samples


def train_model(model, train_loader, test_loader, config):
    """Train the model"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    print("\n--- Starting Finetuning ---")
    model.train()
    best_val_loss = float('inf')
    step = 0

    for epoch in range(config.max_iters // len(train_loader) + 1):
        for batch in train_loader:
            if step >= config.max_iters:
                break

            input_ids = batch['input_ids'].to(config.device)
            targets = batch['targets'].to(config.device)
            loss_mask = batch['loss_mask'].to(config.device)

            # Forward pass
            logits, loss = model(input_ids, targets, loss_mask)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Evaluation
            if step % config.eval_interval == 0 or step == config.max_iters - 1:
                val_loss = evaluate_model(model, test_loader, config)
                print(f"step {step}: train loss {loss.item():.4f}, val loss {val_loss:.4f}")

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_path = os.path.join(config.pretrained_dir, 'finetuned_model.pth')
                    torch.save(model.state_dict(), save_path)
                    print(f"New best model saved with val loss: {val_loss:.4f}")

            step += 1

        if step >= config.max_iters:
            break

    print("--- Finetuning Complete ---")
    return best_val_loss


def evaluate_generation(model, test_examples, tokenizer, config, num_demonstration_samples=5):
    """Evaluate model generation quality"""
    print("\n------------------- Evaluating Finetuned Model's generation ability ------------------")

    exact_matches = 0
    bleu_scores = []

    # Show sample predictions
    print("\n--- Sample Predictions ---")


    # Calculate metrics for all test examples
    flag = 1
    for i, example in enumerate(test_examples):
        if i % 10 == 0 and i > 0:
            print(f"Processing {i + 1}/{len(test_examples)} examples...")

        context = example['context']
        question = example['question']
        target_sql = example['sql']

        predicted_sql = generate_sql_from_input_CL(
            model, tokenizer, context, question, config.device, block_size=config.block_size
        )

        if calculate_exact_match(predicted_sql, target_sql):
            exact_matches += 1

        bleu_score = calculate_bleu_score(predicted_sql, target_sql)
        bleu_scores.append(bleu_score)



        if i<= config.num_example_2_print:  # print the first {num_demonstration_samples} generated results
            print(f"\nExample {i + 1}:")
            print(f"Context: {context}")
            print(f"Question: {question}")
            print(f"Target SQL: {target_sql}")
            print(f"Predicted SQL: {predicted_sql}")
            print(f"Exact Match: {exact_matches}")
            print(f"BLEU Score: {bleu_score:.4f}")
            print("--------------------------------------------")

        if i> config.num_example_2_print and flag:
            print("\n----------- Computing Final Metrics -----------")
            flag = 0

    # Print final results
    print("\n--- Final Results ---")
    print(f"Total test examples: {len(test_examples)}")
    print(f"Exact match accuracy: {exact_matches}/{len(test_examples)} = "
          f"{exact_matches / len(test_examples) * 100:.2f}%")
    print(f"Average BLEU score: {sum(bleu_scores) / len(bleu_scores):.4f}")
    print(f"Median BLEU score: {sorted(bleu_scores)[len(bleu_scores) // 2]:.4f}")

    high_bleu = sum(1 for score in bleu_scores if score > 0.5)
    print(f"Examples with BLEU > 0.5: {high_bleu}/{len(test_examples)} = "
          f"{high_bleu / len(test_examples) * 100:.2f}%")

    # Save results
    results = {
        'exact_match_accuracy': exact_matches / len(test_examples),
        'average_bleu': sum(bleu_scores) / len(bleu_scores),
        'median_bleu': sorted(bleu_scores)[len(bleu_scores) // 2],
        'high_bleu_count': high_bleu,
        'total_examples': len(test_examples)
    }

    results_path = os.path.join(config.pretrained_dir, 'finetune_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nFinetuning results saved to {results_path}")

    return results


def main():
    """Main training and evaluation pipeline"""
    # Initialize configuration
    config = Config(debug_mode=False, fine_tune_mode=True)

    # Set random seed for reproducibility
    torch.manual_seed(config.seed)

    # Load data
    print("Loading data...")
    train_text, test_text = load_data(config.data_path)

    # Setup tokenizer
    print("Setting up tokenizer...")
    tokenizer = setup_tokenizer(config, train_text, test_text)

    # Create datasets and dataloaders
    train_loader, test_loader, train_examples, test_examples = create_datasets(
        train_text, test_text, tokenizer, config
    )

    # Load model
    print("Loading model...")
    model = load_model(config, tokenizer)

    # Train model
    best_val_loss = train_model(model, train_loader, test_loader, config)

    # Load best model for evaluation
    print("\n--- Loading Best Model for Evaluation ---")
    best_model_path = os.path.join(config.pretrained_dir, 'finetuned_model.pth')
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    # Evaluate generation quality
    results = evaluate_generation(model, test_examples, tokenizer, config)

    print("\n--- Training and Evaluation Complete ---")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final exact match accuracy: {results['exact_match_accuracy'] * 100:.2f}%")
    print(f"Final average BLEU score: {results['average_bleu']:.4f}")


if __name__ == "__main__":
    main()