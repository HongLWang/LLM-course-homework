from datasets import load_dataset

def load_dataset_from_HuggingFace(config):
    # --- Data Loading from Hugging Face ---
    print(f"Loading dataset: {config.dataset_name} - {config.dataset_config}")
    try:
        # List of datasets to tokenize
        datasets_to_tokenize = [
            ("HuggingFaceFW/fineweb", "sample-10BT", ["text"], "sample/10BT/014_00000.parquet")
        ]

        for dataset_name, remote_name, text_columns, data_file in datasets_to_tokenize:
            dataset = load_dataset(dataset_name, name=remote_name, data_files=data_file, split="train")

        print(f"Dataset loaded successfully")

        # Convert streaming dataset to list for easier handling
        print("Processing dataset samples...")
        texts = []

        for i, example in enumerate(dataset):
            if i >= config.max_samples:
                break

            text = example.get('text', '')

            # Filter by text length
            if config.min_text_length <= len(text) <= config.max_text_length:
                texts.append(text)

            if i % 1000 == 0:
                print(f"Processed {i} samples, kept {len(texts)} texts")

        print(f"Final dataset size: {len(texts)} texts")

        # Combine all texts with separator
        combined_text = ' '.join(texts)
        print(f"Combined text length: {len(combined_text)} characters")

    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Falling back to dummy data for testing...")
        # Fallback dummy data
        combined_text = """
            This is a sample text for training a language model. 
            Language models learn to predict the next word in a sequence.
            They are trained on large amounts of text data.
            The transformer architecture has revolutionized natural language processing.
            """ * 100

    return combined_text
