
import os, torch

# --- Configuration ---
class Config():  # this configuration is for tinyGPT-v1 and tinyGPT-v1-base
    def __init__(self, debug_mode=False, fine_tune_mode=False):
        self.batch_size = 64
        self.block_size = 256
        self.max_iters = 5000
        self.eval_interval = 10  # print status after every eval_interval testing examples
        self.learning_rate = 3e-4
        self.eval_iters = 200
        self.n_embd = 320
        self.n_head = 5
        self.n_layer = 8
        self.dropout = 0.2

        # Paths to save data and pretrianed models, results
        self.data_path = "./sql_data"
        self.pretrained_dir = "./tiny-gpt-base-v1"
        self.output_dir = "./tiny-gpt-base-v1"
        # Device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


        self.seed = 1337
        self.num_example_2_print = 5
        self.debug_mode = debug_mode

        if self.debug_mode:  # for fast debugging
            self.max_iters = 20
            self.debug_dataset_size =256
            self.eval_interval = 5
            self.batch_size = 8
            self.block_size = 32  # this should be set smaller than the token size after encoding
            self.n_embd = 40

        if fine_tune_mode:  # use less iteration than pretrain
            self.max_iters = 1000
            self.eval_interval = 100  # print status after every eval_interval testing examples
            self.learning_rate = 1e-4


class Config2:  # this configuration is for tinyGPT-v2 and tinyGPT-v2-base
    def __init__(self, debug_mode=False, fine_tune_mode=False):
        self.batch_size = 64
        self.block_size = 128
        self.max_iters = 4000
        self.eval_interval = 10
        self.learning_rate = 3e-4
        self.eval_iters = 1
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.n_embd = 192
        self.n_head = 3
        self.n_layer = 3
        self.dropout = 0.1

        # Paths
        self.data_path = "sql_data"
        self.output_dir = "./tiny-gpt-base-v2"
        self.train_file = os.path.join(self.data_path, "train.txt")
        self.test_file = os.path.join(self.data_path, "test.txt")
        self.debug_mode = debug_mode
        self.all_natural_language_mode = False

        if self.debug_mode:  # for fast debugging
            self.max_iters = 10
            self.debug_dataset_size = 128
            self.eval_interval = 5
            self.batch_size = 8
            self.block_size = 32  # this should be set smaller than the token size after encoding, otherwise you will encounter error
            self.n_embd = self.batch_size * self.n_head

        if fine_tune_mode:
            self.batch_size = 32
            self.max_iters = 100
            self.learning_rate = 1e-4
            self.pretrained_model_path = self.output_dir


class Config3: # this configuration is for tinyGPT-v3 and tinyGPT-v3-base
    def __init__(self, debug_mode=False, fine_tune_mode=False):

        self.batch_size = 128
        self.block_size = 128
        self.max_iters = 600
        self.eval_interval = 200
        self.learning_rate = 1e-4
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.eval_iters = 50
        self.n_embd = 192
        self.n_head = 3
        self.n_layer = 3
        self.dropout = 0.1

        # Dataset parameters
        self.dataset_name = "HuggingFaceFW/fineweb"
        self.dataset_config = "sample-10BT"
        self.output_dir = "./tiny-gpt-base-v3"
        self.pretrained_model_path = "./tiny-gpt-base-v3"
        self.max_samples = 10000
        self.min_text_length = 50
        self.max_text_length = 1000

        self.debug_mode = debug_mode
        # --- Adjust for debug mode ---
        if self.debug_mode:
            self.max_iters = 10
            self.debug_dataset_size = 128
            self.eval_interval = 5
            self.batch_size = 8
            self.block_size = 32
            self.n_embd = self.batch_size * self.n_head  # dynamically scale

        # --- Adjust for fine-tune mode ---
        if fine_tune_mode:
            self.batch_size = 32
            self.max_iters = 100
            self.learning_rate = 1e-4



