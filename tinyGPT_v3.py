
from tinyGPT_v2 import main
from Configuration import Config3

if __name__ == '__main__':
    config = Config3(debug_mode=False, fine_tune_mode=True)  # this is the only line that is different from that of tinyGPT_v2
    config.tokenizer_init = False
    main(config)
