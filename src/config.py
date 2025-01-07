from transformers import AutoTokenizer

TRAIN_BATCH_SIZE = 2
VALID_BATCH_SIZE = 2
EPOCHS = 5
MAX_LEN = 512
TRAINING_FILE = "/kaggle/working/train_folds.csv"
MODEL_NAME = "microsoft/deberta-v3-large"
MODEL_PATH = "deberta_v3_large.bin"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)