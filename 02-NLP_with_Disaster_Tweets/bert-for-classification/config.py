import torch
import transformers

MODEL_PATH     = './models/model'
BERT_Model_str = "bert-base-uncased"
BERT_Tokenizer = transformers.BertTokenizer.from_pretrained(BERT_Model_str)


DROPOUT        = 0.3

MAX_LEN        = 50
BATCH_SIZE     = 32

ADAM_LR        = 1e-6
WEIGHT_DECAY   = 1e-3

EPOCHS         = 4

REMOVE_STOPWORDS = True

device = torch.device('cuda')