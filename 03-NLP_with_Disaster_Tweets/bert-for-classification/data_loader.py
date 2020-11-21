import torch

class Dataset(object):
    def __init__(self, text, target, BERT_Tokenizer, MAX_LEN):
        self.text    = text
        self.target  = target
        self.tknzr   = BERT_Tokenizer
        self.max_len = MAX_LEN
        
    def __len__(self):
        return len(self.target)
    
    def __getitem__(self, item_idx):
        
        text = str(self.text[item_idx])
        text = ''.join(text.split()) # to remove any extra spaces (should be moved to preprocess function later)
        
        tknzr_output    = self.tknzr.encode_plus(text, max_length = self.max_len, truncation=True)
        
        input_ids       = tknzr_output['input_ids']
        token_type_ids  = tknzr_output['token_type_ids']
        attention_mask  = tknzr_output['attention_mask']
        
        padding_length  = self.max_len - len(input_ids) # if len less than MAX_LEN right padding to be added
        
        input_ids       = torch.tensor(input_ids + [0]*padding_length, dtype=torch.long)
        token_type_ids  = torch.tensor(token_type_ids + [0]*padding_length, dtype=torch.long)
        attention_mask  = torch.tensor(attention_mask + [0]*padding_length, dtype=torch.long)
        
        target = torch.tensor(self.target[item_idx], dtype=torch.float)
        
        #return input_ids, token_type_ids, attention_mask, target
        
        input_dict = {
            'input_ids':input_ids,
            'token_type_ids':token_type_ids,
            'attention_mask':attention_mask,
            'target':target
        }
        
        return input_dict

def get_data_loader(xData, yData, BERT_Tokenizer, MAX_LEN, BATCH_SIZE):
    dataset = Dataset(xData, yData, BERT_Tokenizer, MAX_LEN)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE)
    return data_loader