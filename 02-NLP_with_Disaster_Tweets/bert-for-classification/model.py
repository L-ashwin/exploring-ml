import torch
import torch.nn as nn
from tqdm import tqdm

class Network(nn.Module):
    def __init__(self, BERT_Model, DROPOUT):
        super(Network, self).__init__()
        
        self.bert   = BERT_Model
        self.drop   = nn.Dropout(DROPOUT)
        self.output = nn.Linear(768, 1)
    
    def forward(self, input_ids, token_type_ids, attention_mask):
        _, bert2 = self.bert( #bert2 = cls output of bert model
            input_ids,
            token_type_ids = token_type_ids,
            attention_mask = attention_mask,
        )
        drop   = self.drop(bert2)
        output = self.output(drop)
        
        return output

def train_step(model, data_loader, loss_function, optimizer, device):
    for batch in tqdm(data_loader):
        input_ids       = batch['input_ids'].to(device)
        token_type_ids  = batch['token_type_ids'].to(device)
        attention_mask  = batch['attention_mask'].to(device)
        target          = batch['target'].to(device)

        output = model.forward(input_ids, token_type_ids, attention_mask)
        loss = loss_function(output.flatten(), target.flatten())
        #print(loss)
        
        optimizer.zero_grad() 
        loss.backward() 
        optimizer.step() 

def evaluate(model, data_loader, device):
    targets, outputs = [],[]
    with torch.no_grad():
        for batch in data_loader:
            input_ids       = batch['input_ids'].to(device)
            token_type_ids  = batch['token_type_ids'].to(device)
            attention_mask  = batch['attention_mask'].to(device)
            target          = batch['target'].to(device)
            
            targets.extend(target.cpu().detach().numpy().tolist())
            
            output = model.forward(input_ids, token_type_ids, attention_mask)
            outputs.extend(torch.sigmoid(output).flatten().cpu().detach().numpy().tolist())    
    
    #return accuracy_score(targets,[int(i[0]>0.5) for i in outputs])
    return targets, outputs