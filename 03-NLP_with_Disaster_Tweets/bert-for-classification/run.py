import model
import preProcess
import data_loader
from runBuilder import RunBuilder
import pandas as pd
import json
import torch.nn as nn
import transformers
#from torch.optim import Adam
from transformers import AdamW
from sklearn.metrics import accuracy_score


import config

device           = config.device
EPOCHS           = config.EPOCHS
BERT_Model_str   = config.BERT_Model_str
BERT_Tokenizer   = config.BERT_Tokenizer
MAX_LEN          = config.MAX_LEN
REMOVE_STOPWORDS = config.REMOVE_STOPWORDS
TO_FREEZE        = ['embeddings', '.0.','.1.','.2.','.3.','.4.',]
#MODEL_PATH       = config.MODEL_PATH
#LR               = config.ADAM_LR
#WEIGHT_DECAY     = config.WEIGHT_DECAY
#DROPOUT          = config.DROPOUT
#BATCH_SIZE       = config.BATCH_SIZE

# hyper-parameters to be tuned
params = {
    'LR'               : [1e-4, 5e-5],
    'WEIGHT_DECAY'     : [0.01, 0.1],
    'DROPOUT'          : [0.1, 0.2],
    'BATCH_SIZE'       : [32, 64],
    }
runs = RunBuilder.get_runs(params)
with open('./models/runs.json', 'w') as f:
    json.dump(runs, f)

results = []
for i, run in enumerate(runs):
    print('\n',i, run,'\n')

    MODEL_PATH       = './models/model_'+str(i)
    LR               = run['LR']
    WEIGHT_DECAY     = run['WEIGHT_DECAY']
    DROPOUT          = run['DROPOUT']
    BATCH_SIZE       = run['BATCH_SIZE']

    # load dataset
    dataset = pd.read_csv('../dataset/train.csv') # path to dataset

    xData = dataset.text
    xData = xData.map(lambda tweet:preProcess.preProc(tweet, REMOVE_STOPWORDS))
    yData = dataset.target

    from sklearn.model_selection import train_test_split
    X_train, X_valid, y_train, y_valid = train_test_split(xData.values, yData.values,
                                                        test_size=0.2,
                                                        shuffle = True,
                                                        stratify=yData.values,
                                                        random_state=42)

    train_data_loader = data_loader.get_data_loader(X_train, y_train, BERT_Tokenizer, MAX_LEN, BATCH_SIZE)
    valid_data_loader = data_loader.get_data_loader(X_valid, y_valid, BERT_Tokenizer, MAX_LEN, BATCH_SIZE)

    BERT_Model = transformers.BertModel.from_pretrained(BERT_Model_str)
    network = model.Network(BERT_Model, DROPOUT).to(device)


    to_be_trained = []
    for i, (name, param) in enumerate(network.named_parameters()):
        if any([ i in name for i in  TO_FREEZE]):
            param.requires_grad = False
        else:
            to_be_trained.append(param)



    loss_function = nn.BCEWithLogitsLoss()

    #no_decay = []
    no_decay = ['LayerNorm.weight', 'bias']
    grouped_params = [
        {
            'params':[par for par in to_be_trained if par not in no_decay],
            'weight_decay':WEIGHT_DECAY
            },
        {
            'params':[par for par in to_be_trained if par in no_decay],
            'weight_decay':0
            }
        ]


    #optimizer = Adam(grouped_params, lr=LR)
    optimizer = AdamW(grouped_params, lr=LR)

    best_valid_accuracy = 0
    train_list, valid_list = [], []
    for epoch in range(EPOCHS):
        # tarin step over all batches
        model.train_step(network, train_data_loader, loss_function, optimizer, device)

        train_targets, train_outputs = model.evaluate(network, train_data_loader, device)
        valid_targets, valid_outputs = model.evaluate(network, valid_data_loader, device)

        train_accuracy = accuracy_score(train_targets, [int(i>0.5) for i in train_outputs])
        valid_accuracy = accuracy_score(valid_targets, [int(i>0.5) for i in valid_outputs])

        train_list.append(train_accuracy)
        valid_list.append(valid_accuracy)

        print("Train Accuracy", train_accuracy)
        print("Valid Accuracy", valid_accuracy)

        if valid_accuracy > best_valid_accuracy:
            best_valid_accuracy = valid_accuracy
            #torch.save(network.state_dict(), MODEL_PATH)

    result = {
        'train_list' : train_list,
        'valid_list' : valid_list
    }
    print('\n',result,'\n')
    results.append(result)

with open('./models/results.json', 'w') as f:
    json.dump(results, f)
