import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import random
import pandas as pd
import time
import numpy as np
import warnings
from gensim.models.word2vec import Word2Vec
from torch.autograd import Variable
from sklearn.metrics import precision_recall_fscore_support
from utilities import SharedFunction
# warnings.filterwarnings('ignore')

from transformers import AutoTokenizer, AutoModel, AdamW, AutoConfig
from torch.utils.data import DataLoader
from dataset import CodeClassificationDataset

class BatchProgramClassifier(nn.Module):
    # def __init__(self, embedding_dim, hidden_dim, vocab_size, encode_dim, label_size, batch_size,
    #              use_gpu=True, pretrained_weight=None):
    def __init__(self,
                 encoder_name,
                 hidden_dim,
                 label_size
                ):
        super(BatchProgramClassifier, self).__init__()       

        self.encoder_name = encoder_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.encoder_name)
        self.config = AutoConfig.from_pretrained(self.encoder_name)
        # self.config.num_labels = label_size

        self.encoder = AutoModel.from_pretrained(self.encoder_name, config = self.config)

        # freeze the encoder
        # for param in self.encoder.parameters():
        #     param.requires_grad = False

        self.hidden_dim = hidden_dim
        self.label_size = label_size

        # linear
        self.hidden2label = nn.Linear(self.hidden_dim * 4 + 1, self.label_size)
    
    def encode(self, input_ids):
        return self.encoder(input_ids,attention_mask=input_ids.ne(1))[0][:, 0,:]
    
    def encode_number(self, x):
        return x   
    
    def forward(self, x1=None, x2=None, x3=None, x4=None, x8=None, x9=None, x10=None):
        code = self.encode(x1)
        # code_versions = self.encode(x2)
        calling = self.encode(x3)
        called = self.encode(x4)
        number_of_days = self.encode_number(x8)
        code_versions_all = self.encode(x10)
                
        # inputs = torch.cat([code, code_versions], dim=1)
        inputs = torch.cat([code, code_versions_all, calling, called, number_of_days], dim=1)
        # inputs = torch.cat([code, code_versions, calling, called], dim=1)

        y = self.hidden2label(inputs)
        return y

def get_batch_transformer(batch):
    x1 = batch['code_ids']
    x2 = batch['code_versions_ids']
    x3 = batch['calling_ids']
    x4 = batch['called_ids']
    # x5 = None
    # x6 = None
    # x7 = None
    x8 = batch['number_of_days_ids']
    x9 = batch['number_of_versions_ids']
    x10 = batch['code_versions_all_ids']

    train_labels = batch['label']
    return x1, x2, x3, x4, x8, x9, x10, train_labels

if __name__ == '__main__':
    RANDOM_SEED = 2023
    # DATA_DIR = './data/classification'
    MODEL_DIR = './models/transformer_codebert'
    
    # word2vec = Word2Vec.load(DATA_DIR + '/node_w2v_128').wv
    # MAX_TOKENS = word2vec.vectors.shape[0]
    # EMBEDDING_DIM = word2vec.vectors.shape[1]
    # embeddings = np.zeros((MAX_TOKENS + 1, EMBEDDING_DIM), dtype="float32")
    # embeddings[:word2vec.vectors.shape[0]] = word2vec.vectors

    HIDDEN_DIM = 100
    ENCODE_DIM = 128
    LABELS = 11
    EPOCHS = 20
    BATCH_SIZE = 2
    USE_GPU = True

    torch.manual_seed(RANDOM_SEED)

    print("Train for classification - CONCAT - With VERSIONALL + CALLGRAPH + NUMOFDAYS ")

    model = BatchProgramClassifier("microsoft/codebert-base", hidden_dim=768, label_size=LABELS)

    train_data = CodeClassificationDataset("train", model.tokenizer)
    val_data = CodeClassificationDataset("dev", model.tokenizer)
    test_data = CodeClassificationDataset("test", model.tokenizer)

    #Load data
    # we dont shuffle the data since we want to keep the order of the data (similar to ASTNN using iterrow())
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=False) 
    dev_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)


    if USE_GPU:
        model.cuda()

    parameters = model.parameters()
    optimizer = torch.optim.Adamax(parameters, lr=1e-5)

    loss_function = torch.nn.CrossEntropyLoss()    

    # print(train_data)
    precision, recall, f1 = 0, 0, 0
    print('Start training...')

    # training procedure
    # state_dict = torch.load(MODEL_DIR + '/class_pure_code.pkl')
    # model.load_state_dict(state_dict)
    best_loss = 10
    best_model = None
    for epoch in range(EPOCHS):
        start_time = time.time()
        # training epoch
        total_acc = 0.0
        total_loss = 0.0
        total = 0.0
        i = 0
        
        for batch in train_loader:
            
            train_code, train_code_versions, train_calling, train_called, train_number_of_days, train_number_of_versions, train_code_versions_all, \
                train_labels = get_batch_transformer(batch)

            if USE_GPU:
                # train1_inputs, train2_inputs, train_labels = train1_inputs, train2_inputs, train_labels.cuda()
                train_labels = train_labels.cuda()

            model.zero_grad()
            model.batch_size = len(train_labels)
            # model.hidden = model.init_hidden()

            output = model(train_code, train_code_versions, train_calling, train_called, train_number_of_days, train_number_of_versions, train_code_versions_all)

            # train_labels = train_labels.squeeze()

            # if train_labels.size(0) != 1:
            train_labels = train_labels.squeeze(1)
            loss = loss_function(output, train_labels.long())

            loss.backward()
            optimizer.step()

            log_prediction = torch.softmax(output, dim=1)
            predicted = torch.argmax(log_prediction, dim=1)
            for idx in range(len(predicted)):
                if predicted[idx] == train_labels[idx]:
                    total_acc += 1
            total += len(train_labels)
            total_loss += loss.item() * len(train_labels)
        train_loss = total_loss / total
        train_acc = total_acc / total

        # dev epoch
        total_acc = 0.0
        total_loss = 0.0
        total = 0.0
        i = 0
        # while i < len(dev_data):
        for batch in dev_loader:
            # batch = get_batch(dev_data, i, BATCH_SIZE)
            # i += BATCH_SIZE
            # dev_code, dev_labels = batch
            # val_inputs, val_labels = batch

            dev_code, dev_code_versions, dev_calling, dev_called, dev_number_of_days, dev_number_of_versions, dev_code_versions_all, \
                dev_labels = get_batch_transformer(batch)
            
            if USE_GPU:
                # val_inputs, val_labels = val_inputs, val_labels.cuda()
                dev_labels = dev_labels.cuda()

            model.batch_size = len(dev_labels)
            # model.hidden = model.init_hidden()
            output = model(dev_code, dev_code_versions, dev_calling, dev_called, dev_number_of_days, dev_number_of_versions, dev_code_versions_all)            

            dev_labels = dev_labels.squeeze(1)
            loss = loss_function(output, dev_labels.long())

            log_prediction = torch.softmax(output, dim=1)
            predicted = torch.argmax(log_prediction, dim=1)
            for idx in range(len(predicted)):
                if predicted[idx] == dev_labels[idx]:
                    total_acc += 1
            total += len(dev_labels)
            total_loss += loss.item() * len(dev_labels)
        epoch_loss = total_loss / total
        epoch_acc = total_acc / total
        end_time = time.time()
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model = model
        print('[Epoch: %3d/%3d] Train Loss: %.4f, Validation Loss: %.4f, '
              'Train Acc: %.3f, Validation Acc: %.3f, Time Cost: %.3f s'
              % (epoch + 1, EPOCHS, train_loss, epoch_loss, train_acc,
                 epoch_acc, end_time - start_time))

    model = best_model
    torch.save(model.state_dict(), MODEL_DIR + '/class_concat.pth')

    """
    test
    """    
    # model = BatchProgramClassifier("microsoft/codebert-base", hidden_dim=768, label_size=LABELS)

    # model.load_state_dict(torch.load(MODEL_DIR + '/class_concat.pth'))

    if USE_GPU:
        model.cuda()

    # testing procedure
    predicts = []
    trues = []
    total_loss = 0.0
    total_acc = 0
    total = 0.0
    i = 0
    # while i < len(test_data):
    for batch in test_loader:
        # batch = get_batch(test_data, i, BATCH_SIZE)
        # i += BATCH_SIZE
        # test1_inputs, test2_inputs, test_labels = batch
        test_code, test_code_versions, test_calling, test_called, test_number_of_days, test_number_of_versions, test_code_versions_all, \
                test_labels = get_batch_transformer(batch)
        
        if USE_GPU:
            test_labels = test_labels.cuda()

        model.batch_size = len(test_labels)
        # model.hidden = model.init_hidden()
        output = model(test_code, test_code_versions, test_calling, test_called, test_number_of_days, test_number_of_versions, test_code_versions_all)

        # test_labels = test_labels.squeeze()
        log_prediction = torch.softmax(output, dim=1)
        predicted = torch.argmax(log_prediction, dim=1)
        for idx in range(len(predicted)):
            if predicted[idx] == test_labels[idx]:
                total_acc += 1
        total += len(test_labels)
        #         predicted = (output.data > 0.5).cpu().numpy()
        predicts.extend(predicted.cpu().detach().numpy())
        trues.extend(test_labels.cpu().numpy())

    acc = total_acc / total
    print(f"Total accuracy: {total_acc}/{int(total)}")
    p, r, f, _ = precision_recall_fscore_support(trues, predicts, average='macro')    
    
    print("Total testing results(acc,P,R,F1):%.5f, %.5f, %.5f, %.5f" % (acc, p, r, f))

    # store model result    
    model_log = "Transformer-CodeBert-VersionAll_CallGraph_NumOfDays-Class Concat, %.5f, %.5f, %.5f, %.5f" % (acc, p, r, f)
    obj = SharedFunction(model_log)
    obj.AppendFile()