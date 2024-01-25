import torch.nn as nn
import torch.nn.functional as F
import torch
# from torch.autograd import Variable
import random
# import pandas as pd
import torch
import time
import numpy as np
import warnings
from gensim.models.word2vec import Word2Vec
from sklearn.metrics import precision_recall_fscore_support
# warnings.filterwarnings('ignore')

from transformers import AutoTokenizer, AutoModel, AdamW, AutoConfig
from torch.utils.data import DataLoader
from dataset import CodeCloneDataset
from utilities import SharedFunction

    
class BatchProgramClassifier(nn.Module):
    def __init__(self,
                 encoder_name,
                 hidden_dim,
                 label_size= 2
                ):
        super(BatchProgramClassifier, self).__init__()
        self.encoder_name = encoder_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.encoder_name)
        self.config = AutoConfig.from_pretrained(self.encoder_name)
        self.config.num_labels = 1

        self.encoder = AutoModel.from_pretrained(self.encoder_name, config = self.config)

        # freeze the encoder
        # for param in self.encoder.parameters():
        #     param.requires_grad = False

        self.hidden_dim = hidden_dim
        self.label_size = label_size

        # self.hidden2label = nn.Linear(self.hidden_dim * 2, self.label_size)
        self.hidden2label = nn.Linear(self.hidden_dim, self.label_size)

    def encode(self, input_ids):
        return self.encoder(input_ids,attention_mask=input_ids.ne(1))[0][:, 0,:]
    
    def forward(self, x1=None, x2=None, x3=None, x4=None, x8=None, x9=None, x10=None, y1=None, y2=None, y3=None, y4=None, y8=None, y9=None, y10=None):
        # outputs = self.encoder(input_ids,attention_mask=input_ids.ne(1))[0]
        l_code, r_code = self.encode(x1), self.encode(y1)

        abs_dist = torch.abs(torch.add(l_code, -r_code))

        y = self.hidden2label(abs_dist)
        
        return y

# def get_batch(dataset, idx, bs):
#     tmp = dataset.iloc[idx: idx+bs]
#     x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, labels = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []    
#     for _, item in tmp.iterrows():
#         x1.append(item['code_x'])
#         x2.append(item['code_versions_x'])
#         x3.append(item['calling_x'])
#         x4.append(item['called_x'])
#         x5.append(item['code_v1_x'])
#         x6.append(item['calling_v1_x'])
#         x7.append(item['called_v1_x'])
#         x8.append(item['number_of_days_x'])
#         x9.append(item['number_of_versions_x'])
#         x10.append(item['code_versions_all_x'])

#         y1.append(item['code_y'])
#         y2.append(item['code_versions_y'])
#         y3.append(item['calling_y'])
#         y4.append(item['called_y'])
#         y5.append(item['code_v1_y'])
#         y6.append(item['calling_v1_y'])
#         y7.append(item['called_v1_y'])
#         y8.append(item['number_of_days_y'])
#         y9.append(item['number_of_versions_y'])
#         y10.append(item['code_versions_all_y'])

#         labels.append([item['label']])
#     return x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, torch.FloatTensor(labels)
#     # return x5, x6, x7, y5, y6, y7, torch.FloatTensor(labels)    

def get_batch_transformer(batch):
    x1 = batch['code_ids_x']
    x2 = batch['code_versions_ids_x']
    x3 = batch['calling_ids_x']
    x4 = batch['called_ids_x']
    # x5 = None
    # x6 = None
    # x7 = None
    x8 = batch['number_of_days_ids_x']
    x9 = batch['number_of_versions_ids_x']
    x10 = batch['code_versions_all_ids_x']

    y1 = batch['code_ids_y']
    y2 = batch['code_versions_ids_y']
    y3 = batch['calling_ids_y']
    y4 = batch['called_ids_y']
    # y5 = None
    # y6 = None
    # y7 = None    
    y8 = batch['number_of_days_ids_y']
    y9 = batch['number_of_versions_ids_y']
    y10 = batch['code_versions_all_ids_y']

    train_labels = batch['label']
    # return x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, train_labels
    return x1, x2, x3, x4, x8, x9, x10, y1, y2, y3, y4, y8, y9, y10, train_labels

if __name__ == '__main__':
    RANDOM_SEED = 2023 
    DATA_DIR = './data/clone_detection'
    MODEL_DIR = './models/transformer_codebert'

    word2vec = Word2Vec.load(DATA_DIR + '/node_w2v_128').wv
    MAX_TOKENS = word2vec.vectors.shape[0]
    EMBEDDING_DIM = word2vec.vectors.shape[1]
    embeddings = np.zeros((MAX_TOKENS + 1, EMBEDDING_DIM), dtype="float32")
    embeddings[:word2vec.vectors.shape[0]] = word2vec.vectors

    HIDDEN_DIM = 100
    ENCODE_DIM = 128
    LABELS = 1
    EPOCHS = 20
    BATCH_SIZE = 2
    USE_GPU = True

    torch.manual_seed(RANDOM_SEED)

    print("Train for clone detection - CODEBERT with BOTH VERSION MAX + CALL GRAPH - Clone Pure Code ")
    
    model = BatchProgramClassifier("microsoft/codebert-base", hidden_dim=768, label_size=LABELS)    
    
    train_data = CodeCloneDataset("train", model.tokenizer)
    val_data = CodeCloneDataset("dev", model.tokenizer)
    test_data = CodeCloneDataset("test", model.tokenizer)
        
    #Load data
    # we dont shuffle the data since we want to keep the order of the data (similar to ASTNN using iterrow())
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=False) 
    dev_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    
    if USE_GPU:
        model.cuda()

    parameters = model.parameters()

    # when freeze encoder, we should use smaller learning rate
    # optimizer = torch.optim.Adamax(parameters, lr=1e-3)
    optimizer = torch.optim.Adamax(parameters, lr=1e-5)

    loss_function = torch.nn.BCEWithLogitsLoss()

    # print(train_data)
    precision, recall, f1 = 0, 0, 0
    print('Start training...')

    # training procedure
    # state_dict = torch.load(MODEL_DIR + '/clone_pure_code.pth')
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
        # while i < len(train_data):
        for batch in train_loader:
            # batch = get_batch_transformer(train_data, i, BATCH_SIZE)
            # i += BATCH_SIZE                        

            # train_code_x, train_code_versions_x, train_calling_x, train_called_x, train_code_v1_x, train_calling_v1_x, train_called_v1_x, \
            #     train_code_y, train_code_versions_y, train_calling_y, train_called_y, train_code_v1_y, train_calling_v1_y, train_called_v1_y, \
            #         train_labels = batch

            # train_code_x = batch['input_ids_x']
            # train_code_y = batch['input_ids_y']
            # train_labels = batch['label']

            train_code_x, train_code_versions_x, train_calling_x, train_called_x, train_number_of_days_x, train_number_of_versions_x, train_code_versions_all_x, \
                train_code_y, train_code_versions_y, train_calling_y, train_called_y, train_number_of_days_y, train_number_of_versions_y, train_code_versions_all_y, \
                    train_labels = get_batch_transformer(batch)
            
            if USE_GPU:
                # train1_inputs, train2_inputs, train_labels = train1_inputs, train2_inputs, train_labels.cuda()
                train_labels = train_labels.cuda()

            model.zero_grad()
            model.batch_size = len(train_labels)
            # model.hidden = model.init_hidden()
            # output = model(train_code_x, train_calling_x, train_called_x, train_code_y, train_calling_y, train_called_y)
            output = model(
                train_code_x, train_code_versions_x, train_calling_x, train_called_x, train_number_of_days_x, train_number_of_versions_x, train_code_versions_all_x, \
                train_code_y, train_code_versions_y, train_calling_y, train_called_y, train_number_of_days_y, train_number_of_versions_y, train_code_versions_all_y, \
                )

            # train_labels = train_labels.squeeze()
            loss = loss_function(output, train_labels.float())
            loss.backward()
            optimizer.step()
            
            # output = output.squeeze()
            output = torch.sigmoid(output)
            predicted = torch.round(output)
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
        for batch in dev_loader:
            # dev_code_x, dev_code_versions_x, dev_calling_x, dev_called_x, dev_code_y, dev_code_versions_y, dev_calling_y, dev_called_y, dev_labels = batch           
            # val_inputs, val_labels = batch           
            
            dev_code_x, dev_code_versions_x, dev_calling_x, dev_called_x, dev_number_of_days_x, dev_number_of_versions_x, dev_code_versions_all_x, \
                dev_code_y, dev_code_versions_y, dev_calling_y, dev_called_y, dev_number_of_days_y, dev_number_of_versions_y, dev_code_versions_all_y, \
                dev_labels = get_batch_transformer(batch)

            if USE_GPU:
                # val_inputs, val_labels = val_inputs, val_labels.cuda()
                dev_labels = dev_labels.cuda()

            model.batch_size = len(dev_labels)
            # model.hidden = model.init_hidden()
            output = model(
                dev_code_x, dev_code_versions_x, dev_calling_x, dev_called_x, dev_number_of_days_x, dev_number_of_versions_x, dev_code_versions_all_x, \
                dev_code_y, dev_code_versions_y, dev_calling_y, dev_called_y, dev_number_of_days_y, dev_number_of_versions_y, dev_code_versions_all_y, \
                )
            # output = model(dev_code_x, dev_code_versions_x, dev_code_y, dev_code_versions_y)

            # dev_labels = dev_labels.squeeze()
            loss = loss_function(output, dev_labels.float())

            # output = output.squeeze()
            output = torch.sigmoid(output)
            predicted = torch.round(output)
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
    torch.save(model.state_dict(), MODEL_DIR + '/clone_pure_code.pth')

    """
    test
    """
    model = BatchProgramClassifier("microsoft/codebert-base", hidden_dim=768, label_size=LABELS)

    # model = BatchProgramClassifier(EMBEDDING_DIM, HIDDEN_DIM, MAX_TOKENS + 1, ENCODE_DIM, LABELS, BATCH_SIZE,
    #                                USE_GPU, embeddings)
                                   
    model.load_state_dict(torch.load(MODEL_DIR + '/clone_pure_code.pth'))

    if USE_GPU:
        model.cuda()

    # testing procedure
    predicts = []
    trues = []
    total_loss = 0.0
    total_acc = 0
    total = 0.0
    i = 0
    for batch in test_loader:
        # test1_inputs, test2_inputs, test_labels = batch
        # test_code_x, test_code_versions_x,test_calling_x, test_called_x, test_code_y, test_code_versions_y, test_calling_y, test_called_y, test_labels = batch
        
        # test_code_x = batch['input_ids_x']
        # test_code_y = batch['input_ids_y']
        # test_labels = batch['label'] 
        test_code_x, test_code_versions_x, test_calling_x, test_called_x, test_number_of_days_x, test_number_of_versions_x, test_code_versions_all_x, \
            test_code_y, test_code_versions_y, test_calling_y, test_called_y, test_number_of_days_y, test_number_of_versions_y, test_code_versions_all_y, \
            test_labels = get_batch_transformer(batch)
        
        if USE_GPU:
            test_labels = test_labels.cuda()

        model.batch_size = len(test_labels)
        # model.hidden = model.init_hidden()
        output = model(
                test_code_x, test_code_versions_x, test_calling_x, test_called_x, test_number_of_days_x, test_number_of_versions_x, test_code_versions_all_x, \
                test_code_y, test_code_versions_y, test_calling_y, test_called_y, test_number_of_days_y, test_number_of_versions_y, test_code_versions_all_y, \
                )
        # output = model(test_code_x, test_code_versions_x, test_code_y, test_code_versions_y)

        output = torch.sigmoid(output)
        predicted = torch.round(output)
        for idx in range(len(predicted)):
            if predicted[idx] == test_labels[idx]:
                total_acc += 1
        total += len(test_labels)
        #         predicted = (output.data > 0.5).cpu().numpy()
        predicts.extend(predicted.cpu().detach().numpy())
        trues.extend(test_labels.cpu().numpy())

    acc = total_acc / total
    print("total accuracy: ", total_acc)
    p, r, f, _ = precision_recall_fscore_support(trues, predicts, average='binary')

    print("Total testing results(acc,P,R,F1):%.5f, %.5f, %.5f, %.5f" % (acc, p, r, f))

    # store model result
    # model_log = f"Transformer-CodeBert-Clone Pure Code, {acc}, {p}, {r}, {f}"
    model_log = "Transformer-CodeBert-BOTH_VersionALl_CallGraph-Clone Pure Code, %.5f, %.5f, %.5f, %.5f" % (acc, p, r, f)
    obj = SharedFunction(model_log)
    obj.AppendFile()