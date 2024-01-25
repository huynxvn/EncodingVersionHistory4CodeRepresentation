from torch.utils.data import Dataset
import json
import os 
import torch 
import pickle as pkl

class InputFeatures(object):
    """
    A single training/test features for a data instance.
    """
    
    # def __init__(self,
    #              input_tokens,
    #              input_ids,
    #              idx,
    #              label,

    # ):
    #     self.input_tokens = input_tokens
    #     self.input_ids = input_ids
    #     self.idx=str(idx)
    #     self.label=label

    def __init__(self,
                 code_tokens_x, code_tokens_y,
                 code_versions_tokens_x, code_versions_tokens_y,
                 calling_token_x, calling_token_y,
                 called_token_x, called_token_y,
                 number_of_days_token_x, number_of_days_token_y,
                 number_of_versions_token_x, number_of_versions_token_y,
                 versions_all_token_x, versions_all_token_y,

                 code_ids_x, code_ids_y,
                 code_versions_ids_x, code_versions_ids_y,
                 calling_ids_x, calling_ids_y,
                 called_ids_x, called_ids_y,
                 number_of_days_ids_x, number_of_days_ids_y,
                 number_of_versions_ids_x, number_of_versions_ids_y,
                 versions_all_ids_x, versions_all_ids_y,

                 idx_x, idx_y,

                 label,

    ):
        # token
        self.code_tokens_x = code_tokens_x
        self.code_versions_tokens_x = code_versions_tokens_x
        self.calling_token_x = calling_token_x
        self.called_token_x = called_token_x
        self.number_of_days_token_x = number_of_days_token_x
        self.number_of_versions_token_x = number_of_versions_token_x
        self.versions_all_token_x = versions_all_token_x

        self.code_tokens_y = code_tokens_y
        self.code_versions_tokens_y = code_versions_tokens_y
        self.calling_token_y = calling_token_y
        self.called_token_y = called_token_y
        self.number_of_days_token_y = number_of_days_token_y
        self.number_of_versions_token_y = number_of_versions_token_y
        self.versions_all_token_y = versions_all_token_y

        #ids_x
        self.code_ids_x = code_ids_x
        self.code_versions_ids_x = code_versions_ids_x
        self.calling_ids_x = calling_ids_x
        self.called_ids_x = called_ids_x
        self.number_of_days_ids_x = number_of_days_ids_x
        self.number_of_versions_ids_x = number_of_versions_ids_x
        self.code_versions_all_ids_x = versions_all_ids_x

        self.code_ids_y = code_ids_y
        self.code_versions_ids_y = code_versions_ids_y
        self.calling_ids_y = calling_ids_y
        self.called_ids_y = called_ids_y
        self.number_of_days_ids_y = number_of_days_ids_y
        self.number_of_versions_ids_y = number_of_versions_ids_y
        self.code_versions_all_ids_y = versions_all_ids_y

        # id num
        self.idx_x=str(idx_x)
        self.idx_y=str(idx_y)

        # label
        self.label=label

class InputFeaturesClassification(object):
    """
    A single training/test features for a data instance.
    """
    
    # def __init__(self,
    #              input_tokens,
    #              input_ids,
    #              idx,
    #              label,

    # ):
    #     self.input_tokens = input_tokens
    #     self.input_ids = input_ids
    #     self.idx=str(idx)
    #     self.label=label

    def __init__(self,
                 code_tokens,
                 code_versions_tokens,
                 calling_token,
                 called_token,
                 number_of_days_token,
                 number_of_versions_token,
                 versions_all_token,

                 code_ids,
                 code_versions_ids,
                 calling_ids,
                 called_ids, 
                 number_of_days_ids,
                 number_of_versions_ids,
                 versions_all_ids, 

                 idx,

                 label,

    ):
        # token
        self.code_tokens = code_tokens
        self.code_versions_tokens = code_versions_tokens
        self.calling_token = calling_token
        self.called_token = called_token
        self.number_of_days_token = number_of_days_token
        self.number_of_versions_token = number_of_versions_token
        self.versions_all_token = versions_all_token
        
        #ids
        self.code_ids = code_ids
        self.code_versions_ids = code_versions_ids
        self.calling_ids = calling_ids
        self.called_ids = called_ids
        self.number_of_days_ids = number_of_days_ids
        self.number_of_versions_ids = number_of_versions_ids
        self.code_versions_all_ids = versions_all_ids

        # id num
        self.idx=str(idx)

        # label
        self.label=label
        
class CodeCloneDataset(Dataset):
    def __init__(self, 
                 data_type, 
                 tokenizer,
                 is_force = True
                 ):
        
        # Config meta data 
        # self.raw_path = f"data/CC/{data_type}.jsonl"
        # self.save_dir = f"processed/CC/{data_type}/"
        # self.save_path = os.path.join(self.save_dir, "data.pkl")
        # self.tokenizer = tokenizer
        # self.data = []

        # self.raw_path = f"./codebert_dataset/data/sesame/split/{data_type}.json"
        self.raw_path = f"./data/clone_detection/{data_type}_blocks.pkl"

        # self.save_dir = f"./codebert_dataset/processed/sesame/{data_type}/"
        self.save_dir = f"./data/clone_detection/processed/{data_type}/"

        self.save_path = os.path.join(self.save_dir, data_type + "_blocks.pkl")
        self.tokenizer = tokenizer
        self.data = []

        #Process
        if self.has_cache() and not is_force:
            self.load()
        else:
            self.process()
            self.save()
    
    def __len__(self):
        """
        Return the total number of data instances
        """
        return len(self.data)
    
    def __getitem__(self, 
                    i
                    ):  
        """
        Return the i-th data instance
        """        
        
        # label = 1 if self.data[i].label > 0 else 0
        label = 1 if self.data[i].label > 0 else 0
        return {
            # "code_ids_x": torch.tensor(self.data[i].code_ids_x, dtype=torch.long).cuda(),
            # "code_versions_ids_x": torch.tensor(self.data[i].code_versions_ids_x, dtype=torch.long).cuda(),
            # "calling_ids_x": torch.tensor(self.data[i].calling_ids_x, dtype=torch.long).cuda(),
            # "called_ids_x": torch.tensor(self.data[i].called_ids_x, dtype=torch.long).cuda(),
            # "number_of_days_ids_x": torch.tensor(self.data[i].number_of_days_ids_x, dtype=torch.long).cuda(),
            # "number_of_versions_ids_x": torch.tensor(self.data[i].number_of_versions_ids_x, dtype=torch.long).cuda(),
            # "code_versions_all_ids_x": torch.tensor(self.data[i].code_versions_all_ids_x, dtype=torch.long).cuda(),           

            # "code_ids_y": torch.tensor(self.data[i].code_ids_y, dtype=torch.long).cuda(),
            # "code_versions_ids_y": torch.tensor(self.data[i].code_versions_ids_y, dtype=torch.long).cuda(),
            # "calling_ids_y": torch.tensor(self.data[i].calling_ids_y, dtype=torch.long).cuda(),
            # "called_ids_y": torch.tensor(self.data[i].called_ids_y, dtype=torch.long).cuda(),
            # "number_of_days_ids_y": torch.tensor(self.data[i].number_of_days_ids_y, dtype=torch.long).cuda(),
            # "number_of_versions_ids_y": torch.tensor(self.data[i].number_of_versions_ids_y, dtype=torch.long).cuda(),
            # "code_versions_all_ids_y": torch.tensor(self.data[i].code_versions_all_ids_y, dtype=torch.long).cuda(),

            "code_ids_x": torch.tensor(self.data[i].code_ids_x, dtype=torch.long).cuda(),
            "code_versions_ids_x": torch.tensor(self.data[i].code_versions_ids_x, dtype=torch.long).cuda(),
            "calling_ids_x": torch.tensor(self.data[i].calling_ids_x, dtype=torch.long).cuda(),
            "called_ids_x": torch.tensor(self.data[i].called_ids_x, dtype=torch.long).cuda(),
            "number_of_days_ids_x": torch.tensor(self.data[i].number_of_days_ids_x, dtype=torch.float).cuda(),
            "number_of_versions_ids_x": torch.tensor(self.data[i].number_of_versions_ids_x, dtype=torch.float).cuda(),
            "code_versions_all_ids_x": torch.tensor(self.data[i].code_versions_all_ids_x, dtype=torch.long).cuda(),           

            "code_ids_y": torch.tensor(self.data[i].code_ids_y, dtype=torch.long).cuda(),
            "code_versions_ids_y": torch.tensor(self.data[i].code_versions_ids_y, dtype=torch.long).cuda(),
            "calling_ids_y": torch.tensor(self.data[i].calling_ids_y, dtype=torch.long).cuda(),
            "called_ids_y": torch.tensor(self.data[i].called_ids_y, dtype=torch.long).cuda(),
            "number_of_days_ids_y": torch.tensor(self.data[i].number_of_days_ids_y, dtype=torch.float).cuda(),
            "number_of_versions_ids_y": torch.tensor(self.data[i].number_of_versions_ids_y, dtype=torch.float).cuda(),
            "code_versions_all_ids_y": torch.tensor(self.data[i].code_versions_all_ids_y, dtype=torch.long).cuda(),

            "label": torch.tensor(label, dtype=torch.long).unsqueeze(0).cuda(),


        }
            
    def load_raw_data(self):
        """
        Load raw data from jsonl file
        """
        
        # data_list = []
        # with open(self.raw_path, 'r') as file:
        #     for line in file:
        #         # Load each line as a JSON object
        #         data = json.loads(line)
        #         data_list.append(data)
        # return data_list

        # load items in a .json file into a list, the input file contain a list of json objects
        # with open(self.raw_path, 'r') as file:
        #     data_list = json.load(file) 
        #     file.close()
        # return data_list
    
        with open(self.raw_path, "rb") as pf:            
            # self.data = pkl.load(pf)["data"]
            data_list = pkl.load(pf)
            pf.close()            
        return data_list
    
    def extract_features(self,
                         js, 
                         tokenizer
                         ):
        """
        Extract features from a data instance (in json format)
        """
        
        # code=' '.join(js['func'].split())
        
        # code_x                  = ' '.join(js['code_x'].split())        
        # code_versions_x         = ' '.join(js['code_versions_x'][0].split())
        # calling_x               = ' '.join(js['calling_x'].split())
        # called_x                = ' '.join(js['called_x'].split())
        # number_of_days_x        = ' '.join(js['number_of_days_x'].split())
        # number_of_versions_x    = ' '.join(js['number_of_versions_x'].split())
        # code_versions_all_x     = ' '.join(js['code_versions_all_x'].split())

        # code_y                  = ' '.join(js['code_y'].split())
        # code_versions_y         = ' '.join(js['code_versions_y'].split())
        # calling_y               = ' '.join(js['calling_y'].split())
        # called_y                = ' '.join(js['called_y'].split())
        # number_of_days_y        = ' '.join(js['number_of_days_y'].split())
        # number_of_versions_y    = ' '.join(js['number_of_versions_x'].split())
        # code_versions_all_y     = ' '.join(js['code_versions_all_y'].split())

        # code_x                  = js['t_code_x']
        # code_versions_x         = js['t_code_versions_x'][0] if len(js['t_code_versions_x']) > 0 else ''
        # calling_x               = js['t_calling_x'][0] if len(js['t_calling_x']) > 0 else ''
        # called_x                = js['t_called_x'][0] if len(js['t_called_x']) > 0 else ''
        # number_of_days_x        = js['t_number_of_days_x'][0] if len(js['t_number_of_days_x']) > 0 else ''
        # number_of_versions_x    = js['t_number_of_versions_x'][0] if len(js['t_number_of_versions_x']) > 0 else ''
        # code_versions_all_x     = js['t_code_versions_all_x'][0] if len(js['t_code_versions_all_x']) > 0 else ''

        # code_y                  = js['t_code_y'] 
        # code_versions_y         = js['t_code_versions_y'][0] if len(js['t_code_versions_y']) > 0 else ''
        # calling_y               = js['t_calling_y'][0] if len(js['t_calling_y']) > 0 else ''
        # called_y                = js['t_called_y'][0] if len(js['t_called_y']) > 0 else ''
        # number_of_days_y        = js['t_number_of_days_y'][0] if len(js['t_number_of_days_y']) > 0 else ''
        # number_of_versions_y    = js['t_number_of_versions_x'][0] if len(js['t_number_of_versions_x']) > 0 else ''
        # code_versions_all_y     = js['t_code_versions_all_y'][0] if len(js['t_code_versions_all_y']) > 0 else ''

        code_x                  = js['t_code_x']
        code_versions_x         = js['t_code_versions_x']
        calling_x               = js['t_calling_x']
        called_x                = js['t_called_x']
        number_of_days_x        = js['number_of_days_x'] # normalised
        number_of_versions_x    = js['number_of_versions_x'] # normalised
        code_versions_all_x     = js['t_code_versions_all_x']

        code_y                  = js['t_code_y'] 
        code_versions_y         = js['t_code_versions_y']
        calling_y               = js['t_calling_y']
        called_y                = js['t_called_y']
        number_of_days_y        = js['number_of_days_y'] # normalised
        number_of_versions_y    = js['number_of_versions_x'] # normalised
        code_versions_all_y     = js['t_code_versions_all_y']

        # print(len(code_x))
        # print(len(code_y))
        
        #Truncation
        # code_tokens=tokenizer.tokenize(code)[:tokenizer.max_len_single_sentence-2]
        code_tokens_x                   = tokenizer.tokenize(code_x)[:tokenizer.max_len_single_sentence-2]
        code_versions_token_x           = tokenizer.tokenize(code_versions_x)[:tokenizer.max_len_single_sentence-2]
        calling_token_x                 = tokenizer.tokenize(calling_x)[:tokenizer.max_len_single_sentence-2]
        called_token_x                  = tokenizer.tokenize(called_x)[:tokenizer.max_len_single_sentence-2]
        # number_of_days_token_x          = tokenizer.tokenize(number_of_days_x)[:tokenizer.max_len_single_sentence-2]
        # number_of_versions_token_x      = tokenizer.tokenize(number_of_versions_x)[:tokenizer.max_len_single_sentence-2]
        code_versions_all_token_x       = tokenizer.tokenize(code_versions_all_x)[:tokenizer.max_len_single_sentence-2]

        code_tokens_y                   = tokenizer.tokenize(code_y)[:tokenizer.max_len_single_sentence-2]
        code_versions_token_y           = tokenizer.tokenize(code_versions_y)[:tokenizer.max_len_single_sentence-2]
        calling_token_y                 = tokenizer.tokenize(calling_y)[:tokenizer.max_len_single_sentence-2]
        called_token_y                  = tokenizer.tokenize(called_y)[:tokenizer.max_len_single_sentence-2]
        # number_of_days_token_y          = tokenizer.tokenize(number_of_days_y)[:tokenizer.max_len_single_sentence-2]
        # number_of_versions_token_y      = tokenizer.tokenize(number_of_versions_y)[:tokenizer.max_len_single_sentence-2]
        code_versions_all_token_y       = tokenizer.tokenize(code_versions_all_y)[:tokenizer.max_len_single_sentence-2]
        
        #Add CLS + SEP
        # source_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
        cs_code_tokens_x =[tokenizer.cls_token]+code_tokens_x+[tokenizer.sep_token]
        cs_code_versions_tokens_x =[tokenizer.cls_token]+code_versions_token_x+[tokenizer.sep_token]
        cs_calling_token_x =[tokenizer.cls_token]+calling_token_x+[tokenizer.sep_token]
        cs_called_token_x =[tokenizer.cls_token]+called_token_x+[tokenizer.sep_token]
        # cs_number_of_days_token_x =[tokenizer.cls_token]+number_of_days_token_x+[tokenizer.sep_token]
        # cs_number_of_versions_token_x =[tokenizer.cls_token]+number_of_versions_token_x+[tokenizer.sep_token]
        cs_code_versions_all_token_x =[tokenizer.cls_token]+code_versions_all_token_x+[tokenizer.sep_token]


        cs_code_tokens_y =[tokenizer.cls_token]+code_tokens_y+[tokenizer.sep_token]
        cs_code_versions_tokens_y =[tokenizer.cls_token]+code_versions_token_y+[tokenizer.sep_token]
        cs_calling_token_y =[tokenizer.cls_token]+calling_token_y+[tokenizer.sep_token]
        cs_called_token_y =[tokenizer.cls_token]+called_token_y+[tokenizer.sep_token]
        # cs_number_of_days_token_y =[tokenizer.cls_token]+number_of_days_token_y+[tokenizer.sep_token]
        # cs_number_of_versions_token_y =[tokenizer.cls_token]+number_of_versions_token_y+[tokenizer.sep_token]
        cs_code_versions_all_token_y =[tokenizer.cls_token]+code_versions_all_token_y+[tokenizer.sep_token]
        
        #Convert tokens to ids
        # source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
        code_ids_x =  tokenizer.convert_tokens_to_ids(cs_code_tokens_x)
        code_versions_ids_x =  tokenizer.convert_tokens_to_ids(cs_code_versions_tokens_x)
        calling_ids_x =  tokenizer.convert_tokens_to_ids(cs_calling_token_x)
        called_ids_x =  tokenizer.convert_tokens_to_ids(cs_called_token_x)
        # number_of_days_ids_x =  tokenizer.convert_tokens_to_ids(cs_number_of_days_token_x)
        # number_of_versions_ids_x =  tokenizer.convert_tokens_to_ids(cs_number_of_versions_token_x)
        code_versions_all_ids_x =  tokenizer.convert_tokens_to_ids(cs_code_versions_all_token_x)

        code_ids_y =  tokenizer.convert_tokens_to_ids(cs_code_tokens_y)
        code_versions_ids_y =  tokenizer.convert_tokens_to_ids(cs_code_versions_tokens_y)
        calling_ids_y =  tokenizer.convert_tokens_to_ids(cs_calling_token_y)
        called_ids_y =  tokenizer.convert_tokens_to_ids(cs_called_token_y)
        # number_of_days_ids_y =  tokenizer.convert_tokens_to_ids(cs_number_of_days_token_y)
        # number_of_versions_ids_y =  tokenizer.convert_tokens_to_ids(cs_number_of_versions_token_y)
        code_versions_all_ids_y =  tokenizer.convert_tokens_to_ids(cs_code_versions_all_token_y)        
        
        #Padding
        # padding_length = tokenizer.max_len_single_sentence - len(source_ids)
        code_padding_length_x = tokenizer.max_len_single_sentence - len(code_ids_x)
        code_versions_padding_length_x = tokenizer.max_len_single_sentence - len(code_versions_ids_x)
        calling_padding_length_x = tokenizer.max_len_single_sentence - len(calling_ids_x)
        called_padding_length_x = tokenizer.max_len_single_sentence - len(called_ids_x)
        # number_of_days_padding_length_x = tokenizer.max_len_single_sentence - len(number_of_days_ids_x)
        # number_of_versions_padding_length_x = tokenizer.max_len_single_sentence - len(number_of_versions_ids_x)
        code_versions_all_padding_length_x = tokenizer.max_len_single_sentence - len(code_versions_all_ids_x)        

        code_padding_length_y = tokenizer.max_len_single_sentence - len(code_ids_y)
        code_versions_padding_length_y = tokenizer.max_len_single_sentence - len(code_versions_ids_y)   
        calling_padding_length_y = tokenizer.max_len_single_sentence - len(calling_ids_y)
        called_padding_length_y = tokenizer.max_len_single_sentence - len(called_ids_y)
        # number_of_days_padding_length_y = tokenizer.max_len_single_sentence - len(number_of_days_ids_y)
        # number_of_versions_padding_length_y = tokenizer.max_len_single_sentence - len(number_of_versions_ids_y)
        code_versions_all_padding_length_y = tokenizer.max_len_single_sentence - len(code_versions_all_ids_y)


        # source_ids+=[tokenizer.pad_token_id]*padding_length
        code_ids_x+=[tokenizer.pad_token_id]*code_padding_length_x
        code_versions_ids_x+=[tokenizer.pad_token_id]*code_versions_padding_length_x
        calling_ids_x+=[tokenizer.pad_token_id]*calling_padding_length_x
        called_ids_x+=[tokenizer.pad_token_id]*called_padding_length_x
        # number_of_days_ids_x+=[tokenizer.pad_token_id]*number_of_days_padding_length_x
        # number_of_versions_ids_x+=[tokenizer.pad_token_id]*number_of_versions_padding_length_x        
        code_versions_all_ids_x+=[tokenizer.pad_token_id]*code_versions_all_padding_length_x

        code_ids_y+=[tokenizer.pad_token_id]*code_padding_length_y
        code_versions_ids_y+=[tokenizer.pad_token_id]*code_versions_padding_length_y
        calling_ids_y+=[tokenizer.pad_token_id]*calling_padding_length_y
        called_ids_y+=[tokenizer.pad_token_id]*called_padding_length_y
        # number_of_days_ids_y+=[tokenizer.pad_token_id]*number_of_days_padding_length_y
        # number_of_versions_ids_y+=[tokenizer.pad_token_id]*number_of_versions_padding_length_y
        code_versions_all_ids_y+=[tokenizer.pad_token_id]*code_versions_all_padding_length_y
        
        # return InputFeatures(source_tokens, source_ids, js['idx'], int(js['target']))

        # 'id1', 'id2', 'label', 'code_x', 'code_versions_x', 'calling_x',
        # 'called_x', 'code_v1_x', 'calling_v1_x', 'called_v1_x',
        # 'number_of_days_x', 'number_of_versions_x', 'code_versions_all_x',
        # 'code_y', 'code_versions_y', 'calling_y', 'called_y', 'code_v1_y',
        # 'calling_v1_y', 'called_v1_y', 'number_of_days_y',
        # 'number_of_versions_y', 'code_versions_all_y'
        
        return InputFeatures(
            cs_code_tokens_x, cs_code_tokens_y,
            cs_code_versions_tokens_x, cs_code_versions_tokens_y,
            cs_calling_token_x, cs_calling_token_y,
            cs_called_token_x, cs_called_token_y,
            number_of_days_x, number_of_days_y,
            number_of_versions_x, number_of_versions_y,
            cs_code_versions_all_token_x, cs_code_versions_all_token_y,

            code_ids_x, code_ids_y, 
            code_versions_ids_x, code_versions_ids_y,
            calling_ids_x, calling_ids_y,
            called_ids_x, called_ids_y,
            number_of_days_x, number_of_days_y,
            number_of_versions_x, number_of_versions_y,
            code_versions_all_ids_x, code_versions_all_ids_y,

            js['id1'], js['id2'], 
            int(js['label']))        
    
    def process(self):
        """
        Process raw data and save to self.data
        """
        # json_data = self.load_raw_data()
        
        # for data in json_data:
        #     features = self.extract_features(data, self.tokenizer)
        #     self.data.append(features)

        json_data = self.load_raw_data()

        for data in json_data.to_dict('records'):
            features = self.extract_features(data, self.tokenizer)
            self.data.append(features)
    
    def save(self):
        """
        Save processed data to disk
        """
        
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        with open(self.save_path, "wb") as pf:
            pkl.dump({'data': self.data}, pf)
            
    def load(self):
        """
        Load processed data from disk
        """
        with open(self.save_path, "rb") as pf:            
            # self.data = pkl.load(pf)["data"]
            self.data = pkl.load(pf)
            
    
    def has_cache(self):
        """
        Check whether there is cached data
        """
        if os.path.exists(self.save_path):
            print("Data exists")
            return True
        return False

class CodeClassificationDataset(Dataset):
    def __init__(self, 
                 data_type, 
                 tokenizer,
                 is_force = True
                 ):
        
        # Config meta data 
        # self.raw_path = f"data/CC/{data_type}.jsonl"
        # self.save_dir = f"processed/CC/{data_type}/"
        # self.save_path = os.path.join(self.save_dir, "data.pkl")
        # self.tokenizer = tokenizer
        # self.data = []

        # self.raw_path = f"./codebert_dataset/data/sesame/split/{data_type}.json"
        self.raw_path = f"./data/classification/{data_type}_df.pkl"

        # self.save_dir = f"./codebert_dataset/processed/sesame/{data_type}/"
        self.save_dir = f"./data/classification/processed/{data_type}/"

        self.save_path = os.path.join(self.save_dir, data_type + "_blocks.pkl")
        self.tokenizer = tokenizer
        self.data = []

        #Process
        if self.has_cache() and not is_force:
            self.load()
        else:
            self.process()
            self.save()
    
    def __len__(self):
        """
        Return the total number of data instances
        """
        return len(self.data)
    
    def __getitem__(self, 
                    i
                    ):  
        """
        Return the i-th data instance
        """        
        
        # label = 1 if self.data[i].label > 0 else 0
        label = self.data[i].label
        
        return {
            # "code_ids_x": torch.tensor(self.data[i].code_ids_x, dtype=torch.long).cuda(),
            # "code_versions_ids_x": torch.tensor(self.data[i].code_versions_ids_x, dtype=torch.long).cuda(),
            # "calling_ids_x": torch.tensor(self.data[i].calling_ids_x, dtype=torch.long).cuda(),
            # "called_ids_x": torch.tensor(self.data[i].called_ids_x, dtype=torch.long).cuda(),
            # "number_of_days_ids_x": torch.tensor(self.data[i].number_of_days_ids_x, dtype=torch.long).cuda(),
            # "number_of_versions_ids_x": torch.tensor(self.data[i].number_of_versions_ids_x, dtype=torch.long).cuda(),
            # "code_versions_all_ids_x": torch.tensor(self.data[i].code_versions_all_ids_x, dtype=torch.long).cuda(),

            "code_ids": torch.tensor(self.data[i].code_ids, dtype=torch.long).cuda(),
            "code_versions_ids": torch.tensor(self.data[i].code_versions_ids, dtype=torch.long).cuda(),
            "calling_ids": torch.tensor(self.data[i].calling_ids, dtype=torch.long).cuda(),
            "called_ids": torch.tensor(self.data[i].called_ids, dtype=torch.long).cuda(),
            "number_of_days_ids": torch.tensor(self.data[i].number_of_days_ids, dtype=torch.float).cuda(),
            "number_of_versions_ids": torch.tensor(self.data[i].number_of_versions_ids, dtype=torch.float).cuda(),
            "code_versions_all_ids": torch.tensor(self.data[i].code_versions_all_ids, dtype=torch.long).cuda(),           

            "label": torch.tensor(label, dtype=torch.long).unsqueeze(0).cuda(),
        }
            
    def load_raw_data(self):
        """
        Load raw data from pkl file
        """
        
        # data_list = []
        # with open(self.raw_path, 'r') as file:
        #     for line in file:
        #         # Load each line as a JSON object
        #         data = json.loads(line)
        #         data_list.append(data)
        # return data_list

        # load items in a .json file into a list, the input file contain a list of json objects
        # with open(self.raw_path, 'r') as file:
        #     data_list = json.load(file) 
        #     file.close()
        # return data_list
    
        with open(self.raw_path, "rb") as pf:            
            # self.data = pkl.load(pf)["data"]
            data_list = pkl.load(pf)
            pf.close()            
        return data_list
    
    def extract_features(self,
                         js, 
                         tokenizer
                         ):
        """
        Extract features from a data instance (in json format)
        """
        
        # code=' '.join(js['func'].split())        

        code                  = js['t_code']
        code_versions         = js['t_code_versions']
        calling               = js['t_calling']
        called                = js['t_called']
        number_of_days        = js['number_of_days'] # normalised
        number_of_versions    = js['number_of_versions'] # normalised
        code_versions_all     = js['t_code_versions_all']

        ### -----------------------------------------------------------------------------------------------------
        #Truncation
        # code_tokens=tokenizer.tokenize(code)[:tokenizer.max_len_single_sentence-2]
        code_tokens                   = tokenizer.tokenize(code)[:tokenizer.max_len_single_sentence-2]
        code_versions_token           = tokenizer.tokenize(code_versions)[:tokenizer.max_len_single_sentence-2]
        calling_token                 = tokenizer.tokenize(calling)[:tokenizer.max_len_single_sentence-2]
        called_token                  = tokenizer.tokenize(called)[:tokenizer.max_len_single_sentence-2]
        # number_of_days_token          = tokenizer.tokenize(number_of_days)[:tokenizer.max_len_single_sentence-2]
        # number_of_versions_token      = tokenizer.tokenize(number_of_versions)[:tokenizer.max_len_single_sentence-2]
        code_versions_all_token       = tokenizer.tokenize(code_versions_all)[:tokenizer.max_len_single_sentence-2]
        
        ### -----------------------------------------------------------------------------------------------------
        #Add CLS + SEP
        # source_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
        cs_code_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
        cs_code_versions_tokens =[tokenizer.cls_token]+code_versions_token+[tokenizer.sep_token]
        cs_calling_token =[tokenizer.cls_token]+calling_token+[tokenizer.sep_token]
        cs_called_token =[tokenizer.cls_token]+called_token+[tokenizer.sep_token]
        # cs_number_of_days_token =[tokenizer.cls_token]+number_of_days_token+[tokenizer.sep_token]
        # cs_number_of_versions_token =[tokenizer.cls_token]+number_of_versions_token+[tokenizer.sep_token]
        cs_code_versions_all_token =[tokenizer.cls_token]+code_versions_all_token+[tokenizer.sep_token]
        
        ### -----------------------------------------------------------------------------------------------------
        #Convert tokens to ids
        # source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
        code_ids =  tokenizer.convert_tokens_to_ids(cs_code_tokens)
        code_versions_ids =  tokenizer.convert_tokens_to_ids(cs_code_versions_tokens)
        calling_ids =  tokenizer.convert_tokens_to_ids(cs_calling_token)
        called_ids =  tokenizer.convert_tokens_to_ids(cs_called_token)
        # number_of_days_ids =  tokenizer.convert_tokens_to_ids(cs_number_of_days_token)
        # number_of_versions_ids =  tokenizer.convert_tokens_to_ids(cs_number_of_versions_token)
        code_versions_all_ids =  tokenizer.convert_tokens_to_ids(cs_code_versions_all_token)
        
        ### -----------------------------------------------------------------------------------------------------
        #Padding
        # padding_length = tokenizer.max_len_single_sentence - len(source_ids)
        code_padding_length = tokenizer.max_len_single_sentence - len(code_ids)
        code_versions_padding_length = tokenizer.max_len_single_sentence - len(code_versions_ids)
        calling_padding_length = tokenizer.max_len_single_sentence - len(calling_ids)
        called_padding_length = tokenizer.max_len_single_sentence - len(called_ids)
        # number_of_days_padding_length = tokenizer.max_len_single_sentence - len(number_of_days_ids)
        # number_of_versions_padding_length = tokenizer.max_len_single_sentence - len(number_of_versions_ids)
        code_versions_all_padding_length = tokenizer.max_len_single_sentence - len(code_versions_all_ids)

        ### -----------------------------------------------------------------------------------------------------
        # source_ids+=[tokenizer.pad_token_id]*padding_length
        code_ids+=[tokenizer.pad_token_id]*code_padding_length
        code_versions_ids+=[tokenizer.pad_token_id]*code_versions_padding_length
        calling_ids+=[tokenizer.pad_token_id]*calling_padding_length
        called_ids+=[tokenizer.pad_token_id]*called_padding_length
        # number_of_days_ids+=[tokenizer.pad_token_id]*number_of_days_padding_length
        # number_of_versions_ids+=[tokenizer.pad_token_id]*number_of_versions_padding_length
        code_versions_all_ids+=[tokenizer.pad_token_id]*code_versions_all_padding_length
        
        # return InputFeaturesClassification(source_tokens, source_ids, js['idx'], int(js['target']))        
        
        return InputFeaturesClassification(
            cs_code_tokens,
            cs_code_versions_tokens, 
            cs_calling_token,
            cs_called_token,
            number_of_days,
            number_of_versions,
            cs_code_versions_all_token,

            code_ids,
            code_versions_ids,
            calling_ids,
            called_ids,
            number_of_days,
            number_of_versions,
            code_versions_all_ids,

            js['id'],
            int(js['label']))        
    
    def process(self):
        """
        Process raw data and save to self.data
        """
        # json_data = self.load_raw_data()
        
        # for data in json_data:
        #     features = self.extract_features(data, self.tokenizer)
        #     self.data.append(features)

        json_data = self.load_raw_data()
        
        for data in json_data.to_dict('records'):
            features = self.extract_features(data, self.tokenizer)
            self.data.append(features)        
    
    def save(self):
        """
        Save processed data to disk
        """
        
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        with open(self.save_path, "wb") as pf:
            pkl.dump({'data': self.data}, pf)
            
    def load(self):
        """
        Load processed data from disk
        """
        with open(self.save_path, "rb") as pf:            
            # self.data = pkl.load(pf)["data"]
            self.data = pkl.load(pf)
            
    
    def has_cache(self):
        """
        Check whether there is cached data
        """
        if os.path.exists(self.save_path):
            print("Data exists")
            return True
        return False    
