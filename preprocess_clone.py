import javalang
import json
import os
import pandas as pd
import numpy as np

class Pipeline:
    def __init__(self, data_path, tree_file_path, output_dir, ratio, random_seed, embedding_size, tree_exists=False):
        self.tree_exists = tree_exists
        self.tree_file_path = tree_file_path
        self.data_path = data_path
        self.output_dir = output_dir
        self.ratio = ratio
        self.seed = random_seed
        self.size = embedding_size
        self.dataset = None
        self.tree_ds = None
        self.pair_ds = None
        self.train_pairs = None
        self.dev_pairs = None
        self.test_pairs = None

        self.min_val_days = None
        self.max_val_days = None
        self.min_val_vers = None
        self.max_val_vers = None

    def extract_code_tree(self):
        with open(self.data_path, 'r', encoding='utf-8') as input_file:
            self.dataset = json.load(input_file)            
            print("original", len(self.dataset))
            
            tmp_df = pd.DataFrame(self.dataset)
            
            # keep_df = tmp_df[(tmp_df['callgraph_available']==1) & (tmp_df['callgraph_available_v1']==1)].copy()
            keep_df = tmp_df[(tmp_df['callgraph_available']==1)].copy()
            # keep_df = tmp_df.copy()
            keep_df.reset_index(drop=True, inplace=True)
            
            self.dataset = keep_df.to_dict('records')            
            print("after", len(self.dataset))

        if self.tree_exists:
            if os.path.exists(self.tree_file_path):
                self.tree_ds = pd.read_pickle(self.tree_file_path)                
                return self.tree_ds
            else:
                print('Warning: The path you specify to load tree dataset does not exist.')

        def process_context_code(code_object):
            def parse_program(func):
                tokens = javalang.tokenizer.tokenize(func)
                parser = javalang.parser.Parser(tokens)
                tree = parser.parse_member_declaration()
                return tree

            # original code
            try:                
                original_tree = parse_program(code_object['code'])
                # original_tree = parse_program(code_object['version_history_context'][0]['commit_source_code'])
            except Exception:
                # print(code_object['dbid'], code_object['file'], code_object['method'])
                # print(code_object['version_history_context'][0]['commit_source_code'])
                print(f"Warning::original_code:: No. {code_object['dbid']} target cannot be parsed!")
                return code_object['dbid'], None, None, None, None, None, None, None, None, None, None
            
            # version history
            code_versions_trees = []           

            i = 0            
            for item in code_object['version_history_context']:
                if i >= 1: # skip version 0 as current version                    
                    try:                        
                        temp_tree = parse_program(item['commit_source_code'])                        
                        code_versions_trees.append(temp_tree)
                    except Exception:                        
                        print(f'Warning::version_history:: The version history context {item["commit_version_no"]} | {code_object["dbid"]} {code_object["file"]} | {code_object["method"]} cannot be parsed!')
                i+=1
            
            # call graph
            calling_trees = []
            called_trees = []

            for tag, method, context_code in code_object['callgraph_context']:
                try:
                    temp_tree = parse_program(context_code)
                    if tag == 0:
                        calling_trees.append(temp_tree)
                    elif tag == 1:
                        called_trees.append(temp_tree)
                except Exception:
                    print(f'Warning::call_graph:: The callgraph context {method} cannot be parsed!')
            
            flg_obj = javalang.tree.MethodDeclaration()
            if calling_trees == []:
                calling_trees.append(flg_obj)
            if called_trees == []:
                called_trees.append(flg_obj)
            
            ### code and callgraph from 2022's approach            
            try:                
                original_tree_v1 = parse_program(code_object['callgraph_code_v1'])
            except Exception:                
                print(f"Warning::V1 code:: No. {code_object['dbid']} target cannot be parsed!")                
                # return code_object['dbid'], None, None, None, None, None, None, None, None, None, None
            
            # callgraph v1
            calling_trees_v1 = []
            called_trees_v1 = []
            
            for tag, method, context_code in code_object['callgraph_context_v1']:
                try:
                    temp_tree_v1 = parse_program(context_code)
                    if tag == 0:
                        calling_trees_v1.append(temp_tree_v1)
                    elif tag == 1:
                        called_trees_v1.append(temp_tree_v1)
                except Exception:
                    print(f'Warning::V1 call_graph:: The callgraph context V1 {method} cannot be parsed!')
            
            flg_obj = javalang.tree.MethodDeclaration()
            if calling_trees_v1 == []:
                calling_trees_v1.append(flg_obj)
            if called_trees_v1 == []:
                called_trees_v1.append(flg_obj)

            # number of existing days
            number_of_days_tree = np.array([code_object['days_to_exist']])

            # number of versions
            number_of_versions_tree = np.array([code_object['number_of_versions']])
            
            # i = 0            
            # str_all_ver = ""
            # for item in code_object['version_history_context']:
            #     if i >= 1: # skip version 0 as current version
            #         str_all_ver += item['commit_source_code']                    
            #     i+=1            
            # try:
            #     code_versions_all_trees = parse_program(str_all_ver)
            # except Exception:                
            #     print(f'Warning::all versions to str:: The version history context {item["commit_version_no"]} | {code_object["dbid"]} {code_object["file"]} | {code_object["method"]} cannot be parsed!')
            
            code_versions_all_trees = []
            i = 0            
            for item in code_object['version_history_context']:
                if i >= 1: # skip version 0 as current version                    
                    try:                        
                        temp_tree = parse_program(item['commit_source_code'])                        
                        code_versions_all_trees.append(temp_tree)
                    except Exception:                        
                        print(f'Warning::version_history:: The version history context {item["commit_version_no"]} | {code_object["dbid"]} {code_object["file"]} | {code_object["method"]} cannot be parsed!')
                i+=1
            
            # transformer:
            t_code = code_object['code']
            t_code_versions = [v['commit_source_code'] for v in code_object['version_history_context'][1:]]

            t_calling = []
            t_called = []            
            for tag, method, context_code in code_object['callgraph_context']:
                if tag == 0:
                    t_calling.append(context_code)
                elif tag == 1:
                    t_called.append(context_code)                
            if t_calling == []:
                t_calling.append('')
            if t_called == []:
                t_called.append('')

            t_code_v1 = code_object['callgraph_code_v1']
            t_calling_v1 = []
            t_called_v1 = []
            for tag, method, context_code in code_object['callgraph_context_v1']:
                if tag == 0:
                    t_calling_v1.append(context_code)
                elif tag == 1:
                    t_called_v1.append(context_code)
            if t_calling_v1 == []:
                t_calling_v1.append('')
            if t_called_v1 == []:
                t_called_v1.append('')
            
            t_number_of_days = np.array([code_object['days_to_exist']])
            t_number_of_versions = np.array([code_object['number_of_versions']])
            t_code_versions_all =""
            for item in code_object['version_history_context'][1:]:
                t_code_versions_all += item['commit_source_code']

            # return code_object['dbid'], original_tree, code_versions_trees, calling_trees, called_trees
            return code_object['dbid'], original_tree, code_versions_trees, calling_trees, called_trees, \
                                        original_tree_v1, calling_trees_v1, called_trees_v1, \
                                        number_of_days_tree, number_of_versions_tree, code_versions_all_trees, \
                                        t_code, t_code_versions, t_calling, t_called, t_code_v1, t_calling_v1, t_called_v1, \
                                        t_number_of_days, t_number_of_versions, t_code_versions_all
    
     
        tree_array = []
        record = []

        # excluded list
        excluded_list = []
                
        for sample in self.dataset:
            for number in ['first', 'second']:
                code_ob = sample[number]
                
                if code_ob['dbid'] in record:  # avoid duplicate                    
                    continue
                # dbid, original_tree, code_versions_trees, calling_trees, called_trees, original_tree_v1, calling_trees_v1, called_trees_v1, number_of_days_tree, number_of_versions_tree, code_versions_all_trees = process_context_code(code_ob)
                # tree_array.append([int(dbid), original_tree, code_versions_trees, calling_trees, called_trees, original_tree_v1, calling_trees_v1, called_trees_v1, number_of_days_tree, number_of_versions_tree, code_versions_all_trees])
                # record.append(dbid)

                try:

                    # dbid, original_tree, code_versions_trees, calling_trees, called_trees = process_context_code(code_ob)
                    # dbid, original_tree, code_versions_trees, calling_trees, called_trees, original_tree_v1, calling_trees_v1, called_trees_v1 = process_context_code(code_ob)
                    dbid, original_tree, code_versions_trees, calling_trees, called_trees, original_tree_v1, calling_trees_v1, called_trees_v1, number_of_days_tree, number_of_versions_tree, code_versions_all_trees, \
                        t_code, t_code_versions, t_calling, t_called, t_code_v1, t_calling_v1, t_called_v1, t_number_of_days, t_number_of_versions, t_code_versions_all = process_context_code(code_ob)                    

                    # tree_array.append([int(dbid), original_tree, code_versions_trees, calling_trees, called_trees])
                    # tree_array.append([int(dbid), original_tree, code_versions_trees, calling_trees, called_trees, original_tree_v1, calling_trees_v1, called_trees_v1])
                    tree_array.append([int(dbid), original_tree, code_versions_trees, calling_trees, called_trees, original_tree_v1, calling_trees_v1, called_trees_v1, number_of_days_tree, number_of_versions_tree, code_versions_all_trees, \
                                        t_code, t_code_versions, t_calling, t_called, t_code_v1, t_calling_v1, t_called_v1, t_number_of_days, t_number_of_versions, t_code_versions_all])

                    record.append(dbid)
                    # print(dbid, len(code_versions_trees))

                except Exception:
                    # print(code_ob['dbid'], code_ob['file'], code_ob['method'])
                    excluded_list.append([code_ob['dbid'], code_ob['project'], code_ob['file'], code_ob['method'], code_ob['uniqueid']])
                    print(f"Warning::======>:: No. {code_ob['dbid']} target cannot be parsed!")
                    print(f"------------------------------------------------------------------")
                
        # new_df = pd.DataFrame(tree_array, columns=['id', 'code', 'code_versions', 'calling', 'called'])
        # new_df = pd.DataFrame(tree_array, columns=['id', 'code', 'code_versions', 'calling', 'called', 'code_v1', 'calling_v1', 'called_v1'])
        new_df = pd.DataFrame(tree_array, columns=['id', 'code', 'code_versions', 'calling', 'called', 'code_v1', 'calling_v1', 'called_v1', 'number_of_days', 'number_of_versions', 'code_versions_all',  
                                                   't_code', 't_code_versions', 't_calling', 't_called', 't_code_v1', 't_calling_v1', 't_called_v1', 't_number_of_days', 't_number_of_versions', 't_code_versions_all'])

        # new_df = new_df.loc[pd.notnull(new_df['code']), ['id', 'code', 'code_versions', 'calling', 'called']]
        # new_df = new_df.loc[pd.notnull(new_df['code']), ['id', 'code', 'code_versions', 'calling', 'called', 'code_v1', 'calling_v1', 'called_v1']]
        
        new_df = new_df.loc[pd.notnull(new_df['code']), ['id', 'code', 'code_versions', 'calling', 'called', 'code_v1', 'calling_v1', 'called_v1', 'number_of_days', 'number_of_versions', 'code_versions_all', 
                                                         't_code', 't_code_versions', 't_calling', 't_called', 't_code_v1', 't_calling_v1', 't_called_v1', 't_number_of_days', 't_number_of_versions', 't_code_versions_all']]
        
        self.tree_ds = new_df

        if not os.path.exists(os.path.dirname(self.tree_file_path)):
            os.mkdir(os.path.dirname(self.tree_file_path))
        self.tree_ds.to_pickle(self.tree_file_path)

        print("Excluded list:", excluded_list)

        return self.tree_ds

    def extract_pair(self):
        data_list = []
        confidence_map = {0: 0.6, 1: 0.8, 2: 1}
        tree_df = self.tree_ds
        id_list = list(tree_df['id'].values)        
        for json_dict in self.dataset:
            accumulate = 0
            total_weight = 0
            for field in ['goals', 'operations', 'effects']:
                for rating_object in json_dict[field]:
                    if rating_object['rating'] != -1:
                        accumulate += rating_object['rating'] * confidence_map[rating_object['confidence']]
                        total_weight += confidence_map[rating_object['confidence']]
            score = round(accumulate / total_weight)
            data_ins = [
                int(json_dict['first']['dbid']),
                int(json_dict['second']['dbid']),
                score
            ]

            if data_ins[0] in id_list and data_ins[1] in id_list:
                data_list.append(data_ins)
        self.pair_ds = pd.DataFrame(data_list, columns=['id1', 'id2', 'label'])
        
        # print("---->", self.pair_ds.shape)
        # print("---->", self.pair_ds.head())

        return self.pair_ds

    def split_data(self):
        data = self.pair_ds
        data_num = len(data)
        ratios = [int(r) for r in self.ratio.split(':')]
        train_split = int(ratios[0]/sum(ratios)*data_num)
        val_split = train_split + int(ratios[1]/sum(ratios)*data_num)

        data = data.sample(frac=1, random_state=self.seed)
        self.train_pairs = data.iloc[:train_split]
        self.dev_pairs = data.iloc[train_split:val_split]
        self.test_pairs = data.iloc[val_split:]

        # print("---->", self.train_pairs.shape)
        # print(self.train_pairs.head(5))
        # print("---->", self.dev_pairs.shape)
        # print("---->", self.test_pairs.shape)

    def dictionary_and_embedding(self):
        pairs = self.train_pairs
        # print("---->", pairs.shape)
        # print(pairs.head(5))
        # print("---->", pairs['id1'].append(pairs['id2']))

        # train_ids = pairs['id1'].append(pairs['id2']).unique()
        # https://stackoverflow.com/questions/76102473/how-to-fix-attributeerror-series-object-has-no-attribute-append
        train_ids = pairs['id1']._append(pairs['id2']).unique()
        # train_ids = pd.concat([pairs['id1'], pairs['id2']]).unique()

        trees = self.tree_ds.set_index('id', drop=False).loc[train_ids]
        from utils import get_sequence as func

        def trans_to_sequences(ast):
            sequence = []
            func(ast, sequence)
            return sequence

        trees_array = trees.values
        corpus = []
        for i, tree_sample in enumerate(trees_array):
            # if i == 0: 
            #     print(tree_sample)
            for code_versions_tree in tree_sample[2]: # code_versions
                ins_seq = trans_to_sequences(code_versions_tree)
                corpus.append(ins_seq)

            ins_seq = trans_to_sequences(tree_sample[1]) # code
            corpus.append(ins_seq)

            for calling_tree in tree_sample[3]: # calling
                ins_seq = trans_to_sequences(calling_tree)
                corpus.append(ins_seq)

            for called_tree in tree_sample[4]: # called
                ins_seq = trans_to_sequences(called_tree)
                corpus.append(ins_seq)

            # callgraph V1 from 2022's approach
            ins_seq_v1 = trans_to_sequences(tree_sample[5]) # code_v1
            corpus.append(ins_seq_v1)

            for calling_tree in tree_sample[6]: # calling_v1
                ins_seq_v1 = trans_to_sequences(calling_tree)
                corpus.append(ins_seq_v1)

            for called_tree in tree_sample[7]: # called_v1
                ins_seq_v1 = trans_to_sequences(called_tree)
                corpus.append(ins_seq_v1)

            # for code_versions_tree in tree_sample[10]: # code_versions_all as a list
            #     ins_seq = trans_to_sequences(code_versions_tree)
            #     corpus.append(ins_seq)

            # ins_seq = trans_to_sequences(tree_sample[10]) # code_versions_all as a string
            # corpus.append(ins_seq)

            for code_versions_all_tree in tree_sample[10]: # code_versions_all as a list
                ins_seq = trans_to_sequences(code_versions_all_tree)
                corpus.append(ins_seq)

            # # ins_seq_version_all = trans_to_sequences(tree_sample[10]) # code_versions_all as a string
            # # corpus.append(ins_seq_version_all)

        from gensim.models.word2vec import Word2Vec
        # w2v = Word2Vec(corpus, size=self.size, workers=16, sg=1, max_final_vocab=3000)
        # https://stackoverflow.com/questions/53195906/getting-init-got-an-unexpected-keyword-argument-document-this-error-in
        w2v = Word2Vec(corpus, vector_size=self.size, workers=16, sg=1, max_final_vocab=3000)
        
        w2v.save(self.output_dir+'/node_w2v_' + str(self.size))

    def generate_block_seqs(self):
        from utils import get_blocks_v1 as func
        from gensim.models.word2vec import Word2Vec

        word2vec = Word2Vec.load(self.output_dir+'/node_w2v_' + str(self.size)).wv
        # vocab = word2vec.vocab
        # https://stackoverflow.com/questions/66868221/gensim-3-8-0-to-gensim-4-0-0
        vocab = list(word2vec.index_to_key)
        # vocab = list(word2vec.key_to_index.keys())
        
        # max_token = word2vec.syn0.shape[0]
        max_token = word2vec.vectors.shape[0]
        # print("my max_token:", max_token)
        # print("------------------")

        def tree_to_index(node):            
            token = node.token            
            # result = [vocab[token].index if token in vocab else max_token]
            result = [vocab.index(token) if token in vocab else max_token]
            children = node.children
            for child in children:
                result.append(tree_to_index(child))
            return result

        def trans2seq(r):
            blocks = []
            func(r, blocks)
            tree = []            
            for b in blocks:
                btree = tree_to_index(b)
                tree.append(btree)
            return tree

        def trans2seqs(r):
            ret = []
            for ins_r in r:
                tree = trans2seq(ins_r)
                ret.append(tree)
            return ret

        trees = pd.DataFrame(self.tree_ds, copy=True)
        # print("---->", trees.shape)
        # print(trees.head(5))

        # apply trans2seq
        trees['code'] = trees['code'].apply(trans2seq)
        trees['code_versions'] = trees['code_versions'].apply(trans2seqs)
        
        trees['calling'] = trees['calling'].apply(trans2seqs)
        trees['called'] = trees['called'].apply(trans2seqs)

        # callgraph V1 from 2022's approach
        trees['code_v1'] = trees['code_v1'].apply(trans2seq)

        trees['calling_v1'] = trees['calling_v1'].apply(trans2seqs)
        trees['called_v1'] = trees['called_v1'].apply(trans2seqs)

        # number of existing days        
        # trees['number_of_days'] = trees['number_of_days'].tolist()

        # number of versions
        # trees['number_of_versions'] = trees['number_of_versions'].tolist()

        # all versions of code
        # trees['code_versions_all'] = trees['code_versions_all'].apply(trans2seq) # as a string
        trees['code_versions_all'] = trees['code_versions_all'].apply(trans2seqs)  

        # Save only the longest context
        trees_array = trees.values
        
        # trees_list = []
        for block_sample in trees_array:
            
            # code_versions
            max_tree_length = 0
            max_tree = []            
            i = 0
            flg = 0
            for code_versions_tree in block_sample[2]:
                tree_length = sum([len(statement) for statement in code_versions_tree])
                if tree_length > max_tree_length:
                    max_tree = code_versions_tree
                    max_tree_length = tree_length
                    flg = i
                i+=1                
            block_sample[2] = max_tree
            block_sample[12] = block_sample[12][flg]

            # calling
            max_tree_length = 0
            max_tree = []
            i = 0
            flg = 0
            for calling_tree in block_sample[3]:
                tree_length = sum([len(statement) for statement in calling_tree])
                if tree_length > max_tree_length:
                    max_tree = calling_tree
                    max_tree_length = tree_length
                    flg = i
                i+=1            
            block_sample[3] = max_tree # if max_tree != []
            block_sample[13] = block_sample[13][flg] # if max_tree != [] else ''
            

            # called
            max_tree_length = 0
            max_tree = []
            i = 0
            flg = 0
            for called_tree in block_sample[4]:
                tree_length = sum([len(statement) for statement in called_tree])
                if tree_length > max_tree_length:
                    max_tree = called_tree
                    max_tree_length = tree_length
                    flg = i
                i+=1            
            block_sample[4] = max_tree # if max_tree != []
            block_sample[14] = block_sample[14][flg] # if max_tree != [] else ''           

            # callgraph V1 from 2022's approach
            # calling_v1
            max_tree_length = 0
            max_tree = []
            i = 0
            flg = 0
            for calling_tree_v1 in block_sample[6]:
                tree_length = sum([len(statement) for statement in calling_tree_v1])
                if tree_length > max_tree_length:
                    max_tree = calling_tree_v1
                    max_tree_length = tree_length
                    flg = i
                i+=1
            block_sample[6] = max_tree # if max_tree != []
            block_sample[16] = block_sample[16][flg] # if max_tree != [] else ''

            # called_v1
            max_tree_length = 0
            max_tree = []
            i = 0
            flg = 0
            for called_tree_v1 in block_sample[7]:
                tree_length = sum([len(statement) for statement in called_tree_v1])
                if tree_length > max_tree_length:
                    max_tree = called_tree_v1
                    max_tree_length = tree_length
                    flg = i
                i+=1
            block_sample[7] = max_tree # if max_tree != []
            block_sample[17] = block_sample[17][flg] # if max_tree != [] else ''

            # code_versions_all            
            merged_trees = []            
            for code_versions_tree in block_sample[10]:
                merged_trees.append(code_versions_tree)
                         
            block_sample[10] = merged_trees
            

        # with open('./data/sesame_tokens_with_context.json', 'w', encoding='utf-8') as out_file:
        #     json.dump(trees_list, out_file, indent=4)
        
        # trees = pd.DataFrame(trees_array, columns=['id', 'code', 'calling', 'called'])
        # trees = pd.DataFrame(trees_array, columns=['id', 'code', 'code_versions', 'calling', 'called'])
        # trees = pd.DataFrame(trees_array, columns=['id', 'code', 'code_versions', 'calling', 'called', 'code_v1', 'calling_v1', 'called_v1'])
        trees = pd.DataFrame(trees_array, columns=['id', 'code', 'code_versions', 'calling', 'called', 'code_v1', 'calling_v1', 'called_v1', 'number_of_days', 'number_of_versions', 'code_versions_all',
                                                   't_code', 't_code_versions', 't_calling', 't_called', 't_code_v1', 't_calling_v1', 't_called_v1', 't_number_of_days', 't_number_of_versions', 't_code_versions_all'])
        # trees = pd.DataFrame(trees_array, columns=['id', 
        #                 'code', 'code_versions', 'calling', 'called', 'code_v1', 'calling_v1', 'called_v1', 'number_of_days', 'number_of_versions', 'code_versions_all',
        #                 't_code', 't_code_versions', 't_calling', 't_called', 't_code_v1', 't_calling_v1', 't_called_v1', 't_number_of_days', 't_number_of_versions', 't_code_versions_all'
        #             ])
        
        trees['number_of_days'] = trees['number_of_days'].apply(lambda x: list(x))
        trees['number_of_versions'] = trees['number_of_versions'].apply(lambda x: list(x))
        
        self.tree_ds = trees

    def normalise_minmax(self, df, min_val_days, max_val_days, min_val_vers, max_val_vers) :
        # Normalize both columns using Min-Max scaling
        df['number_of_days_x'] = df['number_of_days_x'].apply(lambda x:np.array([(x[0]-min_val_days)/(max_val_days-min_val_days)]))
        df['number_of_days_y'] = df['number_of_days_y'].apply(lambda x:np.array([(x[0]-min_val_days)/(max_val_days-min_val_days)]))
        df['number_of_versions_x']  = df['number_of_versions_x'].apply(lambda x:np.array([(x[0]-min_val_vers)/(max_val_vers-min_val_vers)]))
        df['number_of_versions_y']  = df['number_of_versions_y'].apply(lambda x:np.array([(x[0]-min_val_vers)/(max_val_vers-min_val_vers)]))
        return df
                                  

    def merge(self, part):
        if part == 'train':
            pairs = self.train_pairs
        elif part == 'dev':
            pairs = self.dev_pairs
        else:
            pairs = self.test_pairs
        pairs['id1'] = pairs['id1'].astype(int)
        pairs['id2'] = pairs['id2'].astype(int)
        df = pd.merge(pairs, self.tree_ds, how='left', left_on='id1', right_on='id')
        df = pd.merge(df, self.tree_ds, how='left', left_on='id2', right_on='id')
        df.drop(['id_x', 'id_y'], axis=1, inplace=True)
        df.dropna(inplace=True)

        if part == 'train': # using train set only to avoid data leakage            
            self.min_val_days = min(df['number_of_days_x'].min(), df['number_of_days_y'].min())[0]
            self.max_val_days = max(df['number_of_days_x'].max(), df['number_of_days_y'].max())[0]
            self.min_val_vers = min(df['number_of_versions_x'].min(), df['number_of_versions_y'].min())[0]
            self.max_val_vers = max(df['number_of_versions_x'].max(), df['number_of_versions_y'].max())[0]
            
        df_norm = self.normalise_minmax(df, self.min_val_days, self.max_val_days, self.min_val_vers, self.max_val_vers)
        
        df_norm.to_pickle(self.output_dir + f'/{part}_blocks.pkl')
        print(part, df_norm.shape)
        
    def run(self):
        self.extract_code_tree()        
        self.extract_pair()
        self.split_data()        
        self.dictionary_and_embedding()
        self.generate_block_seqs()
        self.merge('train')
        self.merge('dev')
        self.merge('test')

if __name__ == '__main__':    
    DATA_PATH = './data/SeSaMe_VersionHistory_Callgraph.vFinal.json'
    TREE_FILE_PATH = './data/trees.pkl'
    OUTPUT_DIR = './data/clone_detection'
    RATIO = '8:1:1'
    RANDOM_SEED = 2023
    EMBEDDING_SIZE = 128
    ppl = Pipeline(DATA_PATH, TREE_FILE_PATH, OUTPUT_DIR, RATIO, RANDOM_SEED, EMBEDDING_SIZE, tree_exists=False)
    ppl.run()