import javalang
import json
import numpy as np
import os
import pandas as pd

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
        self.labels = None
        self.train_trees = None
        self.dev_trees = None
        self.test_trees = None

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
                print(f"Warning: No. {code_object['dbid']} target cannot be parsed!")
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
                        print(f'Warning: The version history context {item["commit_version_no"]} | {code_object["dbid"]} {code_object["file"]} | {code_object["method"]} cannot be parsed!')                        
                i+=1

            # callgraph context
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
                    print(f'Warning: The callgraph context {method} cannot be parsed!')
            
            flg_obj = javalang.tree.MethodDeclaration()
            if calling_trees == []:
                calling_trees.append(flg_obj)
            if called_trees == []:
                called_trees.append(flg_obj)


            ### code and callgraph from 2022's approach
            try:
                original_tree_v1 = parse_program(code_object['callgraph_code_v1'])                
            except Exception:
                print(f"Warning: No. {code_object['dbid']} target cannot be parsed!")
                # return code_object['dbid'], None, None, None, None, None, None, None, None, None, None
            
            # callgraph context v1
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
                    print(f'Warning: The callgraph context V1 {method} cannot be parsed!')
            flg_obj = javalang.tree.MethodDeclaration()
            if calling_trees_v1 == []:
                calling_trees_v1.append(flg_obj)
            if called_trees_v1 == []:
                called_trees_v1.append(flg_obj)


            # number of existing days
            number_of_days_tree = np.array([code_object['days_to_exist']])

            # number of versions
            number_of_versions_tree = np.array([code_object['number_of_versions']])

            # all versions to list
            # code_versions_all_trees = []
            # i = 0
            # for item in code_object['version_history_context']:
            #     if i >= 1: # skip version 0 as current version
            #         try:
            #             temp_tree = parse_program(item['commit_source_code'])
            #             code_versions_all_trees.append(temp_tree)
            #         except Exception:
            #             print(f'Warning::all versions to list:: The version history context {item["commit_version_no"]} | {code_object["dbid"]} {code_object["file"]} | {code_object["method"]} cannot be parsed!')
            #     i+=1

            
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
            t_code_v1 = code_object['callgraph_code_v1']
            if t_calling == []:
                t_calling.append('')
            if t_called == []:
                t_called.append('')

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

            # return code_object['dbid'], original_tree, calling_trees, called_trees            
            # return code_object['dbid'], original_tree, code_versions_trees, calling_trees, called_trees, original_tree_v1, calling_trees_v1, called_trees_v1, number_of_days_tree, number_of_versions_tree, code_versions_all_trees
            return code_object['dbid'], original_tree, code_versions_trees, calling_trees, called_trees, \
                                        original_tree_v1, calling_trees_v1, called_trees_v1, \
                                        number_of_days_tree, number_of_versions_tree, code_versions_all_trees, \
                                        t_code, t_code_versions, t_calling, t_called, t_code_v1, t_calling_v1, t_called_v1, \
                                        t_number_of_days, t_number_of_versions, t_code_versions_all

        tree_array = []
        record = []

        # for sample in self.dataset:
        #     for number in ['first', 'second']:
        #         code_ob = sample[number]
        #         if code_ob['dbid'] in record:
        #             continue
        #         dbid, original_tree, calling_trees, called_trees = process_context_code(code_ob)
        #         tree_array.append([int(dbid), original_tree, calling_trees, called_trees])
        #         record.append(dbid)

        excluded_list = []

        for sample in self.dataset:
            for number in ['first', 'second']:
                code_ob = sample[number]
                if code_ob['dbid'] in record:                    
                    continue
                try:
                    # dbid, original_tree, code_versions_trees, calling_trees, called_trees = process_context_code(code_ob)
                    # dbid, original_tree, code_versions_trees, calling_trees, called_trees, original_tree_v1, calling_trees_v1, called_trees_v1 = process_context_code(code_ob)
                    # dbid, original_tree, code_versions_trees, calling_trees, called_trees, original_tree_v1, calling_trees_v1, called_trees_v1, number_of_days_tree, number_of_versions_tree, code_versions_all_trees = process_context_code(code_ob)
                    dbid, original_tree, code_versions_trees, calling_trees, called_trees, original_tree_v1, calling_trees_v1, called_trees_v1, number_of_days_tree, number_of_versions_tree, code_versions_all_trees, \
                        t_code, t_code_versions, t_calling, t_called, t_code_v1, t_calling_v1, t_called_v1, t_number_of_days, t_number_of_versions, t_code_versions_all = process_context_code(code_ob)                    

                    # tree_array.append([int(dbid), original_tree, code_versions_trees, calling_trees, called_trees])
                    # tree_array.append([int(dbid), original_tree, code_versions_trees, calling_trees, called_trees, original_tree_v1, calling_trees_v1, called_trees_v1])
                    # tree_array.append([int(dbid), original_tree, code_versions_trees, calling_trees, called_trees, original_tree_v1, calling_trees_v1, called_trees_v1, number_of_days_tree, number_of_versions_tree, code_versions_all_trees])
                    tree_array.append([int(dbid), original_tree, code_versions_trees, calling_trees, called_trees, original_tree_v1, calling_trees_v1, called_trees_v1, number_of_days_tree, number_of_versions_tree, code_versions_all_trees, \
                                    t_code, t_code_versions, t_calling, t_called, t_code_v1, t_calling_v1, t_called_v1, t_number_of_days, t_number_of_versions, t_code_versions_all])

                    record.append(dbid)
                    # print(dbid, len(code_versions_trees))
                except Exception:
                    excluded_list.append([code_ob['dbid'], code_ob['project'], code_ob['file'], code_ob['method'], code_ob['uniqueid']])
                    # print(code_ob['dbid'], code_ob['file'], code_ob['method'])
                    print(f"Warning::=====>:: No. {code_ob['dbid']} target cannot be parsed!")
                    print(f"------------------------------------------------------------------")

        # new_df = pd.DataFrame(tree_array, columns=['id', 'code', 'calling', 'called'])
        # new_df = new_df.loc[pd.notnull(new_df['code']), ['id', 'code', 'calling', 'called']]

        # new_df = pd.DataFrame(tree_array, columns=['id', 'code', 'code_versions', 'calling', 'called'])
        # new_df = pd.DataFrame(tree_array, columns=['id', 'code', 'code_versions', 'calling', 'called', 'code_v1', 'calling_v1', 'called_v1'])
        # new_df = pd.DataFrame(tree_array, columns=['id', 'code', 'code_versions', 'calling', 'called', 'code_v1', 'calling_v1', 'called_v1', 'number_of_days', 'number_of_versions', 'code_versions_all' ])
        new_df = pd.DataFrame(tree_array, columns=['id', 'code', 'code_versions', 'calling', 'called', 'code_v1', 'calling_v1', 'called_v1', 'number_of_days', 'number_of_versions', 'code_versions_all',  
                                                   't_code', 't_code_versions', 't_calling', 't_called', 't_code_v1', 't_calling_v1', 't_called_v1', 't_number_of_days', 't_number_of_versions', 't_code_versions_all'])

        # new_df = new_df.loc[pd.notnull(new_df['code']), ['id', 'code', 'code_versions', 'calling', 'called']]
        # new_df = new_df.loc[pd.notnull(new_df['code']), ['id', 'code', 'code_versions', 'calling', 'called', 'code_v1', 'calling_v1', 'called_v1']]
        # new_df = new_df.loc[pd.notnull(new_df['code']), ['id', 'code', 'code_versions', 'calling', 'called', 'code_v1', 'calling_v1', 'called_v1', 'number_of_days', 'number_of_versions', 'code_versions_all']]
        new_df = new_df.loc[pd.notnull(new_df['code']), ['id', 'code', 'code_versions', 'calling', 'called', 'code_v1', 'calling_v1', 'called_v1', 'number_of_days', 'number_of_versions', 'code_versions_all', 
                                                         't_code', 't_code_versions', 't_calling', 't_called', 't_code_v1', 't_calling_v1', 't_called_v1', 't_number_of_days', 't_number_of_versions', 't_code_versions_all']]

        self.tree_ds = new_df

        if not os.path.exists(os.path.dirname(self.tree_file_path)):
            os.mkdir(os.path.dirname(self.tree_file_path))
        self.tree_ds.to_pickle(self.tree_file_path)

        print("Excluded list:", excluded_list)

        return self.tree_ds

    def split_data(self):
        data = self.tree_ds
        data_num = len(data)
        ratios = [int(r) for r in self.ratio.split(':')]
        train_split = int(ratios[0]/sum(ratios)*data_num)
        val_split = train_split + int(ratios[1]/sum(ratios)*data_num)

        data = data.sample(frac=1, random_state=self.seed)
        self.train_trees = data.iloc[:train_split]
        self.dev_trees = data.iloc[train_split:val_split]
        self.test_trees = data.iloc[val_split:]

    def dictionary_and_embedding(self):
        trees = self.train_trees
        train_ids = trees['id'].unique()

        trees = self.tree_ds.set_index('id', drop=False).loc[train_ids]
        from utils import get_sequence as func

        def trans_to_sequences(ast):
            sequence = []
            func(ast, sequence)
            return sequence

        trees_array = trees.values
        corpus = []
        for i, tree_sample in enumerate(trees_array):

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
            
            # ins_seq = trans_to_sequences(tree_sample[10]) # code_versions_all as a string
            # corpus.append(ins_seq)

            for code_versions_all_tree in tree_sample[10]: # code_versions_all as a list
                ins_seq = trans_to_sequences(code_versions_all_tree)
                corpus.append(ins_seq)

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
        # max_token = word2vec.syn0.shape[0]
        max_token = word2vec.vectors.shape[0]

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

        trees['code'] = trees['code'].apply(trans2seq)
                
        trees['code_versions'] = trees['code_versions'].apply(trans2seqs)
        trees['calling'] = trees['calling'].apply(trans2seqs)
        trees['called'] = trees['called'].apply(trans2seqs)

        # callgraph V1 from 2022's approach
        trees['code_v1'] = trees['code_v1'].apply(trans2seq)

        trees['calling_v1'] = trees['calling_v1'].apply(trans2seqs)
        trees['called_v1'] = trees['called_v1'].apply(trans2seqs)

        # all versions of code
        # trees['code_versions_all'] = trees['code_versions_all'].apply(trans2seq) # as a string
        trees['code_versions_all'] = trees['code_versions_all'].apply(trans2seqs)  # as a list

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
            block_sample[3] = max_tree
            block_sample[13] = block_sample[13][flg] 

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
            block_sample[4] = max_tree
            block_sample[14] = block_sample[14][flg] 

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
            block_sample[6] = max_tree
            block_sample[16] = block_sample[16][flg] 

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
            block_sample[7] = max_tree
            block_sample[17] = block_sample[17][flg]

            # code_versions_all            
            merged_trees = []            
            for code_versions_tree in block_sample[10]:
                merged_trees.append(code_versions_tree)
                         
            block_sample[10] = merged_trees


            # trees_list.append(list(block_sample))

        # with open('./data/sesame_tokens_with_context.json', 'w', encoding='utf-8') as out_file:
        #     json.dump(trees_list, out_file, indent=4)

        # trees = pd.DataFrame(trees_array, columns=['id', 'code', 'calling', 'called'])
        # trees = pd.DataFrame(trees_array, columns=['id', 'code', 'code_versions'])
        # trees = pd.DataFrame(trees_array, columns=['id', 'code', 'code_versions', 'calling', 'called'])        
        # trees = pd.DataFrame(trees_array, columns=['id', 'code', 'code_versions', 'calling', 'called', 'code_v1', 'calling_v1', 'called_v1'])
        # trees = pd.DataFrame(trees_array, columns=['id', 'code', 'code_versions', 'calling', 'called', 'code_v1', 'calling_v1', 'called_v1', 'number_of_days', 'number_of_versions', 'code_versions_all'])
        trees = pd.DataFrame(trees_array, columns=['id', 'code', 'code_versions', 'calling', 'called', 'code_v1', 'calling_v1', 'called_v1', 'number_of_days', 'number_of_versions', 'code_versions_all',
                                                   't_code', 't_code_versions', 't_calling', 't_called', 't_code_v1', 't_calling_v1', 't_called_v1', 't_number_of_days', 't_number_of_versions', 't_code_versions_all'])

        trees['number_of_days'] = trees['number_of_days'].apply(lambda x: list(x))
        trees['number_of_versions'] = trees['number_of_versions'].apply(lambda x: list(x))

        self.tree_ds = trees
    
    def normalise_minmax(self, df, min_val_days, max_val_days, min_val_vers, max_val_vers) :
        # Normalize both columns using Min-Max scaling
        df['number_of_days'] = df['number_of_days'].apply(lambda x:np.array([(x[0]-min_val_days)/(max_val_days-min_val_days)]))        
        df['number_of_versions']  = df['number_of_versions'].apply(lambda x:np.array([(x[0]-min_val_vers)/(max_val_vers-min_val_vers)]))        
        return df

    def generate_class_ds(self):
        class_data = []
        for pair in self.dataset:
            sample = [int(pair['first']['dbid']), pair['first']['project']]
            class_data.append(sample)
            sample = [int(pair['second']['dbid']), pair['second']['project']]
            class_data.append(sample)
        class_data = sorted(class_data, key=lambda x: x[0])
        class_data_df = pd.DataFrame(class_data, columns=['id', 'label'])
        class_data_df.drop_duplicates(subset=['id'], keep='first', inplace=True)
        # You can uncomment the following statement to save the class label file.
        # class_data_df.to_csv('./data/classification/class_label.csv', index=False)

        classes = np.unique(class_data_df['label'].values)
        classes = sorted(classes)
        classes_map = {name: i for i, name in enumerate(classes)}
        class_data_df['label'] = class_data_df['label'].apply(lambda x: classes_map[x])
        self.labels = class_data_df

        train_df = pd.merge(self.train_trees[['id']], self.tree_ds, how='left', left_on='id', right_on='id')
        dev_df = pd.merge(self.dev_trees[['id']], self.tree_ds, how='left', left_on='id', right_on='id')
        test_df = pd.merge(self.test_trees[['id']], self.tree_ds, how='left', left_on='id', right_on='id')

        train_df = pd.merge(train_df, class_data_df, how='left', left_on='id', right_on='id')
        dev_df = pd.merge(dev_df, class_data_df, how='left', left_on='id', right_on='id')
        test_df = pd.merge(test_df, class_data_df, how='left', left_on='id', right_on='id')

        # using train set only to avoid data leakage            
        self.min_val_days = train_df['number_of_days'].min()[0]
        self.max_val_days = train_df['number_of_days'].max()[0]
        self.min_val_vers = train_df['number_of_versions'].min()[0]
        self.max_val_vers = train_df['number_of_versions'].max()[0]

        train_df = self.normalise_minmax(train_df, self.min_val_days, self.max_val_days, self.min_val_vers, self.max_val_vers)
        dev_df = self.normalise_minmax(dev_df, self.min_val_days, self.max_val_days, self.min_val_vers, self.max_val_vers)
        test_df = self.normalise_minmax(test_df, self.min_val_days, self.max_val_days, self.min_val_vers, self.max_val_vers)
        
        train_df.to_pickle(self.output_dir + '/train_df.pkl')
        dev_df.to_pickle(self.output_dir + '/dev_df.pkl')
        test_df.to_pickle(self.output_dir + '/test_df.pkl')
        return True

    def generate_random_class_ds(self):
        final_df = pd.merge(self.tree_ds, self.labels, how='left', left_on='id', right_on='id')

        for i in range(11): # 10 classes correspond to 10 projects
            
            code_versions_series = final_df.loc[final_df['label'] == i, 'code_versions']
            calling_series = final_df.loc[final_df['label'] == i, 'calling']
            called_series = final_df.loc[final_df['label'] == i, 'called']
            
            code_versions_random_series = code_versions_series.sample(frac=1, random_state=self.seed)
            calling_random_series = calling_series.sample(frac=1, random_state=self.seed)
            called_random_series = called_series.sample(frac=1, random_state=self.seed)
            
            final_df.loc[final_df['label'] == i, 'code_versions'] = code_versions_random_series.values
            final_df.loc[final_df['label'] == i, 'calling'] = calling_random_series.values
            final_df.loc[final_df['label'] == i, 'called'] = called_random_series.values

        train_df = pd.merge(self.train_trees[['id']], final_df, how='left', left_on='id', right_on='id')
        dev_df = pd.merge(self.dev_trees[['id']], final_df, how='left', left_on='id', right_on='id')
        test_df = pd.merge(self.test_trees[['id']], final_df, how='left', left_on='id', right_on='id')

        # using train set only to avoid data leakage            
        self.min_val_days = train_df['number_of_days'].min()[0]
        self.max_val_days = train_df['number_of_days'].max()[0]
        self.min_val_vers = train_df['number_of_versions'].min()[0]
        self.max_val_vers = train_df['number_of_versions'].max()[0]

        train_df = self.normalise_minmax(train_df, self.min_val_days, self.max_val_days, self.min_val_vers, self.max_val_vers)
        dev_df = self.normalise_minmax(dev_df, self.min_val_days, self.max_val_days, self.min_val_vers, self.max_val_vers)
        test_df = self.normalise_minmax(test_df, self.min_val_days, self.max_val_days, self.min_val_vers, self.max_val_vers)

        train_df.to_pickle(self.output_dir + '/train_random_df.pkl')
        dev_df.to_pickle(self.output_dir + '/dev_random_df.pkl')
        test_df.to_pickle(self.output_dir + '/test_random_df.pkl')
        return True

    def run(self):
        self.extract_code_tree()
        self.split_data()
        self.dictionary_and_embedding()
        self.generate_block_seqs()
        self.generate_class_ds()
        self.generate_random_class_ds()

if __name__ == '__main__':
    DATA_PATH = './data/SeSaMe_VersionHistory_Callgraph.vFinal.json'
    TREE_FILE_PATH = './data/trees.pkl'
    OUTPUT_DIR = './data/classification'
    RATIO = '8:1:1'
    RANDOM_SEED = 2023
    EMBEDDING_SIZE = 128
    ppl = Pipeline(DATA_PATH, TREE_FILE_PATH, OUTPUT_DIR, RATIO, RANDOM_SEED, EMBEDDING_SIZE, tree_exists=True)
    ppl.run()