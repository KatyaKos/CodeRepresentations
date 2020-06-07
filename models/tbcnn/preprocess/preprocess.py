import pandas as pd
from collections import defaultdict
import os
import sys
import pickle


class PreprocessPipeline:

    def __init__(self, config):
        self.dest = config.DATA_PATH
        self.data = config.RAW_DATA_PATH
        self.embed_path = config.EMBEDDING_PATH
        self.logdir = config.LOGDIR
        self.config = config

        self.train_sources = None
        self.val_sources = None
        self.test_sources = None
        self.train_nodes = None
        self.node_map = None
        self.UNK_NODE = '<UNKNOWN_NODE>'

    # parse source code
    def parse_source(self, output_file, option='existing'):
        path = os.path.join(self.dest, output_file)
        if option is 'existing' and os.path.exists(path):
            with open(path, 'rb') as fin:
                self.train_sources, self.val_sources, self.test_sources = pickle.load(fin)
            print("Source code is already parsed, nothing to do.")
            return
        from pycparser import c_parser

        parser = c_parser.CParser()
        source = pd.read_pickle(self.data)
        source.columns = ['id', 'code', 'label']
        source['code'] = source['code'].apply(parser.parse)
        data_num = len(source)
        print("Number of codes", data_num)
        
        source = source.sample(frac=1, random_state=666) # shuffle
        ratios = self.config.SPLIT_RATIO
        train_split = int(ratios[0] / sum(ratios) * data_num)
        val_split = train_split + int(ratios[1] / sum(ratios) * data_num)
        
        self.train_sources = source.iloc[:train_split]
        self.val_sources = source.iloc[train_split:val_split]
        self.test_sources = source.iloc[val_split:]
        with open(path, 'wb') as fout:
            pickle.dump((self.train_sources, self.val_sources, self.test_sources), fout)

    def parse_nodes(self, output_file, option='existing'):
        path = os.path.join(self.dest, output_file)
        if option is 'existing' and os.path.exists(path):
            self.train_nodes = pd.read_pickle(path)
            nodes = list(set(self.train_nodes['node'].tolist()))
            self.node_map = {x: i for i, x in enumerate(nodes)}
            print("Nodes are already sampled, nothing to do.")
            return

        node_counts = defaultdict(int)
        samples = []
        has_capacity = lambda x: self.config.MAX_PER_NODE < 0 or node_counts[x] < self.config.MAX_PER_NODE
        can_add_more = lambda: self.config.MAX_NODES < 0 or len(samples) < self.config.MAX_NODES

        from models.tbcnn.preprocess.node_parser import parse_nodes

        for data in [self.train_sources]:
            for root in data['code']:
                new_samples = parse_nodes(root)

                for sample in new_samples:
                    if has_capacity(sample[0]):
                        samples.append(sample)
                        node_counts[sample[0]] += 1
                    if not can_add_more:
                        break
                if not can_add_more:
                    break

        def create_node_map(counts):
            nodes = [self.UNK_NODE]
            for node in counts:
                if counts[node] >= self.config.MIN_PER_NODE:
                    nodes.append(node)
            return {x: i for i, x in enumerate(nodes)}

        self.node_map = create_node_map(node_counts)
        
        def node_replacer(sample):
            if not (sample[0] in self.node_map):
                sample[0] = self.UNK_NODE
            if not (sample[1] in self.node_map):
                sample[1] = self.UNK_NODE
            for i in range(len(sample[2])):
                if not (sample[2][i] in self.node_map):
                    sample[2][i] = self.UNK_NODE
            return sample 

        samples = list(map(node_replacer, samples))
        #samples = list(filter(lambda s: node_counts[s[0]] > self.config.MIN_PER_NODE, samples))

        df = pd.DataFrame(samples, columns=['node', 'parent', 'children'], dtype=object)
        self.train_nodes = df
        df.to_pickle(path)

        print('Total number of nodes sampled:', len(self.node_map))
        print('Total number of samples:', len(self.train_nodes))

    def vectorize(self, log_file, option='existing'):
        if option is 'existing' and os.path.exists(self.embed_path):
            with open(self.embed_path, 'rb') as fin:
                _, self.node_map = pickle.load(fin)
            print("Nodes are already vectorized, nothing to do.")
            return

        #from models.tbcnn.preprocess.vectorizer import learn_vectors
        #embedding = learn_vectors(self.train_nodes, self.node_map, self.embed_path, self.logdir, log_file,
        #                                 self.config.NUM_FEATURES, self.config.BATCH_SIZE, self.config.HIDDEN_SIZE,
        #                                 self.config.LEARN_RATE, self.config.EPOCHS, self.config.CHECKPOINT_STEP)
        
        from gensim.models.word2vec import Word2Vec
        import numpy as np
        word2vec = Word2Vec.load('data/astnn/embedding/node_embededing_128.wv').wv
        embedding = np.zeros((word2vec.syn0.shape[0] + 1, word2vec.syn0.shape[1]), dtype="float32")
        embedding[:word2vec.syn0.shape[0]] = word2vec.syn0
        vocab = word2vec.vocab
        self.node_map = {t: vocab[t].index for t in vocab}
        self.node_map[self.UNK_NODE] = word2vec.syn0.shape[0]

        #df = pd.DataFrame(embedding, columns=nodes)
        #df.to_pickle(output_path)
        with open(self.embed_path, 'wb') as fout:
            pickle.dump((embedding, self.node_map), fout)

    def parse_trees(self, output_file, option='existing'):
        output_path = os.path.join(self.dest, output_file)
        if option is 'existing' and os.path.exists(output_path):
            print("Trees are already sampled, nothing to do.")
            return

        sys.setrecursionlimit(1000000)
        train, val, test = [], [], []
        labels = set()

        for df_old, df_new in [(self.train_sources, train),
                               (self.val_sources, val),
                               (self.test_sources, test)]:
            for root, label in zip(df_old['code'], df_old['label']):
                from models.tbcnn.preprocess.tree_parser import parse_tree
                sample, num_nodes, depth = parse_tree(root, self.node_map, self.UNK_NODE)
                if num_nodes > self.config.MAX_TREE_SIZE or num_nodes < self.config.MIN_TREE_SIZE or depth > self.config.MAX_DEPTH:
                    continue
                datum = {'tree': sample, 'label': label}
                labels.add(label)
                df_new.append(datum)
                #df_new.append([item['id'], sample, label])

        train_counts, val_counts, test_counts = len(train), len(val), len(test)
        #train = pd.DataFrame(train, columns=['id', 'tree', 'label'])
        #val = pd.DataFrame(val, columns=['id', 'tree', 'label'])
        #test = pd.DataFrame(test, columns=['id', 'tree', 'label'])
        labels = list(labels)
        with open(output_path, 'wb') as fout:
            pickle.dump((train, test, labels), fout)

        print('Total train trees sampled:', train_counts)
        print('Total val trees sampled:', val_counts)
        print('Total test trees sampled:', test_counts)

    # run for processing data to train
    def run(self):
        print('parse source code...')
        self.parse_source(self.config.AST_FILE)
        print('sample nodes...')
        self.parse_nodes(self.config.SAMPLED_NODES_FILE)
        print('vectorize sampled nodes...')
        self.vectorize('ast2vec.ckpt')
        print('sample trees...')
        self.parse_trees(self.config.SAMPLED_TREES_FILE)
        print("preprocessing finished!")
