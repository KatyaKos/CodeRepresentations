import os
import pickle
import sys

import pandas as pd


class PreprocessPipeline:

    def __init__(self, config, parser, tree_sampler):
        self.dest = config.DATA_PATH
        self.data = config.RAW_DATA_PATH
        self.embed_path = config.EMBEDDING_PATH
        self.config = config
        self.parser = parser
        self.tree_sampler = tree_sampler

        self.train_sources = None
        self.val_sources = None
        self.test_sources = None

    # parse and split source code from [id, code, label] pickle file
    def parse_source(self, output_file, option='existing'):
        path = os.path.join(self.dest, output_file)
        if option is 'existing' and os.path.exists(path):
            with open(path, 'rb') as fin:
                self.train_sources, self.val_sources, self.test_sources = pickle.load(fin)
            print("Source code is already parsed, nothing to do.")
            return

        source = pd.read_pickle(self.data)
        source.columns = ['id', 'code', 'label']
        source['label'] = source['label'].apply(str)
        source['code'] = source['code'].apply(self.parser.parse_code)
        data_num = len(source)
        print("Number of codes", data_num)

        source = source.sample(frac=1, random_state=666)  # shuffle
        ratios = self.config.SPLIT_RATIO
        train_split = int(ratios[0] / sum(ratios) * data_num)
        val_split = train_split + int(ratios[1] / sum(ratios) * data_num)

        self.train_sources = source.iloc[:train_split]
        self.val_sources = source.iloc[train_split:val_split]
        self.test_sources = source.iloc[val_split:]
        with open(path, 'wb') as fout:
            pickle.dump((self.train_sources, self.val_sources, self.test_sources), fout)

    # construct nodes dictionary and train word embedding
    def vectorize_nodes(self, input_file=None, option='existing'):
        if option is 'existing' and os.path.exists(self.embed_path):
            with open(self.embed_path, 'rb') as fin:
                _, self.node_map = pickle.load(fin)
            print("Nodes are already vectorized, nothing to do.")
            return

        if input_file:
            trees = pd.read_pickle(input_file)
        else:
            trees = self.train_sources

        def sample_node_sequences(root, to):
            current_token = self.parser.get_token(root)
            to.append(current_token)
            for child in self.parser.get_children(root):
                sample_node_sequences(child, to)
            if current_token.lower() == 'compound':
                to.append('End')

        def trans_to_sequences(ast):
            sequence = []
            sample_node_sequences(ast, sequence)
            return sequence

        corpus = trees['code'].apply(trans_to_sequences)

        from gensim.models.word2vec import Word2Vec
        import numpy as np
        word2vec = Word2Vec(corpus, size=self.config.EMBEDDING_DIM, workers=16, sg=1, min_count=self.config.MIN_COUNT).wv
        embedding = np.zeros((word2vec.syn0.shape[0] + 1, word2vec.syn0.shape[1]), dtype="float32")
        embedding[:word2vec.syn0.shape[0]] = word2vec.syn0
        vocab = word2vec.vocab
        self.node_map = {t: vocab[t].index for t in vocab}
        self.node_map[self.config.UNK_NODE] = word2vec.syn0.shape[0]

        with open(self.embed_path, 'wb') as fout:
            pickle.dump((embedding, self.node_map), fout)

    # generate block sequences with index representations
    def parse_trees(self, output_file, option='existing'):
        output_path = os.path.join(self.dest, output_file)
        if option is 'existing' and os.path.exists(output_path):
            print("Trees are already sampled, nothing to do.")
            return

        labels = set()
        train, val, test = [], [], []

        for df_old, df_new in [(self.train_sources, train),
                               (self.val_sources, val),
                               (self.test_sources, test)]:
            for id, root, label in zip(df_old['id'], df_old['code'], df_old['label']):
                sample, num_nodes, depth = self.tree_sampler.sample(root, self.node_map, self.config.UNK_NODE)
                if num_nodes > self.config.MAX_TREE_SIZE or num_nodes < self.config.MIN_TREE_SIZE or depth > self.config.MAX_DEPTH:
                    continue
                datum = [id, sample, label]
                labels.add(label)
                df_new.append(datum)

            df_new = pd.DataFrame(df_new, columns=['id', 'code', 'label'])

        labels = list(labels)
        with open(output_path, 'wb') as fout:
            pickle.dump((train, val, test, labels), fout)

        print('Total train trees sampled:', len(train))
        print('Total val trees sampled:', len(val))
        print('Total test trees sampled:', len(test))
        print('Total labels:', len(labels))

    # run for processing data to train
    def run(self):
        print('Parse and split source code...')
        self.parse_source(self.config.AST_FILE)
        print('Vectorize nodes from train source...')
        self.vectorize_nodes()
        print('Sample trees...')
        self.parse_trees(self.config.SAMPLED_TREES_FILE)
        print("Preprocessing finished!")