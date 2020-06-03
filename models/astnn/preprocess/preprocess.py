import pandas as pd
from gensim.models.word2vec import Word2Vec
import os


class PreprocessPipeline:
    def __init__(self, config):
        self.dest = config.DATA_PATH
        self.holdout_files = [os.path.join(self.dest, file) for file in ['train_.pkl', 'dev_.pkl', 'test_.pkl']]
        self.data = config.RAW_DATA_PATH
        self.embed_path = config.EMBEDDING_PATH
        self.config = config
        self.sources = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.word2vec = None

    # parse source code
    def parse_source(self, output_file, option='existing'):
        path = os.path.join(self.dest, output_file)
        if option is 'existing' and os.path.exists(path):
            self.sources = pd.read_pickle(path)
            print("Source code is already parsed, nothing to do.")
            return

        from pycparser import c_parser

        parser = c_parser.CParser()
        source = pd.read_pickle(self.data)
        source.columns = ['id', 'code', 'label']
        source['code'] = source['code'].apply(parser.parse)
        print(len(source))

        source.to_pickle(path)
        self.sources = source

    # split data for training, developing and testing
    def split_data(self, option='existing'):
        if option is 'existing':
            flag = True
            for file in self.holdout_files:
                if not os.path.exists(file):
                    flag = False
            if flag:
                self.train_data = pd.read_pickle(self.holdout_files[0])
                self.val_data = pd.read_pickle(self.holdout_files[1])
                self.test_data = pd.read_pickle(self.holdout_files[2])
                print("Source code is already splitted, nothing to do.")
                return

        data = self.sources
        data = data.sample(frac=1, random_state=666)  # shuffle
        data_num = len(data)
        ratios = self.config.SPLIT_RATIO
        train_split = int(ratios[0]/sum(ratios)*data_num)
        val_split = train_split + int(ratios[1]/sum(ratios)*data_num)
        train = data.iloc[:train_split] 
        dev = data.iloc[train_split:val_split] 
        test = data.iloc[val_split:] 

        self.train_data = train
        train.to_pickle(self.holdout_files[0])
        dev.to_pickle(self.holdout_files[1])
        test.to_pickle(self.holdout_files[2])

    # construct dictionary and train word embedding
    def dictionary_and_embedding(self, input_file, option='existing'):
        if option is 'existing' and os.path.exists(self.embed_path):
            self.word2vec = Word2Vec.load(self.embed_path).wv
            print("Embedding already exists, nothing to do.")
            return

        if input_file:
            trees = pd.read_pickle(input_file)
        else:
            trees = self.train_data
        if not os.path.exists(os.path.join(self.dest, 'embedding')):
            os.mkdir(os.path.join(self.dest, 'embedding'))

        from models.astnn.preprocess.sampling import get_sequences

        def trans_to_sequences(ast):
            sequence = []
            get_sequences(ast, sequence)
            return sequence
        corpus = trees['code'].apply(trans_to_sequences)
        # str_corpus = [' '.join(c) for c in corpus]
        # trees['code'] = pd.Series(str_corpus)
        # trees.to_csv(os.path.join(self.dest, 'train_programs_ns.tsv'))

        w2v = Word2Vec(corpus, size=self.config.EMBEDDING_DIM,
                       workers=16, sg=1, min_count=self.config.MIN_COUNT)
        self.word2vec = w2v
        w2v.save(self.embed_path)

    # generate block sequences with index representations
    def generate_block_seqs(self, trees, output_file, option='existing'):
        if option is 'exiting' and os.path.exists(output_file):
            print("Sequence blocks already exist, nothing to do.")
            return

        from models.astnn.preprocess.sampling import get_blocks
        vocab = self.word2vec.vocab
        max_token = self.word2vec.syn0.shape[0]

        def tree_to_index(node):
            token = node.token
            result = [vocab[token].index if token in vocab else max_token]
            children = node.children
            for child in children:
                result.append(tree_to_index(child))
            return result

        def trans2seq(r):
            blocks = []
            get_blocks(r, blocks)
            tree = []
            for b in blocks:
                btree = tree_to_index(b)
                tree.append(btree)
            return tree

        trees['code'] = trees['code'].apply(trans2seq)
        trees.to_pickle(os.path.join(self.dest, output_file))

    # run for processing data to train
    def run(self):
        print('parse source code...')
        self.parse_source('ast.pkl')
        print('split data...')
        self.split_data()
        print('train word embedding...')
        self.dictionary_and_embedding(None)
        print('generate block sequences...')
        self.generate_block_seqs(self.train_data, 'train_blocks.pkl')
        self.generate_block_seqs(self.val_data, 'val_blocks.pkl')
        self.generate_block_seqs(self.test_data, 'test__blocks.pkl')
        print("preprocessing finished!")

