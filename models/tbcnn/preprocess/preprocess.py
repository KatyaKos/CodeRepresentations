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
        self.sources = None
        self.nodes = None

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

    def parse_nodes(self, output_file, option='existing'):
        path = os.path.join(self.dest, output_file)
        if option is 'existing' and os.path.exists(path):
            self.nodes = pd.read_pickle(path)
            print("Nodes are already sampled, nothing to do.")
            return

        node_counts = defaultdict(int)
        samples = []
        has_capacity = lambda x: self.config.MAX_PER_NODE < 0 or node_counts[x] < self.config.MAX_PER_NODE
        can_add_more = lambda: self.config.MAX_NODES < 0 or len(samples) < self.config.MAX_NODES

        from models.tbcnn.preprocess.node_parser import parse_nodes

        for item in self.sources:
            root = item['code']
            new_samples = parse_nodes(root)

            for sample in new_samples:
                if has_capacity(sample[0]):
                    samples.append(sample)
                    node_counts[sample[0]] += 1
                if not can_add_more:
                    break
            if not can_add_more:
                break

        df = pd.DataFrame(samples, columns=['node', 'parent', 'children'], dtype=object)
        self.nodes = df
        df.to_pickle(path)

        print('Total nodes sampled: %d', len(node_counts))

    def vectorize(self, log_file, option='existing'):
        if option is 'existing' and os.path.exists(self.embed_path):
            print("Nodes are already vectorized, nothing to do.")
            return

        from models.tbcnn.preprocess.vectorizer import learn_vectors
        embedding, node_map = learn_vectors(self.nodes, self.embed_path, self.logdir, log_file,
                                         self.config.NUM_FEATURES, self.config.BATCH_SIZE, self.config.HIDDEN_SIZE,
                                         self.config.LEARN_RATE, self.config.EPOCHS, self.config.CHECKPOINT_STEP)
        #df = pd.DataFrame(embedding, columns=nodes)
        #df.to_pickle(output_path)
        with open(self.embed_path, 'wb') as fout:
            pickle.dump((embedding, node_map), fout)

    def parse_trees(self, output_file, option='existing'):
        output_path = os.path.join(self.dest, output_file)
        if option is 'existing' and os.path.exists(output_path):
            print("Trees are already sampled, nothing to do.")
            return

        sys.setrecursionlimit(1000000)
        train, val, test = [], [], []
        labels = set()
        data = self.sources
        data = data.sample(frac=1, random_state=666) # shuffle
        ratios = self.config.SPLIT_RATIO
        data_num = len(data)
        train_split = int(ratios[0] / sum(ratios) * data_num)
        val_split = train_split + int(ratios[1] / sum(ratios) * data_num)

        for df_old, df_new in [(data.iloc[:train_split], train),
                               (data.iloc[train_split:val_split], val),
                               (data.iloc[val_split:], test)]:
            for item in df_old:
                from models.tbcnn.preprocess.tree_parser import parse_tree
                root = item['code']
                label = item['label']
                sample, num_nodes = parse_tree(root)
                if num_nodes > self.config.MAX_TREE_SIZE or num_nodes < self.config.MIN_TREE_SIZE:
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

        print('Total train trees sampled: %d', train_counts)
        print('Total val trees sampled: %d', val_counts)
        print('Total test trees sampled: %d', test_counts)

    # run for processing data to train
    def run(self):
        print('parse source code...')
        self.parse_source(self.config.AST_FILE)
        print('sample nodes...')
        self.parse_nodes(self.config.SAMPLED_NODES_FILE)
        print('vectorize sampled nodes...')
        self.vectorize('embedding.pkl', 'ast2vec.ckpt')
        print('sample trees...')
        self.parse_trees(self.config.SAMPLED_TREES_FILE)
        print("preprocessing finished!")
