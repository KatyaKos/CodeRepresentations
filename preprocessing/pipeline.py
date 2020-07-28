import os
from json import load as json_load
import pickle
from argparse import ArgumentParser

from preprocessing.data_workers import pickle_iof, splitter
from preprocessing.parsers import pycparser, tree_sitter
from preprocessing.samplers import node_sampler, tree_sampler, tree_astnn_sampler
from preprocessing.vectorizers import word2vec

READERS = [pickle_iof.PickleReader]
SPLITTERS = [splitter.Splitter]
VECTORIZERS = [word2vec.W2vVectorizer]
NODE_SAMPLERS = [node_sampler.NodeSampler]
TREE_SAMPLERS = [tree_sampler.TreeSampler, tree_astnn_sampler.TreeAstnnSampler]

PARSERS = [parser.name() for parser in [pycparser.PyCParser, tree_sitter.TreeSitter]]


def choose(name, choice):
    for item in choice:
        if item.name() == name:
            return item
    return None


def run(params):
    parser = choose(params.parser, PARSERS)()

    print('Parse and split source code...')
    reader = choose(params.components.iof, READERS)(parser)
    source = reader.read_source(params.paths.data_file)
    print("Number of codes", len(source))
    splitter = choose(params.components.splitter, SPLITTERS)()
    train_sources, val_sources, test_sources = splitter.split(source, params.split_ratio.split(":"))

    print('Vectorize nodes from train source...')
    output_path = params.paths.embedding_file
    if params.mode is 'existing' and os.path.exists(output_path):
        _, node_map = reader.read(output_path)
        print("Nodes are already vectorized, nothing to do.")
    else:
        sampler = choose(params.components.node_sampler, NODE_SAMPLERS)(parser)
        corpus = train_sources['code'].apply(sampler.sample_nodes)
        vectorizer = choose(params.components.vectorizer, VECTORIZERS)(params.embedding_dim,
                                                                       params.min_count, params.unknown_node)
        embedding, node_map = vectorizer.vectorize(corpus)
        reader.write((embedding, node_map), output_path)

    print('Sample trees...')
    output_path = params.paths.destination_file
    if params.mode is 'existing' and os.path.exists(output_path):
        print("Trees are already sampled, nothing to do.")
    else:
        sampler = choose(params.components.tree_sampler, TREE_SAMPLERS)(parser, node_map, params.max_tree_size,
                                                                        params.min_tree_size, params.max_depth,
                                                                        params.unknown_node)
        train, val, test = [], [], []
        labels = set()
        for df_old, df_new in [(train_sources, train), (val_sources, val), (test_sources, test)]:
            df_new, labels_new = sampler.sample_trees(df_old)
            labels = labels.union(labels_new)

        labels = list(labels)
        with open(output_path, 'wb') as fout:
            pickle.dump((train, val, test, labels), fout)

        print('Total train trees sampled:', len(train))
        print('Total val trees sampled:', len(val))
        print('Total test trees sampled:', len(test))
        print('Total labels:', len(labels))

    print("Preprocessing finished!")


if __name__ == '__main__':
    arg_parser = ArgumentParser(description="preprocess data")
    arg_parser.add_argument('-c', '--config', type=str, help='path to config json')
    args = arg_parser.parse_args()

    with open(args.config) as config_file:
        run(json_load(config_file))
