import os
from json import load as json_load
from typing import Dict

import pandas as pd
from argparse import ArgumentParser

from preprocessing.data_workers.data_worker import DataWorker
from preprocessing.ast_builders.parser import Parser
from preprocessing.samplers import node_sampler, tree_sampler
from preprocessing.vectorizers import vectorizer as vector_tool


def run(params: Dict, mode: str = 'all'):
    parser = Parser().create(params['parser'])

    print('Preprocessing train source code...')
    print('Parsing train source code...')
    worker_name = params['components']['data_worker']
    train_sources = _create_source(worker_name, parser, params['paths']['train_data'])
    print("Number of codes parsed:", len(train_sources))
    val_sources = _create_source(worker_name, parser, params['paths']['val_data'])
    print("Number of val codes parsed:", len(val_sources))
    test_sources = _create_source(worker_name, parser, params['paths']['test_data'])
    print("Number of test codes parsed:", len(test_sources))

    embedding_path = os.path.join(params['paths']['embedding_dir'], params['filenames']['embedding_file'] + '.pkl')
    if mode == 'all' or mode == 'vectorize':
        print('Vectorize nodes from train source...')
        sampler = node_sampler.NodeSampler(parser).create(params['components']['node_sampler'])
        corpus = list(train_sources.get_data('code').values())
        corpus = [sampler.sample_node_sequences(ast) for ast in corpus]
        vectorizer = vector_tool.Vectorizer(params['embedding_dim'], params['min_count'], params['unknown_node'])\
            .create(params['components']['vectorizer'])
        embedding = vectorizer.vectorize(corpus)
        embedding.to_pickle(embedding_path)

    if mode == 'all' or mode == 'sample':
        if os.path.exists(embedding_path):
            embedding = pd.read_pickle(embedding_path)
            node_map = {node: i for node, i in zip(embedding['node'], embedding['id'])}
        else:
            print('No vectorization found, creating node map from the source...')
            node_map = []
            #TODO

        print('Sample trees...')

        sampler = tree_sampler.TreeSampler(parser, node_map, params['batch_size'], params['max_tree_size'], params['min_tree_size'],
                                            params['max_depth'], params['unknown_node'])\
            .create(params['components']['tree_sampler'])
        output_path = params['paths']['destination_dir']
        for source, name in [(train_sources, params['filenames']['train_file']),
                             (val_sources, params['filenames']['val_file']), (test_sources, params['filenames']['test_file'])]:
            sampler.sample_trees(source)
            source.write(output_path, name)
            print('Total trees sampled for ' + name, len(source))

        train_labels = list(set(list(train_sources.get_data('label').values())))
        train_labels = [[i, l] for i, l in enumerate(train_labels)]
        print('Total labels:', len(train_labels))
        train_labels = pd.DataFrame(train_labels, columns=['id', 'label'])
        labels_path = os.path.join(output_path, params['filenames']['labels_file'] + '.pkl')
        train_labels.to_pickle(labels_path)

    print("Preprocessing finished!")


def _create_source(worker_name: str, parser: Parser, path: str) -> DataWorker:
    sources = DataWorker().create(worker_name)
    sources.parse(parser, path)
    return sources

if __name__ == '__main__':
    arg_parser = ArgumentParser(description="preprocess data")
    arg_parser.add_argument('-c', '--config', dest='config', type=str, help='path to config json')
    # 'all' --- all the step, 'emb' --- only vectorization, 'samp' --- only sampling
    # TODO help
    arg_parser.add_argument('-m', '--mode', dest='mode', type=str, help='steps of preprocessing to run')

    args = arg_parser.parse_args()

    with open(args.config) as config_file:
        run(json_load(config_file), args.mode)
