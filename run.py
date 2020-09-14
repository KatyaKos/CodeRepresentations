from argparse import ArgumentParser
from json import load as json_load
from typing import Dict

import numpy as np
import tensorflow as tf

#sys.path.append('')


def run(params: Dict, mode: str):
    model = None
    print('Creating model...')
    if params['model'] == "astnn":
        from models.astnn import astnn
        model = astnn.ASTNN(params)
    elif params['model'] == "tbcnn":
        from models.tbcnn import tbcnn
        model = tbcnn.TBCNN(params)
    else:
        raise ValueError("No such model...")

    if mode == 'evaluate':
        print('Starting evaluation...')
        model.evaluate()
    elif mode == 'predict':
        print('Starting prediction...')
        model.predict()
    else:
        print('Starting training...')
        model.train()


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('-m', '--mode', dest='mode', type=str,
                        help='Working mode: \'train\', \'evaluate\' or \'predict\'. Train by default.')
    parser.add_argument('-c', '--config', dest='config', type=str, help='path to config json')
    parser.add_argument('--seed', type=int, default=666)
    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    with open(args.config) as config_file:
        run(json_load(config_file), args.mode)

