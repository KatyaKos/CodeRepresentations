from argparse import ArgumentParser
from json import load as json_load
import numpy as np
import tensorflow as tf

#sys.path.append('')


def run(params):

    model = None
    if args.model == "astnn":
        from models.astnn import config, astnn
        conf = config.Config.get_config(args)
        model = astnn.ASTNN(conf)
    elif args.model == "tbcnn":
        from models.tbcnn import config, tbcnn
        conf = config.Config.get_config(args)
        model = tbcnn.TBCNN(conf)
    print('Created model')
    if args.evaluate:
        model.evaluate()
    elif args.predict:
        model.predict()
    else:
        model.train()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-eval', '--evaluate', action='store_true')
    parser.add_argument('-pr', '--predict', action='store_true')
    parser.add_argument('-c', '--config', type=str, help='path to config json')
    parser.add_argument('--seed', type=int, default=666)
    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    with open(args.config) as config_file:
        run(json_load(config_file))

