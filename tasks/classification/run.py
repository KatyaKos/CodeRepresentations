from argparse import ArgumentParser
import numpy as np
import tensorflow as tf
import os

from models import astnn

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--preprocess", dest="raw_data_path",
                        help="path to raw data, needed to be preprocessed", metavar="FILE", required=False)
    parser.add_argument("-d", "--data", dest="data_path",
                        help="path to preprocessed data", metavar="FILE", required=True)
    parser.add_argument("-s", "--save", dest="save_path",
                        help="path to save model", metavar="FILE", required=False)
    parser.add_argument("-l", "--load", dest="load_path",
                        help="path to existing model", metavar="FILE", required=False)
    parser.add_argument("--predict", dest="predict_path",
                        help="path to save predictions", metavar="FILE", required=False)
    parser.add_argument("-m", "--model", dest="model_type",
                        help="type of the model", type=str, required=True)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    config, model = None, None

    if args.model_type == "astnn":
        config = astnn.config.Config.get_default_config(args)
        if args.raw_data_path is not None:
            pipe = astnn.pipeline.Pipeline(config)
            pipe.run()
            if args.evaluate or args.predict is not None:
                config.DATA_PATH = os.path.join(pipe.dest, 'test_blocks.pkl')
        model = astnn.astnn.ASTNN(config)

    print('Created model')
    if args.evaluate:
        model.evaluate()
    if args.predict is not None:
        model.predict()
    else:
        model.train()

