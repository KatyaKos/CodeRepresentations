from argparse import ArgumentParser
import numpy as np
import tensorflow as tf
import sys
#sys.path.append('/home/')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-d", "--data", dest="data_path",
                        help="path to preprocessed data", metavar="FILE", required=True)
    parser.add_argument("-e", "--embed", dest="embed_path",
                        help="path to embedding file", metavar="FILE", required=True)
    parser.add_argument("-s", "--save", dest="save_path",
                        help="path to directory with final model", metavar="FILE", required=False)
    parser.add_argument("-l", "--load", dest="load_path",
                        help="path to directory with existing model", metavar="FILE", required=False)
    parser.add_argument("--raw_data", dest="raw_data_path",
                        help="path to raw data needed to be preprocessed", metavar="FILE", required=False)
    parser.add_argument("--predictions", dest="predict_path",
                        help="path to save predictions", metavar="FILE", required=False)
    parser.add_argument("-p", "--parser", dest="parser",
                        help="paresr you want to use, options: pycparser", type=str, required=False)
    parser.add_argument("-m", "--model", dest="model",
                        help="type of the model, options: astnn, tbcnn", type=str, required=True)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--preprocess', action='store_true')
    parser.add_argument('--seed', type=int, default=666)
    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    config, model, preprocess_pipe = None, None, None

    if args.preprocess:
        conf, pars = None, None
        if args.parser == 'pycparser':
            from parsers.pycparser import config, pycparser
            conf = config.Config.get_config(args)
            pars = pycparser.PyCParser()
        else:
            raise Exception("No such parser")
        tree_sampler = None
        if args.model == 'astnn':
            from models.astnn.preprocess.sampler import AstnnSampler
            tree_sampler = AstnnSampler(pars)
        elif args.model == 'tbcnn':
            from models.tbcnn.preprocess.sampler import TbcnnSampler
            tree_sampler = TbcnnSampler(pars)
        else:
            raise Exception("No such model")
        from parsers.pipeline import PreprocessPipeline
        pipeline = PreprocessPipeline(conf, pars, tree_sampler)
        pipeline.run()
    else:
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

