from argparse import ArgumentParser
import numpy as np
import tensorflow as tf
import sys
#sys.path.append('/home/katyakos/Briksin_projects/MLSE')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-d", "--data", dest="data_path",
                        help="path to preprocessed data", metavar="FILE", required=True)
    parser.add_argument("-e", "--embed", dest="embed_path",
                        help="path to embedding file", metavar="FILE", required=True)
    parser.add_argument("-s", "--save", dest="save_path",
                        help="path to save model", metavar="FILE", required=False)
    parser.add_argument("-l", "--load", dest="load_path",
                        help="path to existing model", metavar="FILE", required=False)
    parser.add_argument("--raw_data", dest="raw_data_path",
                        help="path to raw data, needed to be preprocessed", metavar="FILE", required=False)
    parser.add_argument("--logdir", dest="logdir",
                        help="path to logs", metavar="FILE", required=True)
    parser.add_argument("--predict_path", dest="predict_path",
                        help="path to save predictions", metavar="FILE", required=False)
    parser.add_argument("-m", "--model", dest="model_type",
                        help="type of the model", type=str, required=True)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--preprocess', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    config, model, preprocess_pipe = None, None, None

    if args.model_type == "astnn":
        from models.astnn import config, astnn
        from models.astnn.preprocess import preprocess

        if args.preprocess:
            conf = config.Config.get_preprocess_config(args)
            preprocess_pipe = preprocess.PreprocessPipeline(conf)
        else:
            conf = config.Config.get_model_config(args)
            model = astnn.ASTNN(conf)
    elif args.model_type == "tbcnn":
        from models.tbcnn import config, tbcnn
        from models.tbcnn.preprocess import preprocess

        if args.preprocess:
            conf = config.Config.get_preprocess_config(args)
            preprocess_pipe = preprocess.PreprocessPipeline(conf)
        else:
            conf = config.Config.get_model_config(args)
            model = tbcnn.TBCNN(conf)

    print('Created model')
    if args.preprocess:
        preprocess_pipe.run()
    elif args.evaluate:
        model.evaluate()
    elif args.predict:
        model.predict()
    else:
        model.train()

