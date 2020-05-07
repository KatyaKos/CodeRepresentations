from argparse import ArgumentParser
import numpy as np
import tensorflow as tf
import os
import sys
sys.path.append('/home/ec2-user/research/MLSE')
print(sys.path)

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
        from models.astnn import config, pipeline, astnn

        if args.raw_data_path is not None:
            conf = config.Config.get_pipeline_config(args)
            pipe = pipeline.Pipeline(conf)
            pipe.run()
            if args.evaluate or args.predict_path is not None:
                args.DATA_PATH = os.path.join(pipe.dest, 'test_blocks.pkl')
       
        conf = config.Config.get_model_config(args)
        model = astnn.ASTNN(conf)

    print('Created model')
    if args.evaluate:
        model.evaluate()
    if args.predict_path is not None:
        model.predict()
    else:
        model.train()

