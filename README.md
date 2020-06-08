# Code Representations
There is a vast variety of code representation algorithms that can be divided into several classes, e.g. natural language models (BERT, XLNet), graph embeddings (GGNN), AST embeddings (code2seq, ASTNN). These models were introduced being evaluated on different tasks and datasets and were never properly compared. Moreover, parsers that these models use for preprocessing are also quite different, which might affect the results.

The objective of this work is to apply a set of models (of different classes) and parsers to the same tasks/datasets and find out which class achieves better results on each task. In the end, we also expect to have a Python library with all the trained models and a unified interface that allows to use them in other practical tasks.

## How To:
MODEL: \['astnn', 'tbcnn'\]; PARSER: \['pycparser'\]

To preprocess the dataset (poj.pkl in tasks/classification/data):
python run.py --preprocess -p PARSER -m MODEL -d DEST --raw_data PATH_TO_PKL -e DEST_EMBEDDING_FILE

To train:
python run.py -m MODEL -d PREPROCESSED_TREES_FILE -e EMBEDDING_FILE -s DEST_MODEL_DIR

To evaluate:
python run.py -m MODEL --evaluate -d PREPROCESSED_TREES_FILE -e EMBEDDING_FILE -l TRAINED_MODEL_DIR

## Current work:

Models: [ASTNN](https://github.com/zhangj111/astnn), [TBCNN](https://github.com/crestonbunch/tbcnn)

Datasets: POJ104 (code classification)

Parsers: PyCParser
