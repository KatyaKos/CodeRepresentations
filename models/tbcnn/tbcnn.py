"""Train the cnn model as  described in Lili Mou et al. (2015)
https://arxiv.org/pdf/1409.5718.pdf"""

import os
import pickle
import numpy as np
import models.tbcnn.network as network
import models.tbcnn.sampling as sampling
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class TBCNN:

    def __init__(self, config):
        self.config = config

    def train(self):
        """Train a classifier to label ASTs"""

        with open(self.config.DATA_PATH, 'rb') as fh:
            trees, _, labels = pickle.load(fh)
            labels = [str(l) for l in labels]
            trees = [{'tree': t['tree'], 'label': str(t['label'])} for t in trees]

        with open(self.config.EMBEDDING_PATH, 'rb') as fh:
            embeddings, embed_lookup = pickle.load(fh)
            num_feats = len(embeddings[0])

        # build the inputs and outputs of the network
        nodes_node, children_node, hidden_node = network.init_net(
            num_feats,
            len(labels)
        )

        out_node = network.out_layer(hidden_node)
        labels_node, loss_node = network.loss_layer(hidden_node, len(labels))

        optimizer = tf.train.AdamOptimizer(self.config.LEARN_RATE)
        train_step = optimizer.minimize(loss_node)

        tf.summary.scalar('loss', loss_node)

        ### init the graph
        sess = tf.Session()  # config=tf.ConfigProto(device_count={'GPU':0}))
        sess.run(tf.global_variables_initializer())

        with tf.name_scope('saver'):
            saver = tf.train.Saver()
            summaries = tf.summary.merge_all()
            writer = tf.summary.FileWriter(self.config.LOGDIR, sess.graph)

        checkfile = os.path.join(self.config.LOGDIR, 'cnn_tree.ckpt')

        num_batches = len(trees) // self.config.BATCH_SIZE + (1 if len(trees) % self.config.BATCH_SIZE != 0 else 0)
        for epoch in range(1, self.config.EPOCHS + 1):
            for i, batch in enumerate(sampling.batch_samples(
                    sampling.gen_samples(trees, labels, embeddings, embed_lookup), self.config.BATCH_SIZE
            )):
                nodes, children, batch_labels = batch
                step = (epoch - 1) * num_batches + i * self.config.BATCH_SIZE

                if not nodes:
                    continue  # don't try to train on an empty batch

                _, summary, err, out = sess.run(
                    [train_step, summaries, loss_node, out_node],
                    feed_dict={
                        nodes_node: nodes,
                        children_node: children,
                        labels_node: batch_labels
                    }
                )

                writer.add_summary(summary, step)
                if step % self.config.CHECKPOINT_STEP == 0:
                    # save state so we can resume later
                    saver.save(sess, os.path.join(checkfile), step)
                    print('Checkpoint saved.')

            print('Epoch:', epoch, 'Step:', step, 'Loss:', err)

        saver.save(sess, os.path.join(checkfile), step)

        # compute the training accuracy
        correct_labels = []
        predictions = []
        print('Computing training accuracy...')
        for batch in sampling.batch_samples(
                sampling.gen_samples(trees, labels, embeddings, embed_lookup), 1
        ):
            nodes, children, batch_labels = batch
            output = sess.run([out_node],
                              feed_dict={
                                  nodes_node: nodes,
                                  children_node: children,
                              }
                              )
            correct_labels.append(np.argmax(batch_labels))
            predictions.append(np.argmax(output))

        target_names = list(labels)

        #tmp = [(t1, t2) for t1, t2 in zip(correct_labels[:30], predictions[:30])]
        #print(tmp)
        print('Accuracy:', accuracy_score(correct_labels, predictions))
        print(classification_report(correct_labels, predictions, target_names=target_names))
        print(confusion_matrix(correct_labels, predictions))

    def evaluate(self):
        """Test a classifier to label ASTs"""

        with open(self.config.DATA_PATH, 'rb') as fh:
            _, trees, labels = pickle.load(fh)

        with open(self.config.EMBEDDING_PATH, 'rb') as fh:
            embeddings, embed_lookup = pickle.load(fh)
            num_feats = len(embeddings[0])

        # build the inputs and outputs of the network
        nodes_node, children_node, hidden_node = network.init_net(
            num_feats,
            len(labels)
        )
        out_node = network.out_layer(hidden_node)

        # init the graph
        sess = tf.Session()  # config=tf.ConfigProto(device_count={'GPU':0}))
        sess.run(tf.global_variables_initializer())

        with tf.name_scope('saver'):
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(self.config.LOGDIR)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                raise Exception('Checkpoint not found.')

        correct_labels = []
        # make predicitons from the input
        predictions = []
        step = 0
        for batch in sampling.batch_samples(
                sampling.gen_samples(trees, labels, embeddings, embed_lookup), 1
        ):
            nodes, children, batch_labels = batch
            output = sess.run([out_node],
                              feed_dict={
                                  nodes_node: nodes,
                                  children_node: children,
                              }
                              )
            correct_labels.append(np.argmax(batch_labels))
            predictions.append(np.argmax(output))
            step += 1
            print(step, '/', len(trees))

        target_names = list(labels)
        print('Accuracy:', accuracy_score(correct_labels, predictions))
        print(classification_report(correct_labels, predictions, target_names=target_names))
        print(confusion_matrix(correct_labels, predictions))
