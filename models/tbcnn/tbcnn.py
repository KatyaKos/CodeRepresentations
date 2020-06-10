import os
import time

import numpy as np
import models.tbcnn.network as network
import models.tbcnn.sampling as sampling
from sklearn.metrics import accuracy_score
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from models.model import CodeRepresentationModel


class TBCNN(CodeRepresentationModel):
    def __init__(self, config):
        super().__init__(config)
        # build the inputs and outputs of the network
        self.nodes_node, self.children_node, self.hidden_node = network.init_net(
            self.EMBEDDING_DIM,
            self.LABELS_SIZE,
            config.HIDDEN_SIZE
        )
        self.out_node = network.out_layer(self.hidden_node)
        self.labels_node, self.loss_node = None, None
        self.sess = tf.Session()


    def train(self):
        """Train a classifier to label ASTs"""
        # build the inputs and outputs of the network
        self.labels_node, self.loss_node = network.loss_layer(self.hidden_node, self.LABELS_SIZE)
        optimizer = tf.train.AdamOptimizer(self.config.LEARN_RATE)
        optimizer_step = optimizer.minimize(self.loss_node)
        tf.summary.scalar('loss', self.loss_node)

        # init the graph
        self.sess.run(tf.global_variables_initializer())
        with tf.name_scope('saver'):
            saver = tf.train.Saver()
            summaries = tf.summary.merge_all()
            writer = tf.summary.FileWriter(self.config.SAVE_PATH, self.sess.graph)

        checkfile = os.path.join(self.config.SAVE_PATH, 'cnn_tree.ckpt')

        train_loss_, val_loss_, train_acc_, val_acc_ = [], [], [], []
        best_acc = 0.0
        print('Start training...')
        best_sess = self.sess
        for epoch in range(self.config.EPOCHS):
            start_time = time.time()

            total_loss, total_acc, total = self._train_epoch(self.train_data, epoch, optimizer_step, summaries, writer)
            train_loss_.append(total_loss / total)
            train_acc_.append(total_acc / total)

            total_loss, total_acc, total = self._train_epoch(self.val_data, epoch)
            val_loss_.append(total_loss / total)
            val_acc_.append(total_acc / total)

            end_time = time.time()
            if total_acc / total > best_acc:
                best_sess = self.sess
                best_acc = total_acc
            print('[Epoch: %3d/%3d] Training Loss: %.4f, Validation Loss: %.4f,'
                  ' Training Acc: %.3f, Validation Acc: %.3f, Time Cost: %.3f s'
                  % (epoch + 1, self.config.EPOCHS, train_loss_[epoch], val_loss_[epoch],
                     train_acc_[epoch], val_acc_[epoch], end_time - start_time))

            saver.save(self.sess, os.path.join(checkfile), epoch)

        self.sess = best_sess
        saver.save(self.sess, os.path.join(checkfile), self.config.EPOCHS)

        # compute the test accuracy
        total_acc = self._predict_labels(self.test_data)
        print('Accuracy:', total_acc)
        #print(classification_report(correct_labels, predictions, target_names=self.labels))
        #print(confusion_matrix(correct_labels, predictions))

    def evaluate(self):
        """Test a classifier to label ASTs"""
        # init the graph
        with tf.name_scope('saver'):
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(self.config.LOAD_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(self.sess, ckpt.model_checkpoint_path)
            else:
                raise Exception('Checkpoint not found.')

        total_acc = self._predict_labels(self.test_data)
        print('Accuracy:', total_acc)
        #print(classification_report(correct_labels, predictions, target_names=self.labels))
        #print(confusion_matrix(correct_labels, predictions))

    def _train_epoch(self, data, epoch, optimizer_step=None, summaries=None, writer=None):
        total_acc, total_loss, total = 0., 0., 0.
        for i, batch in enumerate(sampling.batch_samples(
                sampling.gen_samples(data, self.labels, self.embedding, self.node_map),
                self.config.BATCH_SIZE
        )):
            nodes, children, batch_labels = batch

            if not nodes:
                continue  # don't try to train on an empty batch

            if optimizer_step:
                _, summary, err, output = self.sess.run(
                    [optimizer_step, summaries, self.loss_node, self.out_node],
                    feed_dict={
                        self.nodes_node: nodes,
                        self.children_node: children,
                        self.labels_node: batch_labels
                    }
                )
                writer.add_summary(summary, epoch)
            else:
                err, output = self.sess.run(
                    [self.loss_node, self.out_node],
                    feed_dict={
                        self.nodes_node: nodes,
                        self.children_node: children,
                        self.labels_node: batch_labels
                    }
                )
            correct_labels = np.argmax(batch_labels, axis=1)
            predictions = np.argmax(output[0], axis=1)
            total_acc += accuracy_score(correct_labels, predictions, normalize=False)
            total_loss += err * len(batch_labels)
            total += len(batch_labels)
        return total_loss, total_acc, total

    def _predict_labels(self, data):
        total_acc, total = 0., 0.
        for batch in sampling.batch_samples(
                sampling.gen_samples(data, self.labels, self.embedding, self.node_map), self.config.BATCH_SIZE
        ):
            nodes, children, batch_labels = batch
            output = self.sess.run([self.out_node],
                              feed_dict={
                                  self.nodes_node: nodes,
                                  self.children_node: children,
                              }
                              )[0]
            correct_labels = np.argmax(batch_labels, axis=1)
            predictions = np.argmax(output, axis=1)
            total_acc += accuracy_score(correct_labels, predictions, normalize=False)
            total += len(batch_labels)
        return total_acc / total
