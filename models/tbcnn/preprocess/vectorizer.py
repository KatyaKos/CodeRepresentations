"""Train the ast2vect network."""

import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import math
import pandas as pd
import pickle


def learn_vectors(samples, node_map,
                  output_path, logdir, log_file,
                  num_features, batch_size, hidden_size,
                  learn_rate, epochs, checkpoint_step):
    # build the inputs and outputs of the network
    input_node, label_node, embed_node, loss_node = _init_net(batch_size, num_features, hidden_size, len(node_map))
    # use gradient descent with momentum to minimize the training objective
    train_step = tf.train.GradientDescentOptimizer(learn_rate).minimize(loss_node)
    tf.summary.scalar('loss', loss_node)
    # init the graph
    sess = tf.Session()
    with tf.name_scope('saver'):
        from tensorboard.plugins import projector
        saver = tf.train.Saver()
        summaries = tf.summary.merge_all()
        writer = tf.summary.FileWriter(logdir, sess.graph)
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = embed_node.name
        embedding.metadata_path = os.path.join(logdir, 'embed_metadata.tsv')
        projector.visualize_embeddings(writer, config)
    sess.run(tf.global_variables_initializer())

    log_path = os.path.join(logdir, log_file)
    fout = open(output_path, 'wb')
    step = 0
    embed = None
    for epoch in range(1, epochs + 1):
        sample_gen = _batch_samples(samples, batch_size, node_map)
        for batch in sample_gen:
            input_batch, label_batch = batch
            _, summary, embed, err = sess.run(
                [train_step, summaries, embed_node, loss_node],
                feed_dict={
                    input_node: input_batch,
                    label_node: label_batch
                }
            )

            writer.add_summary(summary, step)
            if step % checkpoint_step == 0:
                # save state so we can resume later
                saver.save(sess, log_path, step)
                print('Checkpoint saved.')
                # save embeddings
                pickle.dump((embed, node_map), fout)
                #df = pd.DataFrame(embed, columns=NODE_LIST)
                #df.to_pickle(output_path)
            step += 1
        print('Epoch: ', epoch, 'Loss: ', err)
    saver.save(sess, log_path, step)
    fout.close()
    return embed


def _init_net(batch_size, num_feats, hidden_size, nodes_num):
    """Construct the network graph."""

    with tf.name_scope('network'):

        with tf.name_scope('inputs'):
            # input node-child pairs
            inputs = tf.placeholder(tf.int32, shape=[batch_size,], name='inputs')
            labels = tf.placeholder(tf.int32, shape=[batch_size,], name='labels')

            # embeddings to learn
            embeddings = tf.Variable(
                tf.random_uniform([nodes_num, num_feats]), name='embeddings'
            )

            embed = tf.nn.embedding_lookup(embeddings, inputs)
            onehot_labels = tf.one_hot(labels, nodes_num, dtype=tf.float32)

        # weights will have features on the rows and nodes on the columns
        with tf.name_scope('hidden'):
            weights = tf.Variable(
                tf.truncated_normal(
                    [num_feats, hidden_size], stddev=1.0 / math.sqrt(num_feats)
                ),
                name='weights'
            )

            biases = tf.Variable(
                tf.zeros((hidden_size,)),
                name='biases'
            )

            hidden = tf.tanh(tf.matmul(embed, weights) + biases)

        with tf.name_scope('softmax'):
            weights = tf.Variable(
                tf.truncated_normal(
                    [hidden_size, nodes_num],
                    stddev=1.0 / math.sqrt(hidden_size)
                ),
                name='weights'
            )
            biases = tf.Variable(
                tf.zeros((nodes_num,), name='biases')
            )

            logits = tf.matmul(hidden, weights) + biases

        with tf.name_scope('error'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                labels=onehot_labels, logits=logits, name='cross_entropy'
            )

            loss = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')

    return inputs, labels, embeddings, loss


def _batch_samples(samples, batch_size, NODE_MAP):
    """Batch samples and return batches in a generator."""
    batch = ([], [])
    count = 0
    index_of = lambda x: NODE_MAP[x]
    for parent, node in zip(samples['parent'], samples['node']):
        if parent is not None:
            batch[0].append(index_of(node))
            batch[1].append(index_of(parent))
            count += 1
            if count >= batch_size:
                yield batch
                batch, count = ([], []), 0
