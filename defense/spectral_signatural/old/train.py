"""Trains a model, saving checkpoints and tensorboard summaries along
   the way."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
import json
import math
import os
import shutil
import sys
from timeit import default_timer as timer

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tqdm import trange

import dataset_input
from eval import evaluate 
import resnet
import utilities

def train(config):
    # seeding randomness
    tf.set_random_seed(config.training.tf_random_seed)
    np.random.seed(config.training.np_random_seed)

    # Setting up training parameters
    max_num_training_steps = config.training.max_num_training_steps
    step_size_schedule = config.training.step_size_schedule
    weight_decay = config.training.weight_decay
    momentum = config.training.momentum
    batch_size = config.training.batch_size
    eval_during_training = config.training.eval_during_training
    num_clean_examples = config.training.num_examples
    if eval_during_training:
        num_eval_steps = config.training.num_eval_steps

    # Setting up output parameters
    num_output_steps = config.training.num_output_steps
    num_summary_steps = config.training.num_summary_steps
    num_checkpoint_steps = config.training.num_checkpoint_steps

    # Setting up the data and the model
    dataset = dataset_input.CIFAR10Data(config,
                                            seed=config.training.np_random_seed)
    print('Num Poisoned Left: {}'.format(dataset.num_poisoned_left))
    print('Poison Position: {}'.format(config.data.position))
    print('Poison Color: {}'.format(config.data.color))
    num_training_examples = len(dataset.train_data.xs)
    global_step = tf.contrib.framework.get_or_create_global_step()
    model = resnet.Model(config.model)

    # uncomment to get a list of trainable variables
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

    # Setting up the optimizer
    boundaries = [int(sss[0]) for sss in step_size_schedule]
    boundaries = boundaries[1:]
    values = [sss[1] for sss in step_size_schedule]
    learning_rate = tf.train.piecewise_constant(
        tf.cast(global_step, tf.int32),
        boundaries,
        values)
    total_loss = model.mean_xent + weight_decay * model.weight_decay_loss

    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
    train_step = optimizer.minimize( total_loss, global_step=global_step)

    # Setting up the Tensorboard and checkpoint outputs
    model_dir = config.model.output_dir
    if eval_during_training:
        eval_dir = os.path.join(model_dir, 'eval')
        if not os.path.exists(eval_dir):
          os.makedirs(eval_dir)

    # We add accuracy and xent twice so we can easily make three types of
    # comparisons in Tensorboard:
    # - train vs eval (for a single run)
    # - train of different runs
    # - eval of different runs

    saver = tf.train.Saver(max_to_keep=3)

    tf.summary.scalar('accuracy_nat_train', model.accuracy, collections=['nat'])
    tf.summary.scalar('accuracy_nat', model.accuracy, collections = ['nat'])
    tf.summary.scalar('xent_nat_train', model.xent / batch_size,
                                                        collections=['nat'])
    tf.summary.scalar('xent_nat', model.xent / batch_size, collections=['nat'])
    tf.summary.image('images_nat_train', model.train_xs, collections=['nat'])
    tf.summary.scalar('learning_rate', learning_rate, collections=['nat'])
    nat_summaries = tf.summary.merge_all('nat')

    with tf.Session() as sess:
      print('Dataset Size: ', len(dataset.train_data.xs))

      # Initialize the summary writer, global variables, and our time counter.
      summary_writer = tf.summary.FileWriter(model_dir, sess.graph)
      if eval_during_training:
          eval_summary_writer = tf.summary.FileWriter(eval_dir)

      sess.run(tf.global_variables_initializer())
      training_time = 0.0

      # Main training loop
      for ii in range(max_num_training_steps+1):
        x_batch, y_batch = dataset.train_data.get_next_batch(batch_size,
                                                           multiple_passes=True)

        nat_dict = {model.x_input: x_batch,
                    model.y_input: y_batch,
                    model.is_training: False}

        # Output to stdout
        if ii % num_output_steps == 0:
          nat_acc = sess.run(model.accuracy, feed_dict=nat_dict)
          print('Step {}:    ({})'.format(ii, datetime.now()))
          print('    training nat accuracy {:.4}%'.format(nat_acc * 100))
          if ii != 0:
            print('    {} examples per second'.format(
                num_output_steps * batch_size / training_time))
            training_time = 0.0

        # Tensorboard summaries
        if ii % num_summary_steps == 0:
          summary = sess.run(nat_summaries, feed_dict=nat_dict)
          summary_writer.add_summary(summary, global_step.eval(sess))

        # Write a checkpoint
        if ii % num_checkpoint_steps == 0:
          saver.save(sess,
                     os.path.join(model_dir, 'checkpoint'),
                     global_step=global_step)

        if eval_during_training and ii % num_eval_steps == 0:  
            evaluate(model, sess, config, eval_summary_writer)

        # Actual training step
        start = timer()
        nat_dict[model.is_training] = True
        sess.run(train_step, feed_dict=nat_dict)
        end = timer()
        training_time += end - start


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train script options',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--config', type=str,
                        help='path to config file',
                        default='config.json', required=False)
    args = parser.parse_args()

    config_dict = utilities.get_config(args.config)

    model_dir = config_dict['model']['output_dir']
    if not os.path.exists(model_dir):
      os.makedirs(model_dir)

    # keep the configuration file with the model for reproducibility
    with open(os.path.join(model_dir, 'config.json'), 'w') as f:
        json.dump(config_dict, f, sort_keys=True, indent=4)

    config = utilities.config_to_namedtuple(config_dict)
    train(config)
