"""Train the model"""

import argparse
import logging
import os
import random

import tensorflow as tf

from model.dataset import Dataset
from model.model import Model
from model.utils import Params, set_logger

from model.training import train_and_evaluate

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/demo',help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default='data',help="Directory containing the dataset")
parser.add_argument('--restore_from', default=None,help="Optional, directory or file containing weights to reload before training")

if __name__ == '__main__':
    # Get arguments 
    args = parser.parse_args()

    # Set the random seed for the whole graph
    tf.set_random_seed(100)

    # Set the logger
    set_logger(os.path.join(args.model_dir, 'train.log'))

    # Load the parameters
    params = Params(os.path.join(args.model_dir, 'params.json'))

    # Initialize the dataset for training
    dataset = Dataset(args.data_dir, params)

    # Create the model
    model = Model(dataset, params)

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(model.train_spec, model.eval_spec, args.model_dir, params, args.restore_from)
