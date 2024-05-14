import argparse
import torch
import datetime
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=567, help="Random seed.")
    parser.add_argument(
        "--GPU_to_use", type=int, default=None, help="GPU to use for training"
    )

    ############## training hyperparameter ##############
    parser.add_argument(
        "--epochs", type=int, default=500, help="Number of epochs to train."
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Number of samples per batch."
    )
    parser.add_argument(
        "--lr", type=float, default=0.0005, help="Initial learning rate."
    )

     ############## architecture ##############
    parser.add_argument(
        "--encoder_hidden", type=int, default=256, help="Number of hidden units."
    )
    parser.add_argument(
        "--decoder_hidden", type=int, default=256, help="Number of hidden units."
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="mlp",
        help="Type of path encoder model (mlp or cnn).",
    )
    parser.add_argument(
        "--decoder",
        type=str,
        default="mlp",
        help="Type of decoder model (mlp, rnn, or sim).",
    )

    ############## loading and saving ##############
    parser.add_argument(
        "--suffix",
        type=str,
        default="_springs5",
        help='Suffix for training data.',
    )
    parser.add_argument(
        "--timesteps", type=int, default=49, help="Number of timesteps in input."
    )
    parser.add_argument(
        "--num_atoms", type=int, default=5, help="Number of time-series in input."
    )
    parser.add_argument(
        "--dims", type=int, default=4, help="Dimensionality of input."
    )
    parser.add_argument(
        "--datadir",
        type=str,
        default="./data",
        help="Name of directory where data is stored.",
    )
    parser.add_argument(
        "--save_folder",
        type=str,
        default="logs",
        help="Where to save the trained model, leave empty to not save anything.",
    )
    parser.add_argument(
        "--expername",
        type=str,
        default="",
        help="If given, creates a symlinked directory by this name in logdir"
        "linked to the results file in save_folder"
        "(be careful, this can overwrite previous results)",
    )
    parser.add_argument(
        "--sym_save_folder",
        type=str,
        default="../logs",
        help="Name of directory where symlinked named experiment is created."
    )
    parser.add_argument(
        "--load_folder",
        type=str,
        default="",
        help="Where to load pre-trained model if finetuning/evaluating. "
        + "Leave empty to train from scratch",
    )
