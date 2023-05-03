import argparse
from preprocess import data_generator
from train import train
import dynamic_selection

import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(dest='command')
    preprocess = subparser.add_parser('preprocess', help='data pre-processing')
    training = subparser.add_parser('train', help='train models')
    ds = subparser.add_parser('ds', help='dynamic selection')

    training.add_argument('--seq_len', type=int, default=50)
    training.add_argument('--return_sequences', action="store_true", default=False)
    training.add_argument('--models', type=str, choices=['all', 'audio', 'video', 'ecg'], default='all')

    ds.add_argument('--k', type=int, default=15)
    ds.add_argument('--seq_len', type=int, default=50)
    ds.add_argument('--return_sequences', action="store_true", default=False)
    ds.add_argument('--saved_information', action="store_true", default=False)

    args = parser.parse_args()
    if args.command == 'preprocess':
        data_generator.data_generation()
    if args.command == 'train':
        seq_len = args.seq_len
        return_sequences = args.return_sequences
        model = args.models
        train.run(seq_len, return_sequences, model)
    if args.command == 'ds':
        seq_len = args.seq_len
        return_sequences = args.return_sequences
        saved_information = args.saved_information
        k = args.saved_information
        dynamic_selection.ds(k, seq_len, return_sequences, saved_information)
