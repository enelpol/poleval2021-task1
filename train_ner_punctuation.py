import pandas as pd
import numpy as np
import sklearn
import argparse
import os
from simpletransformers.ner import NERArgs, NERModel
import torch
import time
from utils import load_data_as_dataframe
import wandb


labels = ['B', ':', ';', ',', '.', '-', '...', '?', '!']


def f1_per_label(y_true, y_pred):
    values = merge_data(sklearn.metrics.f1_score, y_true, y_pred, labels=labels[1:], average=None, zero_division=0)
    return {str(i): v for i, v in enumerate(values)}


def pr_per_label(y_true, y_pred):
    values = merge_data(sklearn.metrics.precision_score, y_true, y_pred, labels=labels[1:], average=None, zero_division=0)
    return {str(i): v for i, v in enumerate(values)}
    # return values[label]


def rc_per_label(y_true, y_pred):
    values = merge_data(sklearn.metrics.recall_score, y_true, y_pred, labels=labels[1:], average=None, zero_division=0)
    return {str(i): v for i, v in enumerate(values)}
    # return values[label]


def merge_data(func, y_true, y_pred, **kwargs):
    y_true_res = []
    y_pred_res = []

    for t in y_true:
        y_true_res.extend(t)

    for p in y_pred:
        y_pred_res.extend(p)

    return func(y_true_res, y_pred_res, **kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_dir', type=str, default='ner_train', help='path to train data directory')
    parser.add_argument('--eval_data_dir', type=str, default='ner_dev', help='path to train data directory')
    parser.add_argument('--test_data_dir', type=str, default='ner_test', help='path to test data directory')
    parser.add_argument('--wandb_project', type=str, default='poleval2021_task1', help='wandb project id')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='learnign rate')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--acc', type=int, default=1, help='gradient accumulation steps')
    parser.add_argument('--model_name', type=str, default='allegro/herbert-base-cased', help='path to model')
    parser.add_argument('--model_type', type=str, default='herbert', help='model type')
    parser.add_argument('--warmup_steps', type=int, default=1, help='warmup steps')
    parser.add_argument('--eval_steps', type=int, default=100, help='eval steps')
    parser.add_argument('--eval_during_training', action='store_true')
    parser.add_argument('--max_seq_len', type=int,  default=256)
    parser.add_argument('--use_dice', action='store_true')
    parser.add_argument('--use_focal', action='store_true')
    parser.add_argument('--focal_alpha', default=0.25, type=float)
    parser.add_argument('--seed', type=int, default=81945)
    parser.add_argument('--early_stopping_metric', type=str, default='f1_weighted')
    parser.add_argument('--weights', nargs="+", type=float, default=None)
    args = parser.parse_args()

    print(args.weights)

    #train_data = load_data_as_dataframe(args.train_data_dir)
    #eval_data = load_data_as_dataframe(args.eval_data_dir)

    train_data = pd.read_csv(args.train_data_dir, sep='\t', header=0)
    eval_data = pd.read_csv(args.eval_data_dir, sep='\t', header=0)

    sentences_to_delete = train_data[~train_data.labels.isin(labels)].sentence_id #coś to robi?
    train_data = train_data.loc[~train_data.sentence_id.isin(sentences_to_delete)]

    sentences_to_delete = eval_data[~eval_data.labels.isin(labels)].sentence_id
    eval_data = eval_data.loc[~eval_data.sentence_id.isin(sentences_to_delete)]

    ner_args = NERArgs()
    ner_args.early_stopping_metric = args.early_stopping_metric
    ner_args.early_stopping_metric_minimize = False
    ner_args.model_type = args.model_type
    ner_args.model_name = args.model_name
    ner_args.wandb_project = args.wandb_project
    ner_args.wandb_kwargs = {"settings": wandb.Settings(start_method="thread")}
    ner_args.train_batch_size = args.batch_size
    ner_args.eval_batch_size = args.batch_size
    ner_args.gradient_accumulation_steps = args.acc
    ner_args.learning_rate = args.learning_rate
    ner_args.num_train_epochs = args.epochs
    ner_args.evaluate_during_training = args.eval_during_training
    ner_args.evaluate_during_training_steps = 20 # args.eval_steps
    ner_args.max_seq_length = args.max_seq_len
    ner_args.manual_seed = args.seed
    ner_args.warmup_steps = args.warmup_steps
    ner_args.save_eval_checkpoints = False
    ner_args.use_multiprocessing = False
    ner_args.use_multiprocessing_for_evaluation = False

    if args.use_dice:
        ner_args.loss_type = 'dice'
        ner_args.loss_args = {
            'smooth': 0.001,
            'square_denominator': True,
            'with_logits': True,
            'ohem_ratio': 0.0,
            'alpha': 0,
            'reduction': "mean",
            'index_label_position': True
        }
    if args.use_focal:
        ner_args.loss_type = 'focal'
        ner_args.loss_args = {
            'alpha': args.focal_alpha,
            'gamma': 2,
            'reduction': 'mean',
            'eps': 1e-6,
            'ignore_index': -100,
        }

    metrics = {
        'f1_micro': lambda y_true, y_pred: merge_data(sklearn.metrics.f1_score, y_true, y_pred, average='micro',
                                                      zero_division=0, labels=labels[1:]),
        'f1_macro': lambda y_true, y_pred: merge_data(sklearn.metrics.f1_score, y_true, y_pred, average='macro',
                                                      zero_division=0, labels=labels[1:]),
        'f1_weighted': lambda y_true, y_pred: merge_data(sklearn.metrics.f1_score, y_true, y_pred, average='weighted',
                                                         zero_division=0, labels=labels[1:]),
        'pr_micro': lambda y_true, y_pred: merge_data(sklearn.metrics.precision_score, y_true, y_pred, average='micro',
                                                      zero_division=0, labels=labels[1:]),
        'pr_macro': lambda y_true, y_pred: merge_data(sklearn.metrics.precision_score, y_true, y_pred, average='macro',
                                                      zero_division=0, labels=labels[1:]),
        'pr_weighted': lambda y_true, y_pred: merge_data(sklearn.metrics.precision_score, y_true, y_pred, average='weighted',
                                                      zero_division=0, labels=labels[1:]),
        'rc_micro': lambda y_true, y_pred: merge_data(sklearn.metrics.recall_score, y_true, y_pred, average='micro',
                                                                        zero_division=0, labels=labels[1:]),
        'rc_macro': lambda y_true, y_pred: merge_data(sklearn.metrics.recall_score, y_true, y_pred, average='macro',
                                                                        zero_division=0, labels=labels[1:]),
        'rc_weighted': lambda y_true, y_pred: merge_data(sklearn.metrics.recall_score, y_true, y_pred, average='weighted',
                                                      zero_division=0, labels=labels[1:]),
        #'classification_report': lambda y_true, y_pred: sklearn.metrics.classification_report(y_true, y_pred,
        #                                                                                      output_dict=True),
        'confusion_matrix': lambda y_true, y_pred: merge_data(sklearn.metrics.confusion_matrix, y_true, y_pred,
                                                              labels=labels[1:]),

        'f1_class': lambda y_true, y_pred: f1_per_label(y_true, y_pred),
        'pr_class': lambda y_true, y_pred: pr_per_label(y_true, y_pred),
        'rc_class': lambda y_true, y_pred: rc_per_label(y_true, y_pred),
    }

    #labels = list(train_data.labels.unique())
    train_data.words = train_data.words.astype(str)
    eval_data.words = eval_data.words.astype(str)
    print('labels', labels)
    start = time.time()
    output_dir = f'model_dir_{args.model_name}_{start}'

    ner_args.output_dir = output_dir
    ner_args.best_model_dir = os.path.join(output_dir, 'best_model')
    model = NERModel(args.model_type, args.model_name,
                     labels=labels, args=ner_args, weight=args.weights,
                     use_cuda=True if torch.cuda.is_available() else False)

    model.train_model(train_data, output_dir=output_dir, eval_data=eval_data, **metrics)

    #with open('test-A/in.tsv') as f:
    #    to_predict = [line.split('\t')[1] for line in f]

    #print(model.predict(to_predict, split_on_space=True))

