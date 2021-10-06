import pandas as pd
import numpy as np
from argparse import ArgumentParser
import random


def split_dataframe(data, train_prop=0.9):
    sentence_ids = data.sentence_id.unique().tolist()
    train_count = int(len(sentence_ids) * train_prop)

    train_ids = random.sample(sentence_ids, train_count)

    train_data = data.loc[data.sentence_id.isin(train_ids)]
    test_data = data.loc[~data.sentence_id.isin(train_ids)]

    return train_data, test_data


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_file', type=str, required=True)
    parser.add_argument('--out_file', type=str, default=None)
    parser.add_argument('--split_to_files', action='store_true')
    parser.add_argument('--seed', type=int, default=1353)
    parser.add_argument('--train_to_test_dev_ratio', type=float, default=0.8)
    parser.add_argument('--dev_to_test_ratio', type=float, default=0.5)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    words = [[]]
    tags = [[]]
    spaces_after = [[]]

    with open(args.data_file) as f:
        for line in f:
            if line.startswith('-DOCSTART-'):
                continue
            if len(line.strip()) == 0:
                words.append([])
                tags.append([])
                spaces_after.append([])
                continue

            try:
                word, tag, space = line.strip().split('\t')
            except:
                word, tag = line.strip().split('\t')
                space=' '
            words[-1].append(word)
            tags[-1].append(tag)
            spaces_after[-1].append(space)

        if not words[-1]:
            words = words[:-1]
            tags = tags[:-1]
            spaces_after = spaces_after[:-1]

    data = []
    for example_id, (example_words, example_tags, space_after) in enumerate(zip(words, tags, spaces_after)):
        example_data = pd.DataFrame({'words': example_words, 'labels': example_tags,
                                     'times': space_after, 'sentence_id': example_id})
        data.append(example_data)
    data = pd.concat(data)

    if args.split_to_files:
        train_data, dev_test_data = split_dataframe(data, train_prop=args.train_to_test_dev_ratio)
        dev_data, test_data = split_dataframe(dev_test_data, train_prop=args.dev_to_test_ratio)

        if args.out_file:
            train_data.to_csv(f'{args.out_file}_train_{args.seed}.tsv', sep='\t')
            dev_data.to_csv(f'{args.out_file}_dev_{args.seed}.tsv', sep='\t')
            test_data.to_csv(f'{args.out_file}_test_{args.seed}.tsv', sep='\t')
        else:
            train_data.to_csv(f'train_data_{args.seed}.tsv', sep='\t')
            dev_data.to_csv(f'dev_data_{args.seed}.tsv', sep='\t')
            test_data.to_csv(f'test_data_{args.seed}.tsv', sep='\t')
    else:
        if args.out_file:
            data.to_csv(args.out_file, sep='\t')
        else:
            data.to_csv('data.tsv', sep='\t')


