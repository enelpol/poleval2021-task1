import glob
import os.path
import random
from argparse import ArgumentParser

from parse_json2 import load_json


def read_names(path):
    names = []
    for in_line in open(path):
        if in_line[-1]=='\n': in_line = in_line[:-1]
        name, text = in_line.split('\t')
        names.append(name)
    return names


if __name__ == "__main__":
    parser = ArgumentParser(description='convert JSON to CONLL')
    parser.add_argument('--train_path', default='2021-punctuation-restoration/train/in.tsv',
                        help='path to train in.tsv')
    parser.add_argument('--test_path', default='2021-punctuation-restoration/test-A/in.tsv', help='path to test in.tsv')
    parser.add_argument('data', nargs='+', help='paths to dirs with JSONs')
    parser.add_argument('--save_path', default='.', help='path to directory')
    args = parser.parse_args()

    # out = open(args.save_path, 'w')

    train_names = read_names(args.train_path)
    test_names = read_names(args.test_path)

    # print(train_names)

    train_paths = []
    test_paths = []
    rest_paths = []
    for path in args.data:
        json_paths = glob.glob(path + '/*.json')
        # print(json_paths)
        for json_path in json_paths:
            basename = os.path.basename(json_path).split('.')[0]
            if basename in train_names:
                train_paths.append(json_path)
            elif basename in test_names:
                test_paths.append(json_path)
            else:
                rest_paths.append(json_path)

    print(len(test_names) , len(test_paths), len(train_names) , len(train_paths))
    print(test_names)
    print(test_paths)
    # assert len(test_names) == len(test_paths)
    # assert len(train_names) == len(train_paths)

    #order in train and test the same as in original

    test_paths2=[]
    for name in test_names:
        for path in test_paths:
            if name in path:
                test_paths2.append(path)
                break
    test_paths=test_paths2

    train_paths2=[]
    for name in train_names:
        for path in train_paths:
            if name in path:
                train_paths2.append(path)
                break
    train_paths=train_paths2
        

    random.seed(0)
    random.shuffle(rest_paths)

    for name, paths in [('test',test_paths),('train', train_paths),('rest', rest_paths)]:
        with open(args.save_path+f'/{name}_expected.tsv', 'w') as out_expected, \
                open(args.save_path+f'/{name}_in.tsv', 'w') as out_in:
            for path in paths:
                json_in, json_expected = load_json(path)
                
                basename = os.path.basename(path).split('.')[0]
                
                out_in.write(f'{basename}\t{json_in}\n')
                out_expected.write(f'{json_expected}\n')
        