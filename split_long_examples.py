import pandas as pd
import argparse
from transformers import AutoTokenizer


def split_long_examples(data, tokenizer, max_seq_len=256, stride=0.8):

    splitted_data = []

    for sentence_id, example in data.groupby('sentence_id'):
        words_with_labels = []
        words_in_example = 0
        tokenized_len = 0
        token_lens = []
        chunk_id = 0

        for word, label, space in zip(example.words, example.labels, example.times):
            tokenized_word = tokenizer.tokenize(word)
            if tokenized_len + len(tokenized_word) >= max_seq_len - 1:
                splitted_data.extend([(w, l, s, f'{sentence_id}_{chunk_id}')
                                      for w, l, s in words_with_labels])
                chunk_id += 1
                offset = int(words_in_example * stride)
                words_with_labels = words_with_labels[offset:]
                tokenized_len -= sum(token_lens[:offset])
                token_lens = token_lens[offset:]
                words_in_example -= offset

            token_lens.append(len(tokenized_word))
            tokenized_len += len(tokenized_word)
            words_with_labels.append((word, label, space))
            words_in_example += 1

        if tokenized_len >= 0:
            splitted_data.extend([(w, l, s, f'{sentence_id}_{chunk_id}')
                                  for w, l, s in words_with_labels])

    return pd.DataFrame(splitted_data, columns=['words', 'labels', 'times', 'sentence_id'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data.tsv', help='path to .tsv file in simpletransformers train file format') # noqa
    parser.add_argument('--max_seq_len', type=int, default=256)
    parser.add_argument('--stride', type=float, default=1)
    parser.add_argument('--out_file', type=str, default='data_splitted.tsv')
    parser.add_argument('--tokenizer_path', type=str, default='allegro/herbert-base-cased')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    data = pd.read_csv(args.data_path, sep='\t', keep_default_na=False,
                       dtype={'words': 'str', 'labels': 'str', 'times': 'str'})

    data = split_long_examples(data, tokenizer, max_seq_len=args.max_seq_len, stride=args.stride)
    data.to_csv(args.out_file, sep='\t')

