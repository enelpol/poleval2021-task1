import pandas as pd
import os


def load_data_as_dataframe(data_dir, sep=';'):
    files = [file for file in os.listdir(data_dir) if not file.startswith('.')]
    examples = []

    for file in files:

        #print(file)
        #file_df = pd.read_csv(os.path.join(data_dir, file), sep=sep, header=0)
        words = []
        labels = []

        with open(os.path.join(data_dir, file)) as f:
            for idx, l in enumerate(f):
                if idx == 0: #skip header
                    continue

                if l.count(sep) == 3:
                    word, *_ = l.split(sep)
                    tag = ';'
                else:
                    word, tag, _ = l.split(sep)
                    if tag == '':
                        tag = 'B'

                if word not in [':', ';', ',', '.', '-', '...', '?', '!'] and len(word.strip()) > 0:
                    words.append(word.lower())
                    labels.append(tag.strip())
                else:
                    if len(labels) > 0 and labels[-1] == 'B' and len(word.strip()) > 0:
                        labels[-1] = word

            file_df = pd.DataFrame({'words': words, 'labels': labels})
            example_id = file[:-len('.csv')]
            file_df['sentence_id'] = example_id
            #file_df.rename(columns={'word': 'words', 'punctuation': 'labels'})

            #examples = pd.concat([examples, file_df])
            examples.append(file_df)

    return pd.concat(examples)
