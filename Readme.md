# PolEval 2021 Task 1: Punctuation restoration from read text

## Results

|                   | test-D weighted F1 |
|-------------------|--------------------|
| AE1               | 80.09              |
| E1                | 80.86              |
| SE1               | 81.27              |
| S1                | **81.29**       |
|||
| Norbert Ropiak    | 81.25              |
| Michał Marcińczuk | 81.23              |

## Instructions

```shell
git clone -b secret https://github.com/poleval/2021-punctuation-restoration.git
mkdir data
virtualenv -p python3 venv
source venv/bin/activate
pip3 install -r requirements.txt
```

You can use model `enelpol/poleval2021-task1` or train yourself.

### Training


```shell
python3 parse_tsv.py 2021-punctuation-restoration/train/in.tsv 2021-punctuation-restoration/train/expected.tsv original_train.conll
python3 parse_tsv.py 2021-punctuation-restoration/test-A/in.tsv 2021-punctuation-restoration/test-A/expected.tsv original_test-A.conll
```
Download and unpack: https://drive.google.com/file/d/10SdpLHPLXVfhJsq1okgC5fcxbFzCGoR5/view?usp=sharing
```shell
python3 parse_data.py rest/wikinews/all/json/ rest/wikitalks/all/json/ --save_path rest
```


```shell
python3 parse_tsv.py rest/train_in.tsv rest/train_expected.tsv train.conll
python3 parse_tsv.py rest/test_in.tsv rest/test_expected.tsv test.conll
python3 parse_tsv.py rest/rest_in.tsv rest/rest_expected.tsv rest.conll
```
```shell
python3 convert_to_pandas.py --data_file rest.conll --out_file rest.tsv
python3 convert_to_pandas.py --data_file original_test-A.conll --out_file original_test-A.tsv
python3 convert_to_pandas.py --data_file original_train.conll --out_file original_train.tsv
```
```shell
python3 split_long_examples.py --data_path rest.tsv --out_file rest.tsv.s
python3 split_long_examples.py --data_path original_train.tsv --out_file original_train.tsv.s
python3 split_long_examples.py --data_path original_test-A.tsv --out_file original_test-A.tsv.s
```

Train on rest data.
```shell
python3 train_ner_punctuation.py --model_type herbert --model_name allegro/herbert-large-cased --train_data_dir rest.tsv.s --test_data_dir original_test-A.tsv.s --eval_data_dir original_train.tsv.s --wandb_project poleval2021_task1 --epochs 5 --learning_rate 2e-5 --batch_size 12 --acc 1 --warmup_steps 0 --eval_steps 200 --eval_during_training --max_seq_len 256 --seed 2
```

Train on test-A.
```shell
MODEL=""
python3 train_ner_punctuation.py --model_type herbert --model_name $MODEL --train_data_dir original_test-A.tsv.s --test_data_dir original_train.tsv.s --eval_data_dir original_train.tsv.s --wandb_project poleval2021_task1 --epochs 5 --learning_rate 1e-5 --batch_size 12 --acc 1 --warmup_steps 0 --eval_steps 20 --eval_during_training --max_seq_len 256 --seed 2
````

### Predict

```shell
MODEL="enelpol/poleval2021-task1" #edit
python3 test_model.py --path_to_model $MODEL --path_to_test 2021-punctuation-restoration/test-D/in.tsv --path_to_out test-D.tsv
````









