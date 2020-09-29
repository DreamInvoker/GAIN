# GAIN
PyTorch implementation for EMNLP 2020 paper: [Double Graph Based Reasoning for Document-level Relation Extraction]()

## 1. Environments

- python         3.7.4
- cuda           10.2
- Ubuntu-18.0.4  4.15.0-65-generic

## 2. Dependencies

- numpy          1.19.2
- matplotlib     3.3.2
- torch          1.6.0
- transformers   3.1.0
- dgl-cu102      0.4.3
- scikit-learn   0.23.2

PS: dgl >= 0.5 is not compatible with our code, we will fix this compatibility problem in the future.

## 3. Preparation

### 3.1. Dataset
- Download data from [Google Drive link](https://drive.google.com/drive/folders/1c5-0YwnoJx8NS6CV2f-NoTHR__BdkNqw) shared by DocRED authors

- Put `train_annotated.json`, `dev.json`, `test.json`, `word2id.json`, `ner2id.json`, `rel2id.json`, `vec.npy` into the directory `data/`

- If you want to use other datasets, please first process them to fit the same format as DocRED.

### 3.2. (Optional) Pre-trained Language Models
Following the hint in this [link](http://viewsetting.xyz/2019/10/17/pytorch_transformers/?nsukey=v0sWRSl5BbNLDI3eWyUvd1HlPVJiEOiV%2Fk8adAy5VryF9JNLUt1TidZkzaDANBUG6yb6ZGywa9Qa7qiP3KssXrGXeNC1S21IyT6HZq6%2BZ71K1ADF1jKBTGkgRHaarcXIA5%2B1cUq%2BdM%2FhoJVzgDoM7lcmJg9%2Be6NarwsZzpwAbAwjHTLv5b2uQzsSrYwJEdPl7q9O70SmzCJ1VF511vwxKA%3D%3D), download possible required files (`pytorch_model.bin`, `config.json`, `vocab.txt`, etc.) into the directory `PLM/bert-????-uncased` such as `PLM/bert-base-uncased`.

## 4. Training

```shell script
cd code
./runXXX.sh gpu_id   # like ./run_GAIN_BERT.sh 2
tail -f -n 2000 logs/train_xxx.log
```

## 5. Evaluation

```shell script
cd code
./evalXXX.sh gpu_id threshold(optional)  # like ./eval_GAIN_BERT.sh 0 0.5521
tail -f -n 2000 logs/test_xxx.log
```

PS: we recommend to use threshold = -1 (which is the default, you can omit this arguments at this time) for dev set, 
the log will print the optimal threshold in dev set, and you can use this optimal value as threshold to evaluate test set.

## 6. Submission to LeadBoard (CodaLab)
- You will get json output file for test set at step 5. 

- And then you can rename it as `result.json` and compress it as `result.zip`. 

- At last,  you can submit the `result.zip` to [CodaLab](https://competitions.codalab.org/competitions/20717#participate-submit_results).

## 7. Citation

```
@inproceedings{zeng-EMNLP-2020-GAIN,
 author = {Shuang, Zeng and Runxin, Xu and Baobao, Chang and Lei, Li},
 booktitle = {Proc. of EMNLP},
 title = {Double Graph Based Reasoning for Document-level Relation Extraction},
 year = {2020}
}
```

