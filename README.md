# IJCNN 2022: BRec the Bank: Context-aware Self-attentive Encoder for Banking Products Recommendation

This repository contains the original code of the paper *BRec the Bank: Context-aware Self-attentive Encoder for Banking Products Recommendation* presented at the conference [IJCNN 2022](https://wcci2022.org/).

## Dataset

The Santander products recommendation dataset used in our experiments can be downloaded from [here](https://www.kaggle.com/c/santander-product-recommendation/data?select=train_ver2.csv.zip).
For convenience, you can rename the file `train_ver2.csv` into `train.csv` and put it in the folder `data`.

The whole dataset contains hundreds of thousands of users data along a timespan of 17 months. However, we can reduce the dimension of the dataset so to use only a **subsample** of the total users.
```
python subsample_data.py --input_file "data/train.csv" --sample_size 20000 --min_data_points 17
```
`sample_size` is the number of users we want to subsample from the full dataset (input `None` if you want to use the full data), and `min_data_points` is used to filter users having less than `min_data_points` records (ignore it if you don't want to filter).
This process will generate the file `data/train_reduced.csv`.

## Train our Transformer model

The code relative to our model is stored in the directory `model`.
The file `transformer.py` implements the transformer while the file `transformer_model.py` handles the data preprocessing, training and testing.

To begin with, let's preprocess the data.
```
python model/transformer_model.py --save_data --no_load_data --no_train
```
This command creates a file `data.npz` stored in the folder `data` so that next time the preprocessed data can be immediately loaded.

Next, let's train the model for 100 epochs with a warm-up learning rate for the first 10 epochs.
```
python model/transformer_model.py --save_weights --epochs 100 --warmup_epochs 10
```

If we want only to test it, we can simply use the following.
```
python model/transformer_model.py --load_weights --no_train
```

To evaluate the model on the full set of metrics you can run the file `evaluation.py`
```
python model/evaluation.py
```
Default evaluation is on acquisition, but you can evaluate the model on the items ownership task adding the `--ownership` argument.
```
python model/evaluation.py --ownership
```

## Results on items ownership
| Model                      | Prec1  | Prec5 | Prec10 | Rec1 | Rec5 | Rec10 | MRR20 | NDCG20 |
|----------------------------|--------|-------|-------|-------|------|-------|-------|--------|
| Our model                  | **0.9920**| **0.3537**| **0.1875** | **0.7560** | **0.9836** | **0.9990** | **0.9956**| **0.9961**|
| Amazon Personalize         | 0.9622| 0.2825| 0.1664 | 0.7397| 0.8865|0.9571 | 0.9435| 0.9435|
| BERT4Rec                   | 0.9812| 0.3452| 0.1852 | 0.7488| 0.9736|0.9947 | 0.9901| 0.9873|
| SAS                        | 0.9769| 0.3475| 0.1871 | 0.7442| 0.9769|0.9986 | 0.9874| 0.9870|
| TISAS                      | 0.9812| 0.3477| 0.1872 | 0.7472| 0.9773|0.9987 | 0.9897| 0.9888|
| Meantime                   | 0.9801| 0.3389| 0.1846 | 0.7486| 0.9641|0.9931 | 0.9895| 0.9842|

## Results on items acquisition
| Model                      | Prec1  | Prec5 | Prec10 | Rec1 | Rec5 | Rec10 | MRR20 | NDCG20 |
|----------------------------|--------|-------|-------|-------|------|-------|-------|--------|
| Our model                  | **0.9891**| **0.4022**| **0.2157** |**0.6975**|**0.9764**|0.9979| **0.9937**| **0.9941**|
| Xgboost                    | 0.6887| 0.2449| 0.1285 | 0.6062|0.9440|0.9866| 0.8054| 0.8556|
| Amazon Personalize         | 0.4653| 0.1491| 0.0935 |0.4183|0.6396|0.7869| 0.5788| 0.6505|
| BERT4Rec                   | 0.9693| 0.3881| 0.2125 | 0.6823| 0.9581|0.9912 | 0.9830| 0.9796|
| SAS                        | 0.9600| 0.3936| 0.2154 | 0.6736| 0.9665|0.9981 | 0.9781| 0.9782|
| TISAS                      | 0.9599| 0.3937| 0.2155 | 0.6735| 0.9671|**0.9985** | 0.9781| 0.9784|
| Meantime                   | 0.9631| 0.3811| 0.2125 | 0.6792| 0.9485|0.9912 | 0.9791| 0.9724|

**Notes**
- The code for the BERT4Rec, SAS, TISAS, Meantime models has been taken from [(meantime)](https://github.com/SungMinCho/MEANTIME).
- The code for the XGB model has been taken from [Kaggle](https://www.kaggle.com/sudalairajkumar/when-less-is-more).

## Benchmark reproducibility

Adapt the data to make it compatible with the benchmark interface.
```
python make_interactions_data.py --input_file "data/train_reduced.csv"  --output "data/interactions_data.csv"
python preprocess_interactions_data.py
```
Jump into the folder `cd benchmark` and execute the file `benchmark.ipynb`.

## Serendipity, coverage, novelty

Compute items probabilities and joint probabilities. Results will be stored in the `data` folder by default as `csv` files.
```
python utils/joint_probs.py --input_file "data/train_reduced.csv"
```
For each user, get the number of times items have been recommended, the history, predictions and ground truths for different values of `k`.
`k` determines the amount of top recommendations included. Results are also stored in the `data` folder as `csv` files.
```
- python utils/user_history_pred.py --k 1
- python utils/user_history_pred.py --k 3
- python utils/user_history_pred.py --k 5
```
Now we can compute serendipity, coverage, and novelty with the following command.
```
python utils/extra_metrics.py --model_version "final_model"
```

## Cite us

TODO

## Disclaimer

This repo **doesn't include** the pretrained weights of our model, our preprocessed data.
If you need any of the above, you can contact `davide97ls@gmail.com` or `alex@genify.ai` and we will do our best to help you.