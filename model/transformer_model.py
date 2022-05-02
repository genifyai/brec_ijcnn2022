import copy
import os
import torch
from model.transformer import get_model
import time
import torch.nn as nn
from utils.metrics import *
from torch.utils.data import Dataset
from tqdm import tqdm
import pytorch_warmup as warmup
import argparse
import pandas as pd
import math
from sklearn.preprocessing import MinMaxScaler
import numpy as np


class CustomDataset(Dataset):
    def __init__(self, train_x, train_y, nrows=None):
        if nrows is None:
            self.data = [(x, y) for x, y in zip(train_x, train_y)]
        else:
            self.data = [(x, y) for x, y in zip(train_x[:nrows], train_y[:nrows])]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x, y = self.data[index]
        x = torch.FloatTensor(x)
        y = torch.FloatTensor(y)
        return x, y


def clean_data(df):
    def median_income(df):
        df.loc[df.renta.isnull(), 'renta'] = df.renta.median(skipna=True)
        return df
    # provide median income by province
    df = df.groupby('nomprov').apply(median_income)
    df.loc[df.renta.isnull(), "renta"] = df.renta.median(skipna=True)
    # set entries whose "antiguedad" field is missing as minimum seniority
    df.antiguedad = pd.to_numeric(df.antiguedad, errors="coerce")
    df.loc[df.antiguedad.isnull(), "antiguedad"] = df.antiguedad.min()
    df.loc[df.antiguedad < 0, "antiguedad"] = 0
    df["antiguedad"] = df["antiguedad"].astype(int)
    # fix customers age
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["age"].fillna(df["age"].mean(), inplace=True)
    df["age"] = df["age"].astype(int)
    # fill missing field "segmento" with most frequent one
    df.loc[df["segmento"].isnull(), "segmento"] = "03 - UNIVERSITARIO"
    # normalize scalar columns
    scale_cols = ["antiguedad", "age", "renta"]
    for col in scale_cols:
        scaler = MinMaxScaler()
        df[col] = scaler.fit_transform(df[[col]])
    return df


def preprocess(input_file, y_date, seq_len=16, batch_size=32, exclude_date=None, d_model=35):
    """
    Preprocess data and split it in train and test data
    :param d_model:
    :param input_file: string, path to raw dataset, csv file
    :param y_date: string, timestamp use for testing
    :param exclude_date: list[string] timestamps to ignore
    :return: train_x, train_y (both are np.array)
    """
    months_one_hot = [0 for _ in range(12)]
    segmentation_dict = {}
    x_users, y_users = {}, {}
    df = pd.read_csv(input_file)
    df = clean_data(df)
    users = []
    for i, row in df.iterrows():
        if row['fecha_dato'] in exclude_date:
            pass
        user = row['ncodpers']
        date = row['fecha_dato'].split("-")
        year = [int(date[0] == "2016")]  # 1=2016, 0=2015 (1)
        month = copy.copy(months_one_hot)
        month[int(date[1]) - 1] = 1  # months one-hot encoded (12)
        items = list(row.values)[26:]  # items are one-hot encoded (22)
        items = [int(item) if not math.isnan(item) and item != 'NA' else 0 for item in items]
        # one-hot encode segmentation (4)
        segmentation = row['segmento']
        segmentation_array = [0, 0, 0, 0]
        if segmentation not in segmentation_dict.keys():
            segmentation_dict[segmentation] = len(segmentation_dict)
        segmentation_array[segmentation_dict[segmentation]] = 1
        # one-hot encode new-index (1)
        #new_index = [1] if row['ind_nuevo'] == 1 else [0]
        # seniority + age + income (3) - values features
        seniority = float(row['antiguedad'])
        age = float(row['age'])
        income = float(row['renta'])
        value_features = [seniority, age, income]
        # put the data together
        data = year + month + segmentation_array + value_features + items  # (42) values
        if row['fecha_dato'] == y_date and user in x_users.keys():
            y_users[user] = np.array(items)
            users.append(user)
        elif user in x_users.keys():
            x_users[user] = np.vstack((x_users[user], np.array(data)))
        else:
            x_users[user] = np.array(data)

    assert len(x_users) == len(y_users)
    x_data = []
    y_data = []
    for user in users:
        if np.array(x_users[user]).shape[0] == seq_len:
            x_data.append(x_users[user].reshape((seq_len, d_model)))
        else:
            continue
        y_data.append(y_users[user])
    x_data = np.stack(x_data)
    y_data = np.stack(y_data)
    num_users = x_data.shape[0]
    x_data = x_data[:num_users - num_users % batch_size]
    y_data = y_data[:num_users - num_users % batch_size]

    return x_data, y_data


def logits_to_recs(logits):
    logits = np.squeeze(logits)
    recs = np.argsort(logits)[::-1]
    return recs


def train_one_epoch(model, optimizer, criterion, dataset,
                    lr_scheduler, warmup_scheduler, epoch, batch_size=32, device="cpu"):
    generator = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    model.train()
    tot_loss = 0.0
    for batch, labels in tqdm(generator):
        batch, labels = batch.to(device), labels.to(device)
        logits = model(batch)
        lr_scheduler.step(epoch)
        warmup_scheduler.dampen()
        optimizer.zero_grad()
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    tot_loss /= len(dataset) // batch_size
    return tot_loss


def evaluate_one_epoch(model, criterion, dataset, device="cpu", owned_items=None):
    batch_size = 1
    generator = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size
    )
    model.eval()
    tot_loss = 0.0
    tot_prec1, tot_prec3, tot_prec5, tot_prec10 = 0.0, 0.0, 0.0, 0.0
    mrr20, ndcg20 = 0.0, 0.0
    n_users = 0
    j = 0
    with torch.no_grad():
        for batch, labels in tqdm(generator):
            batch, labels = batch.to(device), labels.to(device)
            logits = model(batch)
            loss = criterion(logits, labels)
            tot_loss += loss.item()
            recommendations = logits_to_recs(logits.detach().cpu().numpy())
            real_recommendations = [i for i, p in enumerate(labels[0].detach().cpu().numpy()) if int(float(p)) == 1]
            if owned_items is not None:
                old_items = [i for i, p in enumerate(owned_items[j]) if int(float(p)) == 1]
                real_recommendations = [i for i in real_recommendations if i not in old_items]
                recommendations = [i for i in recommendations if i not in old_items]
            if len(real_recommendations) > 0:
                n_users += 1
            else:
                continue
            tot_prec1 += precision_k(1, real_recommendations, recommendations)
            tot_prec3 += precision_k(3, real_recommendations, recommendations)
            tot_prec5 += precision_k(5, real_recommendations, recommendations)
            tot_prec10 += precision_k(10, real_recommendations, recommendations)
            mrr20 += mrr_k(20, real_recommendations, recommendations)
            ndcg20 += ndcg_k(20, real_recommendations, recommendations)
        tot_loss /= len(dataset) // batch_size
        tot_prec1 /= n_users
        tot_prec3 /= n_users
        tot_prec5 /= n_users
        tot_prec10 /= n_users
        mrr20 /= n_users
        ndcg20 /= n_users
        metrics_dict = {"prec1": tot_prec1, "prec3": tot_prec3, "prec5": tot_prec5,
                        "prec10": tot_prec10, "mrr20": mrr20, "ndcg20": ndcg20}
    return tot_loss, metrics_dict


def train_pipeline(args):

    # use preprocessed data from npz file
    if not args.no_load_data and os.path.exists(os.path.join(args.data_path, 'data.npz')):
        data = np.load(os.path.join(args.data_path, 'data.npz'))
        x_train = data['x_train']
        x_test = data['x_test']
        y_train = data['y_train']
        y_test = data['y_test']
        print("data loaded from", os.path.join(args.data_path, 'data.npz'))
    # preprocess data from csv file
    else:
        x_train, y_train = preprocess(args.dataset, y_date="2016-04-28", exclude_date=["2016-05-28"],
                                      seq_len=args.seq_len, batch_size=args.batch_size, d_model=args.d_model)
        x_test, y_test = preprocess(args.dataset, y_date="2016-05-28", exclude_date=["2015-01-28"],
                                    seq_len=args.seq_len, batch_size=args.batch_size, d_model=args.d_model)
    print("train set x:", x_train.shape)
    print("train set y:", y_train.shape)
    print("test set x:", x_test.shape)
    print("test set y:", y_test.shape)

    owned_items = []
    for i in range(x_test.shape[0]):
        owned_items.append(x_test[i][-1][-22:])

    if args.save_data:
        np.savez(os.path.join(args.data_path, 'data.npz'),
                 x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
        print("data saved at", os.path.join(args.data_path, 'data.npz'))

    train_set = CustomDataset(x_train, y_train, nrows=args.limit_rows)
    test_set = CustomDataset(x_test, y_test, nrows=args.limit_rows)

    if args.load_weights:
        model = get_model(args.n_items, args.d_model, args.heads, args.dropout,
                          args.n_layers, args.hidden_size, args.weights_path, args.device)
        print("model loaded from", args.weights_path)
    else:
        model = get_model(args.n_items, args.d_model, args.heads, args.dropout,
                          args.n_layers, args.hidden_size, None, args.device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99), eps=1e-9)
    # warmup lr
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.warmup_epochs], gamma=0.1)
    if args.warmup_type == 'linear':
        warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)
    elif args.warmup_type == 'exponential':
        warmup_scheduler = warmup.UntunedExponentialWarmup(optimizer)
    elif args.warmup_type == 'radam':
        warmup_scheduler = warmup.RAdamWarmup(optimizer)
    else:
        warmup_scheduler = warmup.LinearWarmup(optimizer, 1)
    warmup_scheduler.last_step = -1  # initialize the step counter

    best_model = copy.deepcopy(model.state_dict())
    best_test_loss = np.inf

    start = time.time()
    if not args.no_train:
        print("training...")
        for epoch in range(args.epochs):

            train_loss = train_one_epoch(model, optimizer, criterion, train_set,
                                         lr_scheduler, warmup_scheduler, epoch, args.batch_size, args.device)
            print("epoch {} | train loss: {}".format(epoch + 1, train_loss))

            if args.test_in_train:
                test_loss, _ = evaluate_one_epoch(model, criterion, test_set, args.device)
                print("epoch {} | test loss: {}".format(epoch + 1, test_loss))
                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                    best_model = copy.deepcopy(model.state_dict())

            if args.save_weights_epoch is not None and epoch % args.save_weights_epoch == 0:
                torch.save(model.state_dict(), os.path.join(args.weights_path, "weights_{}.pth".format(epoch)))
                print("model saved at", os.path.join(args.weights_path, "weights_{}.pth".format(epoch)))

        print("finished training in", time.time() - start)
    if args.test_in_train:
        model.load_state_dict(best_model)
        print("restored best model")
    print("testing...")
    print("--ownership results--")
    test_loss, test_metrics = evaluate_one_epoch(model, criterion, test_set, args.device, owned_items=None)
    print("Test loss:", test_loss)
    print_metrics_dict(test_metrics)
    print("--acquisition results--")
    test_loss, test_metrics = evaluate_one_epoch(model, criterion, test_set, args.device, owned_items)
    print("Test loss:", test_loss)
    print_metrics_dict(test_metrics)
    if args.save_weights > 0:
        torch.save(model.state_dict(), os.path.join(args.weights_path, "weights.pth"))
        print("model saved at", os.path.join(args.weights_path, "weights.pth"))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="data/train_reduced.csv")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seq_len', type=int, default=16)
    parser.add_argument('--n_items', type=int, default=22,
                        help='number of different items that can be recommended')
    parser.add_argument('--save_data', default=False, action='store_true',
                        help='if True save preprocessed data in npz format')
    parser.add_argument('--no_load_data', default=False, action='store_true',
                        help='if True skip data preprocessing and reuse npz data preprocessed if available')
    parser.add_argument('--data_path', type=str, default="data",
                        help='path to the data')
    parser.add_argument('--no_train', default=False, action='store_true',
                        help='if True skip training')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--warmup_type', type=str, default="radam",
                        help='choose from "linear", "exponential", "radam", "none"')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--d_model', type=int, default=42,
                        help='dimension of the model')
    parser.add_argument('--heads', type=int, default=7,
                        help='number of Transformer heads')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--n_layers', type=int, default=6,
                        help='number of Transformer layer')
    parser.add_argument('--load_weights', default=False, action='store_true')
    parser.add_argument('--test_in_train', default=False, action='store_true',
                        help='if True test the model after each epoch during training')
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--save_weights', default=False, action='store_true')
    parser.add_argument('--weights_path', type=str, default="model/weights",
                        help='path where to store the model weights')
    parser.add_argument('--limit_rows', type=int, default=None,
                        help='if not None limit the size of the dataset')
    parser.add_argument('--hidden_size', type=int, default=2048,
                        help='hidden size of the encoders forward layer')
    parser.add_argument('--save_weights_epoch', type=int, default=None,
                        help='during training save a copy of the model weights every "save_weights_epoch" epochs')
    args = parser.parse_known_args()[0]
    return args


"""
Some usage examples
use a small dataset to make sure the code can run:
- python model/transformer_model.py --limit_rows 100 --epochs 10 --warmup_epochs 2
process the data without training
- python model/transformer_model.py --save_data --no_load_data --no_train --dataset "data/train_reduced.csv"
train the model for 100 epochs with 10 warmup epochs and save final weights
- python model/transformer_model.py --save_weights --epochs 100 --warmup_epochs 10
load pretrained weights and test without training
- python model/transformer_model.py --load_weights --no_train
"""
if __name__ == '__main__':
    args = get_args()
    train_pipeline(args)
