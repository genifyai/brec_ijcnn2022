from meantime.models import model_factory
from meantime.options import parse_args
from dotmap import DotMap
from meantime.config import *
from meantime.dataloaders import dataloader_factory
from metrics import *
import pandas as pd
from datetime import date

import sys
from typing import List
import torch
from torch.utils.data import Dataset


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


def main(sys_argv: List[str] = None):
    if sys_argv is None:
        sys_argv = sys.argv[1:]
    conf = parse_args(sys_argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = DotMap(conf, _dynamic=False)
    train_loader, val_loader, test_loader = dataloader_factory(args)
    model = model_factory(args)
    model = model.to(device)
    if args.pretrained_weights is not None:
        model.load(args.pretrained_weights)
    model.eval()

    data_path = "../data"
    ownership = 0  # compute results on products ownership (see blog for reference)
    length_history = 16
    max_len_sequence = args.max_len

    data = np.load(os.path.join(data_path, 'data.npz'))
    np.random.seed(1)
    x_test = data['x_test'][:, -length_history:]
    y_test = data['y_test']
    print(x_test.shape)
    print(y_test.shape)
    print("data loaded from", os.path.join(data_path, 'data.npz'))
    owned_items = None
    if not ownership:
        owned_items = []
        for i in range(x_test.shape[0]):
            owned_items.append(x_test[i][-1][-22:])
    test_set = CustomDataset(x_test, y_test)

    # get list of timestamps and days (required by meantime model)
    df = pd.read_csv('data/santander/ratings.csv')
    df.columns = ['uid', 'sid', 'rating', 'timestamp']
    min_date = date.fromtimestamp(df.timestamp.min())
    df['days'] = df.timestamp.map(lambda t: (date.fromtimestamp(t) - min_date).days)
    month_timestamps = np.sort(df["timestamp"].unique())[:length_history]
    month_days = np.sort(df["days"].unique())[:length_history]

    print("model loaded")

    generator = torch.utils.data.DataLoader(
        test_set, batch_size=1
    )
    j = 0
    n_users = 0
    smap = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 12, 12: 13, 13: 14, 14: 15, 15: 16, 16: 17, 17: 18, 18: 19, 19: 20, 20: 21, 21: 22}
    recommendations_tot, real_recommendations_tot = [], []
    print('making predictions...')
    with torch.no_grad():
        for batch, labels in generator:

            candidates = torch.LongTensor([np.arange(1, 23)]).to(device)
            batch = batch[0]
            history = []
            timestamps = []  # used for "tisas" model
            days = []
            for month in range(length_history):
                idxs = np.where(batch[month][-22:] != 0.)[0]
                for idx in idxs:
                    history += [idx]
                    if args.model_code in ["tisas"]:
                        timestamps += [month + 1]
                    if args.model_code in ["meantime"]:
                        timestamps += [month_timestamps[month]]
                        days += [month_days[month]]
            if len(history) > max_len_sequence:
                history = history[-max_len_sequence:]
                if args.model_code in ["tisas", "meantime"]:
                    timestamps = timestamps[-max_len_sequence:]
                if args.model_code in ["meantime"]:
                    days = days[-max_len_sequence:]
            tokens = [smap[it] for it in history]
            if len(tokens) < max_len_sequence:
                tokens = [0] * (max_len_sequence - len(tokens)) + tokens
                if args.model_code in ["tisas", "meantime"]:
                    timestamps = [0] * (max_len_sequence - len(timestamps)) + timestamps
                if args.model_code in ["meantime"]:
                    days = [0] * (max_len_sequence - len(days)) + days
            tokens = torch.LongTensor(tokens).to(device)
            tokens = torch.unsqueeze(tokens, 0)
            timestamps = torch.LongTensor(timestamps).to(device)
            timestamps = torch.unsqueeze(timestamps, 0)
            days = torch.LongTensor(days).to(device)
            days = torch.unsqueeze(days, 0)
            labels = labels.long().to(device)

            batch = {"tokens": tokens,
                     "candidates": candidates,
                     "labels": labels,
                     "timestamps": timestamps,
                     "days": days}

            result = model(batch)
            scores = result['scores'].detach().cpu().numpy()[0]

            top_items = np.argsort(-scores)
            recommendations = top_items
            real_recommendations = [i for i, p in enumerate(labels[0].detach().cpu().numpy()) if int(float(p)) == 1]

            if owned_items is not None:
                # filter acquisition
                old_items = [i for i, p in enumerate(owned_items[j]) if int(float(p)) == 1]
                real_recommendations = [i for i in real_recommendations if i not in old_items]
                recommendations = [i for i in recommendations if i not in old_items]
            if len(real_recommendations) > 0:
                n_users += 1
            else:
                continue

            # store results
            recommendations_tot.append(recommendations)
            real_recommendations_tot.append(real_recommendations)

            j += 1

    recommendations_tot, real_recommendations_tot = np.array(recommendations_tot), np.array(real_recommendations_tot)

    print("Normal metrics")
    compute_metrics(real_recommendations_tot, recommendations_tot)


if __name__ == "__main__":
    main()
