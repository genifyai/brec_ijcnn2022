import copy
import os
import torch
from model.transformer import get_model
from utils.metrics import *
from torch.utils.data import Dataset
from model.transformer_model import CustomDataset, logits_to_recs
import argparse
import csv


def main(args):
    name = args.model_version  # version of the model
    top_k = args.k  # in [1, 3, 5], top recommendations
    data_path = args.data
    n_items = args.n_items
    d_model = args.d_model
    heads = args.heads
    n_layers = args.n_layers
    seq_len = args.seq_len
    weights_path = args.weights_path
    ownership = args.ownership  # if True compute results on products ownership, else compute on acquisition
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = np.load(data_path)
    x_test_original = copy.deepcopy(data['x_test'])
    x_test = data['x_test'][:, -seq_len:]
    y_test = data['y_test']
    print("data loaded from", data_path)
    owned_items = None
    if not ownership:
        owned_items = []
        for i in range(x_test.shape[0]):
            owned_items.append(x_test[i][-1][-22:])

    test_set = CustomDataset(x_test, y_test)

    model = get_model(n_items, d_model, heads, 0,
                      n_layers, 2048, weights_path, device)
    print("model loaded from", weights_path)

    generator = torch.utils.data.DataLoader(
        test_set, batch_size=1
    )
    model.eval()
    j = 0
    history = []
    pred = []
    acquired = []
    item_recommended = {}
    with torch.no_grad():
        for batch, labels in generator:
            batch, labels = batch.to(device), labels.to(device)
            logits = model(batch)
            recommendations = logits_to_recs(logits.detach().cpu().numpy())
            real_recommendations = [i for i, p in enumerate(labels[0].detach().cpu().numpy()) if int(float(p)) == 1]
            if owned_items is not None:
                old_items = [i for i, p in enumerate(owned_items[j]) if int(float(p)) == 1]
                real_recommendations = [i for i in real_recommendations if i not in old_items]
                recommendations = [i for i in recommendations if i not in old_items]
            if len(real_recommendations) == 0:
                j += 1
                continue
            recommendations_k = recommendations[:top_k]
            pred.append(recommendations_k)
            acquired.append(real_recommendations)
            hist = []
            for h, i in enumerate(x_test_original[j][::-1]):
                _items = [it for it, p in enumerate(i[-22:]) if int(float(p)) == 1]
                for it in _items:
                    if it not in hist:
                        hist.append(it)
            history.append(hist)
            for r in recommendations_k:
                if r not in item_recommended.keys():
                    item_recommended[r] = 1
                else:
                    item_recommended[r] += 1
            j += 1

    csv_file = open("data/{}_hist_pred_top_{}.csv".format(name, top_k), "w")
    writer = csv.writer(csv_file)
    for i in range(len(history)):
        writer.writerow([np.array(history[i]), np.array(pred[i]), np.array(acquired[i])])
    csv_file.close()
    print('created file', "data/{}_hist_pred_top_{}.csv".format(name, top_k))

    csv_file = open("data/{}_item_recommended_top_{}.csv".format(name, top_k), "w")
    writer = csv.writer(csv_file)
    for k, it in item_recommended.items():
        writer.writerow([k, it])
    csv_file.close()
    print('created file', "data/{}_item_recommended_top_{}.csv".format(name, top_k))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default="data/data.npz")
    parser.add_argument('--k', type=int, default=1,
                        help="keep only the first k recommendations")
    parser.add_argument('--model_version', type=str, default="final_model",
                        help='identifier of the version of the model')
    parser.add_argument('--seq_len', type=int, default=16)
    parser.add_argument('--n_items', type=int, default=22,
                        help='number of different items that can be recommended')
    parser.add_argument('--d_model', type=int, default=42,
                        help='dimension of the model')
    parser.add_argument('--heads', type=int, default=7,
                        help='number of Transformer heads')
    parser.add_argument('--n_layers', type=int, default=6,
                        help='number of Transformer layer')
    parser.add_argument('--weights_path', type=str, default="model/weights",
                        help='path where the model weights are stored')
    parser.add_argument('--ownership', default=False, action='store_true')
    args = parser.parse_known_args()[0]
    return args


"""
Some usage examples
For each user, get the number of times items have been recommended, the history, predictions and ground truths
- python utils/user_history_pred.py --k 1
- python utils/user_history_pred.py --k 3
- python utils/user_history_pred.py --k 5
"""
if __name__ == '__main__':
    args = get_args()
    main(args)
