import os
import torch
from model.transformer import get_model
from torch.utils.data import Dataset
from model.transformer_model import CustomDataset, logits_to_recs
from utils.metrics import *
import argparse

items = ['Current Accounts',
        'Derivada Account',
        'Payroll Account',
        'Junior Account',
        'Mas particular Account',
        'particular Account',
        'particular Plus Account',
        'Short-term deposits',
        'Medium-term deposits',
        'Long-term deposits',
        'e-account',
        'Funds',
        'Mortgage',
        'Pensions 1',
        'Loans',
        'Taxes',
        'Credit Card',
        'Securities',
        'Home Account',
        'Payroll',
        'Pensions 2',
        'Direct Debit']

min_age = 2
max_age = 116
min_antiguedad = 3
max_antiguedad = 256
min_income = 7507.32
max_income = 11900871.51
segmento = ["Individuals", "College graduated", "VIP"]


# Flattens a list of dicts with torch Tensors
def flatten_list_dicts(list_dicts, key):
    if isinstance(list_dicts[0][key], (list, np.ndarray)):
        return np.concatenate([d[key] for d in list_dicts], axis=0)
    return [d[key] for d in list_dicts]


def inverse_scaler(v, vmin, vmax):
    return v * (vmax - vmin) + vmin


# python model/evaluation.py
def evaluate(args):
    # those params should not be changed
    data_path = args.data
    n_items = args.n_items
    d_model = args.d_model
    heads = args.heads
    n_layers = args.n_layers
    length_history = args.seq_len
    weights_path = args.weights_path
    # those params can be changed
    limit_users = args.limit_rows  # int or None if we don't want to limit
    ownership = args.ownership  # if True compute results on products ownership, else compute on acquisition
    no_metadata = args.no_metadata
    no_history = args.no_history
    # end params

    device = args.device
    data = np.load(data_path)
    np.random.seed(1)
    x_test = data['x_test'][:, -length_history:]
    y_test = data['y_test']
    if no_metadata:
        x_test[:, :, :20] = 0
    if no_history:
        x_test[:, :, 20:] = 0
    print(x_test.shape)
    print(y_test.shape)
    print("data loaded from", os.path.join(data_path))
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
    users = []
    n_users = 0
    recommendations_tot, real_recommendations_tot = [], []
    print('evaluation...')
    with torch.no_grad():
        for batch, labels in generator:
            batch, labels = batch.to(device), labels.to(device)
            logits, embedding = model(batch, get_embedding=True)
            recommendations = logits_to_recs(logits.detach().cpu().numpy())
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

            # compute seniority
            seniority = inverse_scaler(x_test[j][0][17], vmin=min_antiguedad, vmax=max_antiguedad)

            # store results
            recommendations_tot.append(recommendations)
            real_recommendations_tot.append(real_recommendations)

            # compute number of owned items in history
            history = x_test[j][:, -22:]
            ownerships = 0

            if j == limit_users:
                break

    recommendations_tot, real_recommendations_tot = np.array(recommendations_tot), np.array(real_recommendations_tot)

    print("Normal metrics")
    compute_metrics(real_recommendations_tot, recommendations_tot)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default="data/data.npz")
    parser.add_argument('--seq_len', type=int, default=16)
    parser.add_argument('--n_items', type=int, default=22,
                        help='number of different items that can be recommended')
    parser.add_argument('--d_model', type=int, default=42,
                        help='dimension of the model')
    parser.add_argument('--heads', type=int, default=7,
                        help='number of Transformer heads')
    parser.add_argument('--n_layers', type=int, default=6,
                        help='number of Transformer layer')
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--weights_path', type=str, default="model/weights",
                        help='path where the model weights are stored')
    parser.add_argument('--ownership', default=False, action='store_true')
    parser.add_argument('--no_metadata', default=False, action='store_true')
    parser.add_argument('--no_history', default=False, action='store_true')
    parser.add_argument('--limit_rows', type=int, default=None,
                        help='if not None limit the size of the dataset')
    args = parser.parse_known_args()[0]
    return args


"""
Some usage examples
Evaluate on acquisition
- python model/evaluation.py
Evaluate on ownership
- python model/evaluation.py --ownership
"""
if __name__ == '__main__':
    args = get_args()
    evaluate(args)
