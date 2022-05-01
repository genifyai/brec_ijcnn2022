import os
import torch
from transformer import get_model
from torch.utils.data import Dataset
from transformer_model import CustomDataset, logits_to_recs
from utils.metrics import *

items = ['Current Accounts',
        'Derivada Account',
        'Payroll Account',
        'Junior Account',
        'Más particular Account',
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
if __name__ == '__main__':
    # those params should not be changed
    data_path = "data"
    n_items = 22
    d_model = 42
    heads = 7
    n_layers = 6
    length_history = 16
    weights_path = "model/weights/"
    # those params can be changed
    limit_users = None  # int or None if we don't want to limit
    ownership = False  # if True compute results on products ownership
    no_metadata = False
    no_history = False
    # end params

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = np.load(os.path.join(data_path, 'data.npz'))
    np.random.seed(1)
    x_test = data['x_test'][:, -length_history:]
    y_test = data['y_test']
    if no_metadata:
        x_test[:, :, :20] = 0
    if no_history:
        x_test[:, :, 20:] = 0
    print(x_test.shape)
    print(y_test.shape)
    print("data loaded from", os.path.join(data_path, 'data.npz'))
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
                j += 1
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
