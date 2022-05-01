import os
import torch
from transformer import get_model
from utils.metrics import *
from torch.utils.data import Dataset
from transformer_model import CustomDataset, logits_to_recs
from sklearn.cluster import KMeans
import numpy as np
from utils.clustering import *
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



# python genify_recosys/evaluation.py
if __name__ == '__main__':
    # those params should not be changed
    data_path = "data"
    n_items = 22
    d_model = 42
    heads = 7
    n_layers = 6
    length_history = 16
    weights_path = "genify_recosys/weights/"
    # those params can be changed
    limit_users = None  # int or None if we don't want to limit
    ownership = 0  # compute results on products ownership (see blog for reference)
    no_metadata = 0
    no_history = 0
    seniority_metrics = 0
    ownership_metrics = 0
    ownership_threshold = 1  # between 1-16 (used to compute metrics on users having x products for y months in history)
    mask_current_account = False
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
    if mask_current_account:
        x_test[:, :, 0] = 0
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
            embedding = embedding.detach().cpu().numpy().squeeze()
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

            for i, item in enumerate(items):
                item_count = 0
                for h in history:
                    if h[i] == 1:
                        item_count += 1
                if item_count >= ownership_threshold:
                    ownerships += 1

            users.append(dict(
                              ownerships=ownerships,
                              seniority=seniority
                             )
                         )
            j += 1
            if j == limit_users:
                break

    recommendations_tot, real_recommendations_tot = np.array(recommendations_tot), np.array(real_recommendations_tot)

    print("Normal metrics")
    compute_metrics(real_recommendations_tot, recommendations_tot)

    if seniority_metrics:
        seniorities = flatten_list_dicts(users, "seniority")
        p1, p3, p5, p10 = [], [], [], []
        x = []
        for i in range(10):
            seniority_threshold = max_antiguedad*(i+1)/10
            idxs = np.argwhere(seniorities <= np.full_like(seniorities, seniority_threshold))
            print("Metrics new users: seniority <=", seniority_threshold)
            idxs = idxs.flatten()
            print("Total users:", len(idxs))
            a = np.array([x for x in range(200)])
            result = compute_metrics(real_recommendations_tot[idxs], recommendations_tot[idxs])
            p1.append(result["prec1"])
            p3.append(result["prec3"])
            p5.append(result["prec5"])
            p10.append(result["prec10"])
            x.append(seniority_threshold)
        fig, ax = plt.subplots()
        # ax.plot(x, p1)
        ax.plot(x, p3, marker='D')
        ax.plot(x, p5, marker='D')
        ax.plot(x, p10, marker='D')
        ax.set_xlabel('Seniority')
        ax.set_ylabel('Precision@K')
        ax.legend(['Precision3', 'Precision5', 'Precision10'], loc='upper right')

    if ownership_metrics:
        ownerships = flatten_list_dicts(users, "ownerships")
        p1, p3, p5, p10 = [], [], [], []
        x = []
        for i in range(0, 8):
            print("Metrics on users with {} owned items for at least {} months ----------------------".format(i, ownership_threshold))
            idxs = np.argwhere(ownerships == np.full_like(ownerships, i)).flatten()
            print("Number of users:", len(idxs))
            if len(idxs) == 0:
                continue
            result = compute_metrics(real_recommendations_tot[idxs], recommendations_tot[idxs])
            p1.append(result["prec1"])
            p3.append(result["prec3"])
            p5.append(result["prec5"])
            p10.append(result["prec10"])
            x.append(i)
        fig, ax = plt.subplots()
        #ax.plot(x, p1)
        ax.plot(x, p3, marker='D')
        ax.plot(x, p5, marker='D')
        ax.plot(x, p10, marker='D')
        ax.set_xlabel('Number of items owned')
        ax.set_ylabel('Precision@K')
        ax.legend(['Precision3', 'Precision5', 'Precision10'], loc='upper right')
        #ax.legend(['Precision3'])
    plt.show()
