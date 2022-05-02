import pandas as pd
import numpy as np
import argparse
import warnings
import csv
from tqdm import tqdm
from model.evaluation import items
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    """
    input_file: csv file containing the train set
    output 2 csv files:
    - joint_probs.csv: items joint probabilities. This computation may take a while.
     (for each x and y for each : probability of x and y owned by two different users)
     Saved in the format "itemX-itemY", p(itemX, itemY)
    - item_probs.csv: items probability (for each x: probability to own x)
     Saved in the format "itemX", p(itemX)
    USAGE: python utils/joint_probs.py --input_file "data/train_reduced.csv"
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    args = parser.parse_known_args()[0]
    input_file = args.input_file

    df = pd.read_csv(input_file)
    users = list(df['ncodpers'].unique())
    # we want for each user a list of acquired items
    raw_items = list(df.columns)[-22:]

    user_items = []
    for user in tqdm(users):
        user_array = []
        for obj in raw_items:
            temp = df[(df['ncodpers'] == user) & (df[obj]==1)]
            if temp.size > 0:
                user_array.append(1)
            else:
                user_array.append(0)
        user_items.append(np.array(user_array))
    user_items = np.stack(user_items, 0)

    items_probs = {}
    for i, obj in enumerate(items):
        items_probs[obj] = np.mean(user_items[:, i])
    # print(items_probs)
    csv_file = open("data/item_probs.csv", "w")
    writer = csv.writer(csv_file)
    for key, value in items_probs.items():
        writer.writerow([key, value])
    csv_file.close()

    joint_prob = {}
    for i, obj1 in tqdm(enumerate(raw_items)):
        for j, obj2 in enumerate(raw_items):
            have_used = 0
            for user in users:
                temp = df[(df['ncodpers'] == user) & (df[obj1]==1) & (df[obj2]==1)]
                if temp.size > 0:
                    have_used += 1
            if j != i:
                name = str(items[i])+"-"+str(items[j])
                if name not in joint_prob.keys():
                    joint_prob[name] = have_used / len(users)
    # print(joint_prob)
    csv_file = open("data/joint_probs.csv", "w")
    writer = csv.writer(csv_file)
    for key, value in joint_prob.items():
        writer.writerow([key, value])
    csv_file.close()

