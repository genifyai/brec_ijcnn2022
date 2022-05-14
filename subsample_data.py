import pandas as pd
import argparse
import numpy as np


if __name__ == '__main__':
    """
    input_file: csv file containing the original Santander product recommendation data.
    - can be downloaded from https://www.kaggle.com/c/santander-product-recommendation/data.
    sample_size: number of users to be sub-sampled from the full dataset (if None use the full data).
    min_data_points: filter users having less than min_data_points records (0 if don't want to filter).
    If 'sample_size' is not None or 'min_data_points' > 0, it saves the sub-sampled dataset:
    - dataset_reduced: sub-sampled dataset containing (sample_size) users,
      each of them having at least 'min_data_points' timestamps.
    USAGE: python subsample_data.py --input_file "data/train.csv" --sample_size 20000 --min_data_points 17
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--sample_size', type=int, default=20000)
    parser.add_argument('--min_data_points', type=float, default=17)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_known_args()[0]
    input_file = args.input_file
    df = pd.read_csv(input_file)
    sample_size = args.sample_size
    min_data_points = args.min_data_points
    assert sample_size > 0, "sample_size param should be > 0"
    assert min_data_points >= 0, "min_data_points param should be >= 0"
    np.random.seed(args.seed)

    # get distinct
    def distinct_timestamps(x):
        if min_data_points and len(x.fecha_dato.unique()) >= min_data_points:
            return 1
        return np.nan


    df.dropna(subset=['ncodpers'], inplace=True)
    df.dropna(subset=['fecha_dato'], inplace=True)
    df.dropna(subset=['cod_prov'], inplace=True)
    df_users = df.groupby('ncodpers').apply(distinct_timestamps)
    df_users = pd.DataFrame({'ncodpers': df_users.index, 'values': df_users.values})
    df_users.dropna(inplace=True)
    if sample_size:
        df_users = df_users.sample(n=sample_size)
    df = df[df.ncodpers.isin(df_users.ncodpers)]
    print("dataset reduced to " + str(df.shape[0]) + " entries")

    if sample_size or min_data_points:
        df.to_csv(input_file.split(".csv")[0] + "_reduced.csv", index=False)
    print("process done")
