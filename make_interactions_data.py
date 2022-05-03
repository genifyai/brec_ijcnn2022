import pandas as pd
import time
import datetime
import argparse


target_cols = ['ind_ahor_fin_ult1','ind_aval_fin_ult1','ind_cco_fin_ult1','ind_cder_fin_ult1','ind_cno_fin_ult1','ind_ctju_fin_ult1','ind_ctma_fin_ult1','ind_ctop_fin_ult1','ind_ctpp_fin_ult1','ind_deco_fin_ult1','ind_deme_fin_ult1','ind_dela_fin_ult1','ind_ecue_fin_ult1','ind_fond_fin_ult1','ind_hip_fin_ult1','ind_plan_fin_ult1','ind_pres_fin_ult1','ind_reca_fin_ult1','ind_tjcr_fin_ult1','ind_valo_fin_ult1','ind_viv_fin_ult1','ind_nomina_ult1','ind_nom_pens_ult1','ind_recibo_ult1']
target_cols = target_cols[2:]


def median_income(df):
    df.loc[df.renta.isnull(), 'renta'] = df.renta.median(skipna=True)
    return df


def make_interactions_data(df, output_interactions_data):
    df_interactions_vector = df[["fecha_dato", "ncodpers"] + target_cols]
    df_interactions_index = pd.DataFrame(columns=["TIMESTAMP", "USER_ID", "ITEM_ID"])

    def get_user_product(row):
        if not hasattr(get_user_product, "count"):
            get_user_product.count = 0
            get_user_product.count_rows = 0
        for i, c in enumerate(df_interactions_vector.columns[2:]):
            if row[c] == 1:
                pass
                timestamp = int(time.mktime(datetime.datetime.strptime(row["fecha_dato"], "%Y-%m-%d").timetuple()))
                df_interactions_index.loc[get_user_product.count] = [timestamp, row["ncodpers"], i]
                get_user_product.count += 1
        get_user_product.count_rows += 1
        print('progress:', round(get_user_product.count_rows/df_interactions_vector.shape[0], 3) * 100, "%")
    df_interactions_vector.apply(get_user_product, axis=1)
    df_interactions_index["ITEM_ID"] = df_interactions_index["ITEM_ID"].astype(int)
    df_interactions_index["USER_ID"] = df_interactions_index["USER_ID"].astype(int)
    df_interactions_index["TIMESTAMP"] = df_interactions_index["TIMESTAMP"].astype(int)
    print(df_interactions_index)
    df_interactions_index.to_csv(output_interactions_data, index=False)


if __name__ == '__main__':
    """
    input_file: csv file containing the original Santander product recommendation data.
    - can be downloaded from https://www.kaggle.com/c/santander-product-recommendation/data.
    It generates the following file.
    - interactions_data.csv: csv file containing users interactions
    USAGE: python make_interactions_data.py --input_file "data/train_reduced.csv"  --output "data/interactions_data.csv"
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_known_args()[0]

    df = pd.read_csv(args.input_file)

    # preprocess
    # provide median income by province
    df = df.groupby('nomprov').apply(median_income)
    df.loc[df.renta.isnull(), "renta"] = df.renta.median(skipna=True)

    # make datasets
    print("making interactions data...")
    make_interactions_data(df, args.output)
    print("process done")
