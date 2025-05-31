# explore the data
import pandas as pd
import os

# look at the statistical distribution of each column
def explore_dataset(dataset):
    print(f"Exploring {dataset} dataset...")
    df = pd.read_csv(f'../{dataset}.csv')
    desc = df.describe()
    os.makedirs(f'explore/{dataset}', exist_ok=True)
    for category in ["delivery_rate", "cwnd", "in_flight", "min_rtt", "rtt", 
                     "size", "trans_time", "actual"]:
        with open(f'explore/{dataset}/stats_{category}.txt', 'w') as f:
            cat_desc = desc.loc[:, desc.columns.str.find(category) == 0]
            f.write(cat_desc.to_string())

    print(f"unique prediction values for {dataset}: ", df["predict"].unique())

    # with open(f'explore/{dataset}/results_out.txt', 'w') as f:
    #     f.write(df[['predict', 'actual', 'error']].sort_values('predict', ascending=True).to_string())

for dataset in ['netrep_100ms', 'oct_100ms', ]:
    explore_dataset(dataset)
    print(f"Exploration of {dataset} dataset completed.")

for dataset in ['netrep', 'oct']:
    for direction in ['ascending', 'descending']:
        for net in [1, 3]:
            dataset_filename = f"{dataset}_{direction}_{net}"
            print(f"Exploring {dataset_filename} dataset...")
            explore_dataset(dataset_filename)
            print(f"Exploration of {dataset_filename} dataset completed.")