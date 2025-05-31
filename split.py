# split data into categories to introduce and test bias

import pandas as pd
import numpy as np
import os

# split into ascending and descending based on the following variables:
# delivery_rate
# cwnd
# size
# in_flight?
# inverse trans_time?

# split the data into long and short df's based on the average {group} value
def long_short_split(df, title, group, threshold, max_index=8):
    df[f'avg_{group}'] = df[[f"{group}{i}" for i in range(0, max_index+1)]].mean(axis=1)
    df_long = df[df[f'avg_{group}'] > df[f'avg_{group}'].quantile(threshold)]
    df_short = df[df[f'avg_{group}'] <= df[f'avg_{group}'].quantile(threshold)]
    os.makedirs(f"data/{group}_len", exist_ok=True)
    df_long.to_csv(f"data/{group}_len/{title}_longest_{int(100*(1-threshold))}p_{group}.csv", index=False)
    df_short.to_csv(f"data/{group}_len/{title}_shortest_{int(100*threshold)}p_{group}.csv", index=False)

# return the net number of variabled that are ascending out of 
# delivery_rate, cwnd, and size. +3 means all are ascending, +1 means
# 2 are ascending and 1 is descending, etc
def net_ascending(row):
    net = 0
    
    for var in ['delivery_rate', 'cwnd', 'size']:
        vals = []
        cols = [col for col in row.index if col.startswith(var)]
        for col in cols:
            vals.append(row[col])
        indices = list(range(len(vals)))
        slope, _ = np.polyfit(indices, vals, 1)

        if slope < 0:
            net -= 1
        elif slope > 0:
            net += 1

    return net

# split the data into all categories
for title in ["netrep_100ms", "oct_100ms"]:
    df = pd.read_csv(f"data/{title}.csv")
    
    # ascending-descending split
    ascending = df[df.apply(net_ascending, axis=1) > 0]
    descending = df[df.apply(net_ascending, axis=1) < 0]

    os.makedirs(f"data/ascend_descend", exist_ok=True)
    ascending.to_csv(f"data/ascend_descend/{title}_ascending.csv", index=False)
    descending.to_csv(f"data/ascend_descend/{title}_descending.csv", index=False)

    # split the data into long_rtt and short_rtt based on the rtt value
    long_short_split(df, title, 'rtt', 0.97)

    # split the data into long_trans_time and short_trans_time based on the trans_time value
    long_short_split(df, title, 'trans_time', 0.97, max_index=7)

    # split the data into long_delivery_rate and short_delivery_rate based on the delivery_rate value
    long_short_split(df, title, 'delivery_rate', 0.77)
    
    # split the data into long_cwnd and short_cwnd based on the cwnd value
    long_short_split(df, title, 'cwnd', 0.96)

    # dropping params
    for param in ["delivery_rate", "cwnd", "in_flight", "min_rtt", "rtt", "size", "trans_time"]:
        columns_to_keep = [col for col in df.columns if col.startswith(param)]
        df_dropped = df[columns_to_keep]
        os.makedirs(f"data/dropped_params", exist_ok=True)
        df_dropped.to_csv(f"data/dropped_params/{title}_{param}.csv", index=False)
