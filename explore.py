# explore the data
import pandas as pd
netrep = pd.read_csv('../netrep_100ms.csv')
# # print the first 5 rows
# print(netrep.head())
# # print the columns
# print(netrep.columns)
# # print the shape of the dataframe
# print(netrep.shape)
# # print the data types of the columns
# print(netrep.dtypes)

# look at the statistical distribution of each column
desc = netrep.describe()
for category in ["delivery_rate", "cwnd", "in_flight", "min_rtt", "rtt", "size", "trans_time", "actual"]:
    with open(f'explore/stats_{category}.txt', 'w') as f:
        cat_desc = desc.loc[:, desc.columns.str.find(category) == 0]
        f.write(cat_desc.to_string())

print("unique prediction values: ", netrep["predict"].unique())

with open('explore/results_out.txt', 'w') as f:
    f.write(netrep[['predict', 'actual', 'error']].sort_values('predict', ascending=True).to_string())