# train models using stored data, evaluate them, and save results and trustee 
# graphs
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

from fugu_modules import Model

def input_output_split(df):
    df['in'] = df.iloc[:, :62].apply(list, axis=1) # 62 input features
    input_data = np.array(df['in'].tolist())
    output_data = np.array(df['actual'].tolist()) # 1 output feature
    return input_data, output_data


def load_train_evaluate(test_name, name_1, name_2, csv_name):
    g1 = glob.glob(f'data/{test_name}/{csv_name}_{name_1}*.csv')
    g2 = glob.glob(f'data/{test_name}/{csv_name}_{name_2}*.csv')

    df_1 = pd.read_csv(g1[0])
    df_2 = pd.read_csv(g2[0])

    train_evaluate(df_1, df_2, test_name, name_1, name_2)


def train_evaluate(df_1, df_2, test_name, name_1, name_2):
    print(f"Training and evaluating test {test_name} with {name_1} and {name_2}\n")
    # os.makedirs(f"models/{test_name}", exist_ok=True)
    # os.makedirs(f"evals/{test_name}", exist_ok=True)

    df_1_x, df_1_y = input_output_split(df_1)
    df_2_x, df_2_y = input_output_split(df_2)

    df_1_train_x, df_1_test_x, df_1_train_y, df_1_test_y = train_test_split(
        df_1_x, df_1_y, test_size=0.05, random_state=42
    )
    df_2_train_x, df_2_test_x, df_2_train_y, df_2_test_y = train_test_split(
        df_2_x, df_2_y, test_size=0.05, random_state=42
    )

    # train on df_1
    print(f"Training on {name_1} training data\n")
    model = Model()
    model.train(df_1_train_x, df_1_train_y, df_1_test_x, df_1_test_y,
                model_path=f'models/{test_name}/train_{name_1}.pt')
    # evaluate on df_1
    print(f"Evaluating on {name_1} testing data\n")
    results_filename_1_1 = f"evals/{test_name}/train_{name_1}_eval_{name_1}/"
    model.evaluate(df_1_test_x, df_1_test_y, results_filename_1_1)
    model.evaluate_with_trustee(df_1_test_x, df_1_test_y, results_filename_1_1)
    # evaluate on df_2
    print(f"Evaluating on {name_2} testing data\n")
    results_filename_1_2 = f"evals/{test_name}/train_{name_1}_eval_{name_2}/"
    model.evaluate(df_2_test_x, df_2_test_y, results_filename_1_2)
    model.evaluate_with_trustee(df_2_test_x, df_2_test_y, results_filename_1_2)

    # train on df_2
    print(f"Training on {name_2} training data\n")
    model = Model()
    model.train(df_2_train_x, df_2_train_y, df_2_test_x, df_2_test_y,
                model_path=f'models/{test_name}/train_{name_2}.pt')
    # evaluate on df_1
    print(f"Evaluating on {name_1} testing data\n")
    results_filename_2_1 = f"evals/{test_name}/train_{name_2}_eval_{name_1}/"
    model.evaluate(df_1_test_x, df_1_test_y, results_filename_2_1)
    model.evaluate_with_trustee(df_1_test_x, df_1_test_y, results_filename_2_1)
    # evaluate on df_2
    print(f"Evaluating on {name_2} testing data\n")
    results_filename_2_2 = f"evals/{test_name}/train_{name_2}_eval_{name_2}/"
    model.evaluate(df_2_test_x, df_2_test_y, results_filename_2_2)
    model.evaluate_with_trustee(df_2_test_x, df_2_test_y, results_filename_2_2)


# load_train_evaluate("ascend_descend", "ascending", "descending", "netrep_100ms")
# load_train_evaluate("cwnd_len", "longest", "shortest", "netrep_100ms")
# load_train_evaluate("delivery_rate_len", "longest", "shortest", "netrep_100ms")
# load_train_evaluate("rtt_len", "longest", "shortest", "netrep_100ms")
# load_train_evaluate("trans_time_len", "longest", "shortest", "netrep_100ms")


df_netrep = pd.read_csv('data/netrep_100ms.csv')
df_oct = pd.read_csv('data/oct_100ms.csv')
train_evaluate(df_netrep, df_oct, "compare_netrep_oct", "netrep", "oct")
df_combined = pd.concat([df_netrep, df_oct])

df_combined_x, df_combined_y = input_output_split(df_combined)
train_input, test_input, train_output, test_output = train_test_split(
    df_combined_x, df_combined_y, test_size=0.05, random_state=42
)
model = Model()
model.train(train_input, train_output, test_input, test_output,
            model_path='models/compare_netrep_oct/train_combined.pt'
)

results_fn_combined = "evals/compare_netrep_oct/train_combined_eval_combined/"
mse = model.evaluate(test_input, test_output, results_fn_combined)
model.evaluate_with_trustee(test_input, test_output, results_fn_combined)


# # Training
# df = pd.read_csv( '../netrep_100ms.csv') # from net replica
# df2 = pd.read_csv('../oct_100ms.csv') # from puffer
# # add df2 to df

# # TODO: can add df and df2 together to get a combined dataset (may need to shuffle)
# # df = pd.concat([df, df2])

# input_data, output_data = input_output_split(df)

# model = Model()
# train_input, test_input, train_output, test_output = train_test_split(
#     input_data, output_data, test_size=0.05, random_state=42
# )

# model.train(train_input, train_output, test_input, test_output,
#             model_path='../models/combine_puffer_netrep.pt')


# model.evaluate(test_input, test_output, "evals/eval.txt")

# # evaluate test set with trustee
# model.evaluate_with_trustee(test_input, test_output, "evals/")

# def train_evaluate_compare_netrep_oct():
#     os.makedirs('models/compare_netrep_oct', exist_ok=True)
#     os.makedirs('evals/compare_netrep_oct', exist_ok=True)

#     netrep_df = pd.read_csv('data/netrep_100ms.csv')
#     oct_df = pd.read_csv('data/oct_100ms.csv')

#     netrep_x, netrep_y = input_output_split(netrep_df)
#     oct_x, oct_y = input_output_split(oct_df)

#     netrep_train_x, netrep_test_x, netrep_train_y, netrep_test_y = train_test_split(
#         netrep_x, netrep_y, test_size=0.05, random_state=42
#     )
#     oct_train_x, oct_test_x, oct_train_y, oct_test_y = train_test_split(
#         oct_x, oct_y, test_size=0.05, random_state=42
#     )

#     # train on netrep data
#     model = Model()
#     model.train(netrep_train_x, netrep_train_y, netrep_test_x, netrep_test_y,
#                 model_path='models/compare_netrep_oct/train_netrep.pt')
#     # evaluate on netrep data
#     mse = model.evaluate(netrep_test_x, netrep_test_y, "evals/compare_netrep_oct/train_netrep_eval_netrep.txt")
#     model.evaluate_with_trustee(netrep_test_x, netrep_test_y, "evals/compare_netrep_oct/train_netrep_eval_netrep_trustee")
#     # evaluate on oct data
#     mse = model.evaluate(oct_test_x, oct_test_y, "evals/compare_netrep_oct/train_netrep_eval_oct.txt")
#     model.evaluate_with_trustee(oct_test_x, oct_test_y, "evals/compare_netrep_oct/train_netrep_eval_oct_trustee")

#     # train on oct data
#     model = Model()
#     model.train(oct_train_x, oct_train_y, oct_test_x, oct_test_y,
#                 model_path='models/compare_netrep_oct/train_oct.pt')
#     # evaluate on netrep data
#     mse = model.evaluate(netrep_test_x, netrep_test_y, "evals/compare_netrep_oct/train_oct_eval_netrep.txt")
#     model.evaluate_with_trustee(netrep_test_x, netrep_test_y, "evals/compare_netrep_oct/train_oct_eval_netrep_trustee")
#     # evaluate on oct data
#     mse = model.evaluate(oct_test_x, oct_test_y, "evals/compare_netrep_oct/train_oct_eval_oct.txt")
#     model.evaluate_with_trustee(oct_test_x, oct_test_y, "evals/compare_netrep_oct/train_oct_eval_oct_trustee")


# # train and evaluate the model on ascending and descending data
# def train_evaluate_ascend_descend():
#     os.makedirs('models/ascend_descend', exist_ok=True)
#     os.makedirs('evals/ascend_descend', exist_ok=True)
    
#     ascend3_df = pd.read_csv('data/ascend_descend/netrep_100ms_ascending_3.csv')
#     ascend1_df = pd.read_csv('data/ascend_descend/netrep_100ms_ascending_1.csv')
#     descend1_df = pd.read_csv('data/ascend_descend/netrep_100ms_descending_1.csv')
#     descend3_df = pd.read_csv('data/ascend_descend/netrep_100ms_descending_3.csv')

#     ascend_df = pd.concat([ascend3_df, ascend1_df])
#     descend_df = pd.concat([descend1_df, descend3_df])

#     ascend_x, ascend_y = input_output_split(ascend_df)
#     descend_x, descend_y = input_output_split(descend_df)

#     ascend_train_x, ascend_test_x, ascend_train_y, ascend_test_y = train_test_split(
#         ascend_x, ascend_y, test_size=0.05, random_state=42
#     )
#     descend_train_x, descend_test_x, descend_train_y, descend_test_y = train_test_split(
#         descend_x, descend_y, test_size=0.05, random_state=42
#     )

#     # train on ascending data
#     model = Model()
#     model.train(ascend_train_x, ascend_train_y, ascend_test_x, ascend_test_y,
#                 model_path='models/ascend_descend/train_ascend.pt')
#     # evaluate on ascending data
#     mse = model.evaluate(ascend_test_x, ascend_test_y, "evals/ascend_descend/train_ascend_eval_ascend.txt")
#     model.evaluate_with_trustee(ascend_test_x, ascend_test_y, "evals/ascend_descend/train_ascend_eval_ascend_trustee")
#     # evaluate on descending data
#     mse = model.evaluate(descend_test_x, descend_test_y, "evals/ascend_descend/train_ascend_eval_descend.txt")
#     model.evaluate_with_trustee(descend_test_x, descend_test_y, "evals/ascend_descend/train_ascend_eval_descend_trustee")
    
#     # train on descending data
#     model = Model()
#     model.train(descend_train_x, descend_train_y, descend_test_x, descend_test_y,
#                 model_path='models/ascend_descend/train_descend.pt')
#     # evaluate on ascending data
#     mse = model.evaluate(ascend_test_x, ascend_test_y, "evals/ascend_descend/train_descend_eval_ascend.txt")
#     model.evaluate_with_trustee(ascend_test_x, ascend_test_y, "evals/ascend_descend/train_descend_eval_ascend_trustee")
#     # evaluate on descending data
#     mse = model.evaluate(descend_test_x, descend_test_y, "evals/ascend_descend/train_ascend_eval_descend.txt")
#     model.evaluate_with_trustee(descend_test_x, descend_test_y, "evals/ascend_descend/train_descend_eval_descend_trustee")


# def train_evaluate_long_rtt():
#     os.makedirs('models/rtt_len', exist_ok=True)
#     os.makedirs('evals/rtt_len', exist_ok=True)

#     long_file_list = glob.glob('data/rtt_len/netrep_100ms_long*.csv')
#     short_file_list = glob.glob('data/rtt_len/netrep_100ms_short*.csv')

#     long_df = pd.read_csv(long_file_list[0])
#     short_df = pd.read_csv(short_file_list[0])

#     long_x, long_y = input_output_split(long_df)
#     short_x, short_y = input_output_split(short_df)

#     long_train_x, long_test_x, long_train_y, long_test_y = train_test_split(
#         long_x, long_y, test_size=0.05, random_state=42
#     )
#     short_train_x, short_test_x, short_train_y, short_test_y = train_test_split(
#         short_x, short_y, test_size=0.05, random_state=42
#     )

#     # train on long rtt data
#     model = Model()
#     model.train(long_train_x, long_train_y, long_test_x, long_test_y,
#                 model_path='models/long_rtt/train_long.pt')
#     # evaluate on long rtt data
#     mse = model.evaluate(long_test_x, long_test_y, "evals/long_rtt/train_long_eval_long.txt")
#     model.evaluate_with_trustee(long_test_x, long_test_y, "evals/long_rtt/train_long_eval_long_trustee")
#     # evaluate on short rtt data
#     mse = model.evaluate(short_test_x, short_test_y, "evals/long_rtt/train_long_eval_short.txt")
#     model.evaluate_with_trustee(short_test_x, short_test_y, "evals/long_rtt/train_long_eval_short_trustee")

#     # train on short rtt data
#     model = Model()
#     model.train(short_train_x, short_train_y, short_test_x, short_test_y,
#                 model_path='models/long_rtt/train_short.pt')
#     # evaluate on long rtt data
#     mse = model.evaluate(long_test_x, long_test_y, "evals/long_rtt/train_short_eval_long.txt")
#     model.evaluate_with_trustee(long_test_x, long_test_y, "evals/long_rtt/train_short_eval_long_trustee")
#     # evaluate on short rtt data
#     mse = model.evaluate(short_test_x, short_test_y, "evals/long_r
