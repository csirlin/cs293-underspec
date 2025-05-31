# train models using stored data, evaluate them, and save results and trustee 
# graphs
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

import config
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


if config.TEST_ASCEND_DESCEND:
    load_train_evaluate("ascend_descend", "ascending", "descending", "netrep_100ms")

if config.TEST_CWND_LEN:
    load_train_evaluate("cwnd_len", "longest", "shortest", "netrep_100ms")

if config.TEST_DELIVERY_RATE_LEN:
    load_train_evaluate("delivery_rate_len", "longest", "shortest", "netrep_100ms")

if config.TEST_RTT_LEN:
    load_train_evaluate("rtt_len", "longest", "shortest", "netrep_100ms")

if config.TEST_TRANS_TIME_LEN:
    load_train_evaluate("trans_time_len", "longest", "shortest", "netrep_100ms")

if config.TEST_NETREP_OCT:
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
