import sys
import pandas as pd
import numpy as np
import pickle
import argparse
#model side imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch
import pickle
import numpy as np
from os import path
from sklearn.metrics import mean_squared_error
import pandas as pd
from trustee import ClassificationTrustee
import graphviz
from sklearn import tree
import os
from datetime import datetime
# Constants
BATCH_SIZE = 32
# NUM_EPOCHS = 500
# TODO: Check this number of epochs is good 
NUM_EPOCHS = 10 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

inference = True



VIDEO_DURATION = 180180
PKT_BYTES = 1500
MILLION = 1000000
PAST_CHUNKS = 8
FUTURE_CHUNKS = 5

TUNING = True
CHECKPOINT = 100

# class WrappedModel:
#     def __init__(self, model):
#         self.model = model
#     def predict(self, x):
#         print("x is ")
#         print(x)
#         if isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
#             x = x.values
#             print("x is now x.values")
#             print(x.shape, "\n", x)
#         if isinstance(x, np.ndarray):
#             x = torch.tensor(x, dtype=torch.float64)
#             print("x is now tensor")
#             print(x.shape, "\n", x)
#         if isinstance(x, torch.Tensor):
#             x = x.to(dtype=torch.float64)
#         self.model.eval()
#         with torch.no_grad():
#             res = self.model(x) 
#             print("results before transformation:")
#             print(res.shape, "\n", res)
#             res = res.detach().cpu().numpy().squeeze()
#         print("results after:")
#         print(res.shape, "\n", res)
#         return res

class Model:
    #Model constants
    PAST_CHUNKS = 8
    FUTURE_CHUNKS = 5
    DIM_IN = 62
    COLUMNS = [j + str(i) for i in range(PAST_CHUNKS + 1) for j in ['delivery_rate', 'cwnd', 'in_flight', 'min_rtt', 'rtt', 'size', 'trans_time']][:DIM_IN]
    DIM_OUT = 21  # BIN_MAX + 1
    DIM_H1 = 64
    DIM_H2 = 64
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    BIN_SIZE = 0.5  # seconds
    BIN_MAX = 20

    def __init__(self):
        self.model = torch.nn.Sequential(
            torch.nn.Linear(Model.DIM_IN, Model.DIM_H1),
            torch.nn.ReLU(),
            torch.nn.Linear(Model.DIM_H1, Model.DIM_H2),
            torch.nn.ReLU(),
            torch.nn.Linear(Model.DIM_H2, Model.DIM_OUT),
        ).double().to(device=DEVICE)
        self.loss_fn = torch.nn.CrossEntropyLoss().to(device=DEVICE)
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=Model.LEARNING_RATE,
                                          weight_decay=Model.WEIGHT_DECAY)
        self.obs_size = None
        self.obs_mean = None
        self.obs_std = None

    def update_obs_stats(self, raw_in):
        if self.obs_size is None:
            self.obs_size = len(raw_in)
            self.obs_mean = np.mean(raw_in, axis=0)
            self.obs_std = np.std(raw_in, axis=0)
            return
        old_size = self.obs_size
        new_size = len(raw_in)
        self.obs_size = old_size + new_size
        old_mean = self.obs_mean
        new_mean = np.mean(raw_in, axis=0)
        self.obs_mean = (old_mean * old_size + new_mean * new_size) / self.obs_size
        old_std = self.obs_std
        old_sum_square = old_size * (np.square(old_std) + np.square(old_mean))
        new_sum_square = np.sum(np.square(raw_in), axis=0)
        mean_square = (old_sum_square + new_sum_square) / self.obs_size
        self.obs_std = np.sqrt(mean_square - np.square(self.obs_mean))

    def normalize_input(self, raw_in, update_obs=False):
        z = np.array(raw_in)
        if update_obs:
            self.update_obs_stats(z)
        assert self.obs_size is not None
        for col in range(len(self.obs_mean)):
            z[:, col] -= self.obs_mean[col]
            if self.obs_std[col] != 0:
                z[:, col] /= self.obs_std[col]
        return z

    def discretize_output(self, raw_out):
        z = np.array(raw_out)
        z = np.floor((z + 0.5 * Model.BIN_SIZE) / Model.BIN_SIZE).astype(int)
        return np.clip(z, 0, Model.BIN_MAX)

    def train(self, train_input, train_output, test_input, test_output, model_path):
        train_input = torch.from_numpy(self.normalize_input(train_input, update_obs=inference)).to(DEVICE)
        train_output = torch.from_numpy(self.discretize_output(train_output)).to(DEVICE)
        test_input = torch.from_numpy(self.normalize_input(test_input, update_obs=False)).to(DEVICE)
        test_output = torch.from_numpy(self.discretize_output(test_output)).to(DEVICE)

        for epoch in range(NUM_EPOCHS):
            self.model.train()
            perm = np.random.permutation(len(train_input))
            train_input = train_input[perm]
            train_output = train_output[perm]

            num_batches = int(np.ceil(len(train_input) / BATCH_SIZE))
            epoch_loss = 0

            for i in range(num_batches):
                start_idx = i * BATCH_SIZE
                end_idx = min((i + 1) * BATCH_SIZE, len(train_input))

                batch_input = train_input[start_idx:end_idx]
                batch_output = train_output[start_idx:end_idx]

                # Forward pass
                predictions = self.model(batch_input)
                loss = self.loss_fn(predictions, batch_output)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Loss: {epoch_loss / num_batches}")

            # Evaluate after each epoch
            self.evaluate(test_input, test_output, model_path)
        self.save(model_path)
            
    def retrain(self, train_input, train_output, test_input, test_output, input_model_path, model_path):
        # Load the existing model from input_model_path if it exists
        if os.path.exists(input_model_path):
            print(f"Loading model from {input_model_path}")
            self.model.load_state_dict(torch.load(input_model_path))
            self.model.train()  # Set the model to training mode
        else:
            print("No existing model found at input_model_path, starting training from scratch.")
        
        # Optionally load optimizer state if you want to resume optimizer state
        if os.path.exists(input_model_path + "_optimizer"):
            self.optimizer.load_state_dict(torch.load(input_model_path + "_optimizer"))
        
        # Prepare inputs and outputs
        train_input = torch.from_numpy(self.normalize_input(train_input, update_obs=inference)).to(DEVICE)
        train_output = torch.from_numpy(self.discretize_output(train_output)).to(DEVICE)
        test_input = torch.from_numpy(self.normalize_input(test_input, update_obs=False)).to(DEVICE)
        test_output = torch.from_numpy(self.discretize_output(test_output)).to(DEVICE)

        # Training loop
        for epoch in range(NUM_EPOCHS):
            self.model.train()
            perm = np.random.permutation(len(train_input))
            train_input = train_input[perm]
            train_output = train_output[perm]

            num_batches = int(np.ceil(len(train_input) / BATCH_SIZE))
            epoch_loss = 0

            for i in range(num_batches):
                start_idx = i * BATCH_SIZE
                end_idx = min((i + 1) * BATCH_SIZE, len(train_input))

                batch_input = train_input[start_idx:end_idx]
                batch_output = train_output[start_idx:end_idx]

                # Forward pass
                predictions = self.model(batch_input)
                loss = self.loss_fn(predictions, batch_output)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Loss: {epoch_loss / num_batches}")

            # Evaluate after each epoch
            self.evaluate(test_input, test_output, model_path)
        
        # Save the model and optimizer state after training
        torch.save(self.model.state_dict(), model_path)
        torch.save(self.optimizer.state_dict(), model_path + "_optimizer")
           



    def train1(i, args, model, input_data, output_data):
        if TUNING:
            # permutate input and output data before splitting
            perm_indices = np.random.permutation(len(input_data))
            input_data = input_data[perm_indices]
            output_data = output_data[perm_indices]

            # split training data into training/validation
            num_training = int(0.8 * len(input_data))
            train_input = input_data[:num_training]
            train_output = output_data[:num_training]
            validate_input = input_data[num_training:]
            validate_output = output_data[num_training:]
            sys.stderr.write('[{}] training set size: {}\n'
                            .format(i, len(train_input)))
            sys.stderr.write('[{}] validation set size: {}\n'
                            .format(i, len(validate_input)))

            validate_losses = []
        else:
            num_training = len(input_data)
            sys.stderr.write('[{}] training set size: {}\n'
                            .format(i, num_training))

        train_losses = []

        # number of batches
        num_batches = int(np.ceil(num_training / BATCH_SIZE))
        sys.stderr.write('[{}] total epochs: {}\n'.format(i, NUM_EPOCHS))

        # loop over the entire dataset multiple times
        for epoch_id in range(1, 1 + NUM_EPOCHS):
            # permutate data in each epoch
            perm_indices = np.random.permutation(num_training)

            running_loss = 0
            for batch_id in range(num_batches):
                start = batch_id * BATCH_SIZE
                end = min(start + BATCH_SIZE, num_training)
                batch_indices = perm_indices[start:end]

                # get a batch of input data
                batch_input = input_data[batch_indices]
                batch_output = output_data[batch_indices]

                running_loss += model.train_step(batch_input, batch_output)
            running_loss /= num_batches

            # print info
            if TUNING:
                train_loss = model.compute_loss(train_input, train_output)
                validate_loss = model.compute_loss(validate_input, validate_output)
                train_losses.append(train_loss)
                validate_losses.append(validate_loss)

                train_accuracy = 100 * model.compute_accuracy(
                        train_input, train_output)
                validate_accuracy = 100 * model.compute_accuracy(
                        validate_input, validate_output)

                sys.stderr.write('[{}] epoch {}:\n'
                                '\ttraining: loss {:.3f}, accuracy {:.2f}%\n'
                                '\tvalidation: loss {:.3f}, accuracy {:.2f}%\n'
                                .format(i, epoch_id,
                                        train_loss, train_accuracy,
                                        validate_loss, validate_accuracy))
            else:
                train_losses.append(running_loss)
                sys.stderr.write('[{}] epoch {}: training loss {:.3f}\n'
                                .format(i, epoch_id, running_loss))

            # save checkpoints or the final model
            if epoch_id % CHECKPOINT == 0 or epoch_id == NUM_EPOCHS:
                if epoch_id == NUM_EPOCHS:
                    suffix = ''
                else:
                    suffix = '-checkpoint-{}'.format(epoch_id)

                model_path = path.join(args.save_model,
                                    'py-{}{}.pt'.format(i, suffix))
                model.save(model_path)
                sys.stderr.write('[{}] Saved model for Python to {}\n'
                                .format(i, model_path))

                model_path = path.join(args.save_model,
                                    'cpp-{}{}.pt'.format(i, suffix))
                meta_path = path.join(args.save_model,
                                    'cpp-meta-{}{}.json'.format(i, suffix))
                model.save_cpp_model(model_path, meta_path)
                sys.stderr.write('[{}] Saved model for C++ to {} and {}\n'
                                .format(i, model_path, meta_path))

                # plot losses
                losses = {}
                losses['train'] = train_losses
                if TUNING:
                    losses['validate'] = validate_losses

                loss_path = path.join(args.save_model,
                                    'loss{}{}.png'.format(i, suffix))
                plot_loss(losses, loss_path)

    def load(self, model_path):
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        self.obs_size = checkpoint['obs_size']
        self.obs_mean = checkpoint['obs_mean']
        self.obs_std = checkpoint['obs_std']

    def save(self, model_path):
        assert (self.obs_size is not None)

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'obs_size': self.obs_size,
            'obs_mean': self.obs_mean,
            'obs_std': self.obs_std,
        }, model_path)

    def predict(self, x):
        with torch.no_grad():
            x = x.to_numpy()
            x = self.normalize_input(x, update_obs=inference)
            x = torch.from_numpy(x).to(DEVICE)
            y_scores = self.model(x)
            y_predicted = torch.max(y_scores, 1)[1].to(device=DEVICE)
            ret = y_predicted.detach().cpu().numpy()
            return ret
    def predict_discrete(self, x):
        with torch.no_grad():
            x = x.to_numpy()
            x = self.normalize_input(x, update_obs=inference)
            x = torch.from_numpy(x).to(DEVICE)
            y_scores = self.model(x)
            y_predicted = torch.max(y_scores, 1)[1].to(device=DEVICE)
            ret = y_predicted.detach().cpu().numpy()
            return y_scores, ret

    def predict_cont(self, x):
        with torch.no_grad():
            x = x.to_numpy()
            x = self.normalize_input(x, update_obs=inference)
            x = torch.from_numpy(x).to(DEVICE)
            y_scores = self.model(x)
            y_predicted = torch.max(y_scores, 1)[1].to(device=DEVICE)
            ret = y_predicted.double().cpu().numpy()
            for i in range(len(ret)):
                bin_id = ret[i]
                if bin_id == 0:  # the first bin is defined differently
                    ret[i] = 0.25 * Model.BIN_SIZE
                else:
                    ret[i] = bin_id * Model.BIN_SIZE
            return ret

    def evaluate_with_trustee(self, test_input, test_output, output_loc):
        self.model.eval()
        pd_input = pd.DataFrame(test_input, columns=self.COLUMNS)
        test_output_discretized = self.discretize_output(test_output)
        with torch.no_grad():
            predictions_prob, class_preds = self.predict_discrete(pd_input)
            target = torch.from_numpy(test_output_discretized).to(predictions_prob.device)
            cross_entropy_loss = self.loss_fn(predictions_prob, target).item()

            # Cross-Entropy Loss
            # TODO: Did I do the right thing here? in the above line
            # cross_entropy_loss = self.loss_fn(predictions_prob, torch.from_numpy(test_output_discretized)).item()
            # Print metrics
            print(f"Test Cross-Entropy Loss: {cross_entropy_loss}")
            print("Classification Report:")
            print(classification_report(test_output_discretized, class_preds, zero_division=0))
            
            trustee = ClassificationTrustee(expert=self)
            trustee.fit(pd_input, test_output_discretized, num_iter=10, 
                        num_stability_iter=2, samples_size=0.3, verbose=True, 
                        predict_method_name="predict")
            dt, pruned_dt, agreement, reward = trustee.explain()
            dt_y_pred = dt.predict(pd_input)
            print("Model explanation global fidelity report:")
            print(classification_report(class_preds, dt_y_pred))
            dot_data = tree.export_graphviz(pruned_dt, 
                                            class_names=[str(i)for i in range(21)], 
                                            feature_names=self.COLUMNS,
                                            filled=True,rounded=True,
                                            special_characters=True)
            graph = graphviz.Source(dot_data)
            fil = graph.render(output_loc + f"trustee_{datetime.now()}", format="png")

    def evaluate(self, test_input, test_output, save_loc):
        self.model.eval()
        pd_input = pd.DataFrame(test_input, columns=self.COLUMNS)
        # pd_input = pd.DataFrame(test_input.cpu().numpy(), columns=self.COLUMNS)

        cnt = 0
        with torch.no_grad():
            predictions = self.predict_cont(pd_input)
            # Print metrics
            sum = 0
            print("Length of predictions: ", len(predictions))
            print("Length of actual: ", len(test_output))
            # calculate the diffrerence between the predictions and the actual values and save it in a list diffs
            # diffs = []
            to_save = []
            for i in range(len(predictions)):
                trans_times = [0]* 10
                for j in range (8):
                    trans_times[j] = pd_input .iloc[i][f'trans_time{j}'] 
                cnt +=1 
                if cnt % 1000 == 0:
                    print(cnt)        
                trans_times[8] = predictions[i]
                trans_times[9] = test_output[i]
                to_save.append(trans_times)
                # diffs.append(predictions[i] - test_output[i])
                # sum += abs(diffs[i])
            # save the to save list to a csv file
            pd.DataFrame(to_save).to_csv(save_loc, index=False)
            # save diffs to a csv file
            # pd.DataFrame(diffs).to_csv('/mnt/md0/jaber/puffer_trustee/ttpABR_14days/diffs.csv', index=False)
            # Mean Squared Error
            mse_loss = mean_squared_error(test_output, predictions)
            # mse_loss = mean_squared_error(test_output.cpu().numpy(), predictions.cpu().numpy())

            print(f"Test Mean Squared Error: {mse_loss}")


def prepare_raw_data(video_sent_path, video_acked_path, time_start=None, time_end=None):
    """
    Load data from files and calculate chunk transmission times.
    """
    video_sent_df = pd.read_csv(video_sent_path)
    video_acked_df = pd.read_csv(video_acked_path)

    # Rename "time (ns GMT)" to "time" for convenience
    video_sent_df.rename(columns={'time (ns GMT)': 'time'}, inplace=True)

    video_acked_df.rename(columns={'time (ns GMT)': 'time'}, inplace=True)

    # Convert nanosecond timestamps to datetime
    video_sent_df['time'] = pd.to_datetime(video_sent_df['time'], unit='ns')
    video_acked_df['time'] = pd.to_datetime(video_acked_df['time'], unit='ns')

    # Filter by time range
    if time_start:
        time_start = pd.to_datetime(time_start)
        video_sent_df = video_sent_df[video_sent_df['time'] >= time_start]
        video_acked_df = video_acked_df[video_acked_df['time'] >= time_start]
    if time_end:
        time_end = pd.to_datetime(time_end)
        video_sent_df = video_sent_df[video_sent_df['time'] <= time_end]
        video_acked_df = video_acked_df[video_acked_df['time'] <= time_end]

    # Process the data
    return calculate_trans_times(video_sent_df, video_acked_df)


def calculate_trans_times(video_sent_df, video_acked_df):
    """
    Calculate transmission times from video_sent and video_acked datasets using session_id.
    """
    d = {}
    last_video_ts = {}

    for _, row in video_sent_df.iterrows():
        session = row['session_id']  # Use only session_id to track sessions
        if session not in d:
            d[session] = {}
            last_video_ts[session] = None

        video_ts = int(row['video_ts'])
        if last_video_ts[session] is not None:
            if video_ts != last_video_ts[session] + VIDEO_DURATION:
                continue

        last_video_ts[session] = video_ts
        d[session][video_ts] = {
            'sent_ts': pd.Timestamp(row['time']),
            'size': float(row['size']) / PKT_BYTES,
            'delivery_rate': float(row['delivery_rate']) / PKT_BYTES,
            'cwnd': float(row['cwnd']),
            'in_flight': float(row['in_flight']),
            'min_rtt': float(row['min_rtt']) / MILLION,
            'rtt': float(row['rtt']) / MILLION,
        }

    for _, row in video_acked_df.iterrows():
        session = row['session_id']  # Use only session_id
        if session not in d:
            continue

        video_ts = int(row['video_ts'])
        if video_ts not in d[session]:
            continue

        dsv = d[session][video_ts]
        sent_ts = dsv['sent_ts']
        acked_ts = pd.Timestamp(row['time'])
        dsv['acked_ts'] = acked_ts
        # dsv['trans_time'] = (acked_ts - sent_ts) / np.timedelta64(1, 's')
        # TODO: Make sure the below line is equal to the top one
        dsv['trans_time'] = (acked_ts - sent_ts).total_seconds()

    return d

def append_past_chunks(ds, next_ts, row):
    i = 1
    past_chunks = []
    while i <= PAST_CHUNKS:
        ts = next_ts - i * VIDEO_DURATION
        if ts in ds and 'trans_time' in ds[ts]:
            past_chunks = [ds[ts]['delivery_rate'],
                           ds[ts]['cwnd'], ds[ts]['in_flight'],
                           ds[ts]['min_rtt'], ds[ts]['rtt'],
                           ds[ts]['size'], ds[ts]['trans_time']] + past_chunks
        else:
            nts = ts + VIDEO_DURATION  # padding with the nearest ts
            padding = [ds[nts]['delivery_rate'],
                       ds[nts]['cwnd'], ds[nts]['in_flight'],
                       ds[nts]['min_rtt'], ds[nts]['rtt']]
            if nts == next_ts:
                padding += [0, 0]  # next_ts is the first chunk to send
            else:
                padding += [ds[nts]['size'], ds[nts]['trans_time']]
            break
        i += 1
    if i != PAST_CHUNKS + 1:  # break in the middle; padding must exist
        while i <= PAST_CHUNKS:
            past_chunks = padding + past_chunks
            i += 1
    row += past_chunks
    
def prepare_input_output(d):
    ret = [{'in': [], 'out': []} for _ in range(5)]  # FUTURE_CHUNKS = 5

    for session in d:
        ds = d[session]

        for next_ts in ds:
            if 'trans_time' not in ds[next_ts]:
                continue

            row = []

            # Append past chunks
            append_past_chunks(ds, next_ts, row)

            # Append the TCP info of the next chunk
            row += [ds[next_ts]['delivery_rate'],
                    ds[next_ts]['cwnd'], ds[next_ts]['in_flight'],
                    ds[next_ts]['min_rtt'], ds[next_ts]['rtt']]

            # Generate FUTURE_CHUNKS rows
            for i in range(5):  # FUTURE_CHUNKS = 5
                row_i = row.copy()

                ts = next_ts + i * VIDEO_DURATION
                if ts in ds and 'trans_time' in ds[ts]:
                    row_i += [ds[ts]['size']]

                    ret[i]['in'].append(row_i)
                    ret[i]['out'].append(ds[ts]['trans_time'])

    return ret

def save_processed_data(output_file, processed_data):
    """
    Save processed data to a file.
    """
    with open(output_file, 'wb') as f:
        pickle.dump(processed_data, f)
    print(f"Processed data saved to {output_file}")