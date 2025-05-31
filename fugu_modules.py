import json
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
NUM_EPOCHS = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
VIDEO_DURATION = 180180
PKT_BYTES = 1500
MILLION = 1000000
PAST_CHUNKS = 8
FUTURE_CHUNKS = 5
TUNING = True
CHECKPOINT = 100

inference = True

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
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        print("Starting training...")
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
            self.evaluate(test_input, test_output)
        self.save(model_path)
        print("Training complete. Model saved.")

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

    def evaluate_with_trustee(self, test_input, test_output, output_folder):
        os.makedirs(output_folder, exist_ok=True)
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

            with open(output_folder + "model_class_report.json", "a") as f:
                json.dump(classification_report(test_output_discretized, 
                                                class_preds, zero_division=0, 
                                                output_dict=True), f, indent=4)
            print("Wrote model classification report")
            
            trustee = ClassificationTrustee(expert=self)
            trustee.fit(pd_input, test_output_discretized, num_iter=10, 
                        num_stability_iter=4, samples_size=0.3, verbose=True, 
                        predict_method_name="predict")
            dt, pruned_dt, agreement, reward = trustee.explain()
            dt_y_pred = dt.predict(pd_input)

            with open(output_folder + "trustee_class_report.json", "a") as f:
                json.dump(classification_report(class_preds, dt_y_pred,
                                                zero_division=0, 
                                                output_dict=True), f, indent=4)
            print("Wrote trustee classification report")

            dot_data = tree.export_graphviz(pruned_dt, 
                                            class_names=[str(i)for i in range(21)], 
                                            feature_names=self.COLUMNS,
                                            filled=True,rounded=True,
                                            special_characters=True)
            graph = graphviz.Source(dot_data)
            fil = graph.render(output_folder + "trustee", format="png")

    def evaluate(self, test_input, test_output, output_folder=None):
        if output_folder is not None:
            os.makedirs(output_folder, exist_ok=True)
        self.model.eval()
        pd_input = pd.DataFrame(test_input, columns=self.COLUMNS)
        # pd_input = pd.DataFrame(test_input.cpu().numpy(), columns=self.COLUMNS)

        # cnt = 0
        with torch.no_grad():
            predictions = self.predict_cont(pd_input)
            # Print metrics
            # sum = 0
            # print("Length of predictions: ", len(predictions))
            # print("Length of actual: ", len(test_output))
            # calculate the diffrerence between the predictions and the actual values and save it in a list diffs
            # diffs = []
            # to_save = []
            # for i in range(len(predictions)):
            #     trans_times = [0]* 10
            #     for j in range (8):
            #         trans_times[j] = pd_input .iloc[i][f'trans_time{j}'] 
            #     cnt +=1 
            #     # if cnt % 1000 == 0:
            #     #     print(cnt)        
            #     trans_times[8] = predictions[i]
            #     trans_times[9] = test_output[i]
            #     to_save.append(trans_times)
                # diffs.append(predictions[i] - test_output[i])
                # sum += abs(diffs[i])
            # save the to save list to a csv file
            # pd.DataFrame(to_save).to_csv(save_loc, index=False)
            # save diffs to a csv file
            # pd.DataFrame(diffs).to_csv('/mnt/md0/jaber/puffer_trustee/ttpABR_14days/diffs.csv', index=False)
            # Mean Squared Error
            mse_loss = mean_squared_error(test_output, predictions)
            # mse_loss = mean_squared_error(test_output.cpu().numpy(), predictions.cpu().numpy())

            print(f"Test Mean Squared Error: {mse_loss}")
            if output_folder is not None:
                with open(output_folder + "mse.txt", "a") as f:
                    f.write(f"Test Mean Squared Error: {mse_loss}\n\n")
            return mse_loss
