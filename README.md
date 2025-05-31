# 293N: Addressing Underspecification with Trustee, netReplica, and LLMs

The goal of our project is to streamline the process of identifying shortcut learning in a Trustee trust report decision tree with the use of LLMs. To test the feasibility of using an LLM to replace a domain expert in analyzing a model's decision tree, we took datasets collected with netReplica and puffer, purposely skewed them to introduce biases, trained a model on these skewed datsets, and developed an interface that allows an LLM (specifically Google AI's Gemini model) to process the decision tree produced by Trustee from the given model and dataset. We then instructed the LLM to provide possible modifications to the dataset to mitigate any bias it discovered. 


## Instructions for Reproducing Results

0. Get python 3.8.18
1. Clone github repo
2. Create a folder cs293-underspec/data
3. Download a .csv file for a purposely skewed dataset from this [Datasets](https://drive.google.com/drive/u/0/folders/1pguyQTppb_Tkx7trTBFLIMRQpi-62t51) folder (Note: files were too large to be stored on GitHub). Descriptions of the skewed datasets and their expected biases are listed in the "What are the skewed datasets?" section.
4. Move datasets to cs293-underspec/data
5. Edit the boolean variables in `config.py` to enable/disable various tests
6. Run `pip install -r requirements.txt`
7. Run `python split.py` to generate data for the enabled tests
8. Run `python train_evaluate.py` to train and evaluate models for the enabled tests and generate output files
9. In `ask_gemini.py`, replace the file path in the main function (line 76) with the file path to the desired graphviz-formatted file called `trustee.dot`
10. Run `ask_gemini.py`--Gemini will output its opinion on the dataset's biases. Compare this output with the bias that was purposely introduced by skewing the dataset (as described in the document from Step 3). 

## Dataset info:
We have two datasets: netrep_100ms.csv, which was previously collected at UCSB using netReplica and other tools, and oct_100ms.csv, which is data from Stanford's Puffer project. Both datasets have 7 feature types:
- delivery rate
- cwnd
- in flight
- min rtt
- rtt
- size
- trans time
All 7 features are collected for 9 chunks of data transmission numbered 0 through 8, except for trans time, which is not given for the 9th chunk. This results in a total of 62 input features. We use these input features to predict trans time for the 9th chunk. 

## What are the skewed datasets? 
We came up with 6 ways to skew the datasets to artificially induce bias. Tests 1-6 exclusively use netrep_100ms.csv, and test 7 incorporates oct_100ms.csv
1. ascend_descend: we noticed that the `delivery_rate`, `cwnd`, and `size` parameters tend to increase in netrep_100ms.csv from chunk 0 to chunk 8. We thought this could be caused by a strong network connection on an uncongested network, allowing for throughput to improve. However, in some situations these parameters decreased over time, potentially indicating un-ideal network conditions. Excluding these cases from training data might make the model only learn the happy "ascending" cases without getting insight from the arguably more important "descending" cases where network conditions aren't ideal. Datapoints where two or more of the three variables are increasing are called "ascending' and datapoints where two or more of the three variables are decreasing are called "descending."
2. cwnd_len: we noticed a long-tailed distribution in cwnd (number of in-flight packets allowed without a response). We wanted to see what would happen if we separated these out (about 4% of the total).
3. delivery_rate_len: we noticed a spike then a big drop in datapoint frequency at around 310 Mbps, indicating a paradigm shift in network conditions, so we split the data here. (~23% of datapoints had delivery_rate > 340)
4. rtt_len: we noticed a long-tailed distribution in rtt, indicating some tests (~3%) experienced high latency. We wanted to isolate these to see if removing them impacted data quality.
5. trans_time_len: we noticed a long-tailed distributed in trans_time, indicating some tests (~3%) experienced high latency. We wanted to isolate these to see if removing them impacted data quality.
6. compare_netrep_oct: here we train the model on one dataset and evaluate it on the other to see how well the model transfers to new data. we also train and evaluate the model on the union of both datasets to see if more diverse input data improves black-box model and Trustee decision tree quality. 

## What files are generated?
`split.py` will generate sub-folders in `data/` which each contain a subset of the original datasets. These reduced datasets should each have their own biases compared to the original datasets because they remove an important group of datapoints from the distribution. This results in a simpler dataset that does not contain the same diversity of training data.

`train_evaluate.py` will generate sub-folders in `evals/` and `models/`. 
- `evals/` contains all the test results. Each sub-folder corresponds to one of the tests, and each test may have multiple folders depending on which dataset was used for training and which dataset was used for evaluation. For example, `evals/ascend_descend/train_ascending_eval_descending` contains the results of an ascend_descend test where the black-box model was trained with ascending data but evaluated on the descending data. In each of these folders are the following:
  - model_class_report.json: sklearn classification_report evaluating the trained model's performance against the true values
  - mse.txt: the mean squared error of the trained black-box model's predictions
  - trustee_class_report.json: sklearn classification_report evaluated Trustee's decision tree against the trained black-box model
  - trustee.dot: graphviz text-based representation of the Trustee decision tree
  - trustee.png: graphical representation of the Trustee decision tree
- `models/` contains all the trained pytorch models. Each sub-folder corresponds to one of the tests. Inside are the trained model(s) used by that test.
