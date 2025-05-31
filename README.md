# 293N: Addressing Underspecification with Trustee, netReplica, and LLMs

The goal of our project is to streamline the process of identifying shortcut learning in a Trustee trust report decision tree with the use of LLMs. To test the feasibility of using an LLM to replace a domain expert in analyzing a model's decision tree, we took datasets collected with netReplica and puffer, purposely skewed them to introduce biases, trained a model on these skewed datsets, and developed an interface that allows an LLM (specifically Google AI's Gemini model) to process the decision tree produced by Trustee from the given model and dataset. We then instructed the LLM to provide possible modifications to the dataset to mitigate any bias it discovered. 


## Instructions for Reproducing Results

1. Clone github repo
2. Create a folder cs293-underspec/data
3. Download a .csv file for a purposely skewed dataset from this [Datasets](https://drive.google.com/drive/u/0/folders/1pguyQTppb_Tkx7trTBFLIMRQpi-62t51) folder (Note: files were too large to be stored on GitHub). Descriptions of the skewed datasets and their expected biases are listed [here](https://docs.google.com/document/d/1yiwdD8YjEYpeizg4z381aw-lVW_s7Haqur7X7dgoRc8/edit?tab=t.0). 
4. Move datasets to cs293-underspec/data
5. Edit the boolean variables in `config.py` to enable/disable various tests
6. Run `pip install -r requirements.txt`
7. Run `python split.py` to generate data for the enabled tests
8. Run `python train_evaluate.py` to train and evaluate models for the enabled tests and generate output files
9. In `ask_gemini.py`, replace the file path in the main function (line 76) with the file path to the desired graphviz-formatted file called `trustee.dot`
10. Run `ask_gemini.py`--Gemini will output its opinion on the dataset's biases. Compare this output with the bias that was purposely introduced by skewing the dataset (as described in the document from Step 3). 

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

`explore.py` is a script to explore the distributions of parameter values in the datasets. It writes to `explore/`.