# 293N: Addressing Underspecification with Trustee, netReplica, and LLMs

The goal of our project is to streamline the process of identifying shortcut learning in a Trustee trust report decision tree with the use of LLMs. To test the feasibility of using an LLM to replace a domain expert in analyzing a model's decision tree, we took datasets collected with netReplica and puffer, purposely skewed them to introduce biases, trained a model on these skewed datsets, and developed an interface that allows an LLM (specifically Google AI's Gemini model) to process the decision tree produced by Trustee from the given model and dataset. We then instructed the LLM to provide possible modifications to the dataset to mitigate any bias it discovered. 


## Instructions for Reproducing Results

1. Download a .csv file for a purposely skewed dataset from this [Datasets](https://drive.google.com/drive/u/0/folders/1pguyQTppb_Tkx7trTBFLIMRQpi-62t51) folder (Note: files were too large to be stored on GitHub). Descriptions of the skewed datasets and their expected biases are listed [here](https://docs.google.com/document/d/1yiwdD8YjEYpeizg4z381aw-lVW_s7Haqur7X7dgoRc8/edit?tab=t.0). 
2. Train the model on the skewed dataset: store the downloaded .csv file in the "data" folder and run Train_Evaluate_CSV.ipynb
3. Save the Graphviz DOT representation of the resulting decision tree in a .txt file
4. In ask_gemini.py, replace the file path in the main function with the file path to the .txt file (line 76)
5. Run ask_gemini.py--Gemini will output its opinion on the dataset's biases. Compare this output with the bias that was purposely introduced by skewing the dataset (as described in the document from Step 1). 
