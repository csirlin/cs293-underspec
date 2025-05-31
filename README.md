# 293N: Addressing Underspecification with Trustee, netReplica, and LLMs

## Instructions for Reproducing Results

1. Download a .csv file for a purposely skewed dataset from this [Datasets](https://drive.google.com/drive/u/0/folders/1pguyQTppb_Tkx7trTBFLIMRQpi-62t51) folder (Note: files were too large to be stored on GitHub). Descriptions of the skewed datasets and their expected biases are listed [here](https://docs.google.com/document/d/1yiwdD8YjEYpeizg4z381aw-lVW_s7Haqur7X7dgoRc8/edit?tab=t.0). 
2. Train the model on the skewed dataset: store the downloaded .csv file in the "data" folder and run Train_Evaluate_CSV.ipynb
3. Save the Graphviz DOT representation of the resulting decision tree in a .txt file
4. In ask_gemini.py, replace the file path in the main function with the file path to the .txt file (line 76)
5. Run ask_gemini.py--Gemini will output its opinion on the dataset's biases. Compare this output with the bias that was purposely introduced by skewing the dataset (as described in the document from Step 1). 
