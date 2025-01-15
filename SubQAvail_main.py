# ----- Package Import ----- #
from copyreg import pickle
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from torch import manual_seed, no_grad
import esm.pretrained

import os
import subprocess
#################
def process_file(filepath):

    dataset = pd.read_csv(filepath)
    name = dataset['Name'].to_list()
    Heavy_seq = dataset['Heavy_Chain'].to_list()
    Light_seq = dataset['Light_Chain'].to_list()

    model_path = 'SubQAvail_model/Final_Saved_Model.pkl'
    with open(model_path, 'rb') as file:
        Save_Classifier = pickle.load(file)

        Predictions = Save_Classifier.run_clf(Heavy_seq,Light_seq)

    
    df2 = pd.DataFrame({
    'Name': name,
    'High/Low Bioavailability': Predictions
    })

    predictions_path = 'uploads/BioAvail_Prediction.csv'
    df2.to_csv(predictions_path, index=False)

    FOLDER='uploads'
    input_data_path = os.path.join(FOLDER, 'input_data.csv')
    try:
        os.remove(input_data_path)
        print(f"{input_data_path} has been deleted.")
    except FileNotFoundError:
        print(f"{input_data_path} not found. File might have been deleted already.")
    except Exception as e:
        print(f"Error deleting {input_data_path}: {e}")

    return predictions_path
