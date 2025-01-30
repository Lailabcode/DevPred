# ----- Package Import ----- #
from copyreg import pickle
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
from torch import manual_seed, no_grad
import esm.pretrained
import joblib
import os
import subprocess
from torch import manual_seed
from antiberty import AntiBERTyRunner

# class Classifier:
#     def __init__(self):
#         self.loaded_model = joblib.load("SGDClassifier.joblib")

#     def create_inputs_antiBERTy(self, heavy_seqs, light_seqs):
#         manual_seed(42)
#         antiberty = AntiBERTyRunner()
#         heavy_embeddings = antiberty.embed(heavy_seqs)
#         light_embeddings = antiberty.embed(light_seqs)

#         def tensors_to_numpy_average(tensors):
#             np_arrays = [tensor.numpy() for tensor in tensors]
#             avg_arrays = [np.sum(np_array, axis=0) / 512 for np_array in np_arrays]
#             final_array = np.vstack(avg_arrays)
#             return final_array

#         X1 = tensors_to_numpy_average(heavy_embeddings)
#         X2 = tensors_to_numpy_average(light_embeddings)
#         X = np.hstack((X1, X2))
#         return X

#     def run_clf(self, heavy_seqs, light_seqs):
#         manual_seed(42)
#         np.random.seed(42)
#         X = self.create_inputs_antiBERTy(heavy_seqs, light_seqs)
#         SGDClassifier_model = self.loaded_model
#         predictions = SGDClassifier_model.predict(X)
#         return predictions


def create_inputs_antiBERTy(heavy_seqs, light_seqs):
    manual_seed(42)
    antiberty = AntiBERTyRunner()
    heavy_embeddings = antiberty.embed(heavy_seqs)
    light_embeddings = antiberty.embed(light_seqs)

    def tensors_to_numpy_average(tensors):
        np_arrays = [tensor.numpy() for tensor in tensors]
        avg_arrays = [np.sum(np_array, axis=0) / 512 for np_array in np_arrays]
        final_array = np.vstack(avg_arrays)
        return final_array

    X1 = tensors_to_numpy_average(heavy_embeddings)
    X2 = tensors_to_numpy_average(light_embeddings)
    X = np.hstack((X1, X2))
    return X

def process_file(filepath):
    print("check point 1")  ##################

    dataset = pd.read_csv(filepath)
    name = dataset['Name']
    heavy_seqs = dataset['Heavy_Chain']
    light_seqs = dataset['Light_Chain']

    print("check point 2")  ##############

    model_path = '/app/SubQAvail_model/SGDClassifier.joblib'
    # joblib.dump(Classifier(), "Final_Saved_Model.joblib")
    # model_path = '/app/SubQAvail_model/Final_Saved_Model.joblib'
    try:
        model = joblib.load(model_path)
        print("check point 3: model loaded successfully") ############# 
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        raise RuntimeError("Failed to load the model. Please check the model file path and format.")

    try:
        X = create_inputs_antiBERTy(heavy_seqs, light_seqs)
        Predictions = model.predict(X)

        # Predictions = model.run_clf(heavy_seqs, light_seqs)
    except Exception as e:
        print(f"error happens: {e}")
        raise RuntimeError("Prediction failed. Please check input sequences or model compatibility.")

    print("check point 4")  #################

    df2 = pd.DataFrame({
        'Name': name,
        'High/Low Bioavailability': Predictions
    })

    predictions_path = 'uploads/BioAvail_Prediction.csv'
    df2.to_csv(predictions_path, index=False)

    FOLDER = 'uploads'
    input_data_path = os.path.join(FOLDER, 'input_data.csv')
    try:
        os.remove(input_data_path)
        print(f"{input_data_path} has been deleted.")
    except FileNotFoundError:
        print(f"{input_data_path} not found. File might have been deleted already.")
    except Exception as e:
        print(f"Error deleting {input_data_path}: {e}")

    return predictions_path