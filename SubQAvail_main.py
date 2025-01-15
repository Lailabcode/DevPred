# ----- Package Import ----- #
from copyreg import pickle
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from torch import manual_seed, no_grad
import esm.pretrained

import os
import subprocess
class Classifier:

    def create_inputs_ESM(self, heavy_seqs, light_seqs):
        manual_seed(42)
        np.random.seed(42)
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        batch_converter = alphabet.get_batch_converter()

        print("check point 6")#######

        def seqs_to_numpy_array(seqs):
            # Checks if the input is a string, and if it is it turns it into a list
            # allows for single inputs of sequences or multiples
            if isinstance(seqs, str):
                seqs = [seqs]
            batch_converter_input = []
            for seq in seqs:
                batch_converter_input.append(("antibody", seq))

            model.eval()
            batch_labels, batch_strs, batch_tokens = batch_converter(
                batch_converter_input
            )

            # calculates the length of each sequence by counting non-padding tokens.
            batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

            # Extract per-residue representations (on CPU), creates tensors with dimension
            with no_grad():
                results = model(batch_tokens, repr_layers=[33], return_contacts=True)
            token_representations = results["representations"][33]

            # Sequence representation, this turns it from a list of dimension (Variable length, 1280) to
            # a list of tensors with dimension (1280, )
            sequence_representations = []
            for i, tokens_len in enumerate(batch_lens):
                sequence_representations.append(
                    token_representations[i, 1 : tokens_len - 1].mean(0)
                )

            # This converts list of tensors into one numpy array
            np_arrays = [tensor.numpy() for tensor in sequence_representations]
            final_array = np.vstack(np_arrays)
            return final_array

        X1 = seqs_to_numpy_array(heavy_seqs)
        X2 = seqs_to_numpy_array(light_seqs)
        # Combines the 2 light and heavy arrays
        X = np.hstack((X1, X2))

        print("check point 5")#######

        return X


    def run_clf(self, heavy_seqs, light_seqs):
        manual_seed(42)
        np.random.seed(42)

        X = self.create_inputs_ESM(heavy_seqs, light_seqs)

        # Load the model back into memory
        SGDClassifier_model = self.loaded_model

        Predictions = SGDClassifier_model.predict(X)
        
        print("check point 4")#######
        
        return Predictions

#################
def process_file(filepath):

    dataset = pd.read_csv(filepath)
    name = dataset['Name'].to_list()
    heavy_seqs = dataset['Heavy_Chain'].to_list()
    light_seqs = dataset['Light_Chain'].to_list()

    print("check point 2")#######

    model_path = 'SubQAvail_model/Final_Saved_Model.pkl'
    with open(model_path, 'rb') as file:
        Save_Classifier = pickle.load(file)

    print("check point 3")#######

    Predictions = Save_Classifier.run_clf(heavy_seqs,light_seqs)
    
    print("check point 1")#######

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
