# classifier.py
from copyreg import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from torch import manual_seed, no_grad
import esm.pretrained
 
class Classifier:

    def create_inputs_ESM(self, heavy_seqs, light_seqs):
        manual_seed(42)
        np.random.seed(42)
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        batch_converter = alphabet.get_batch_converter()

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

        return X


    def run_clf(self, heavy_seqs, light_seqs):
        manual_seed(42)
        np.random.seed(42)

        X = self.create_inputs_ESM(heavy_seqs, light_seqs)

        # Load the model back into memory
        SGDClassifier_model = self.loaded_model

        predictions = SGDClassifier_model.predict(X)

        return predictions