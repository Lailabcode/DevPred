import numpy as np
from torch import manual_seed
from antiberty import AntiBERTyRunner
import joblib

class Classifier:
    def create_inputs_antiBERTy(self, heavy_seqs, light_seqs):
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

    def run_clf(self, heavy_seqs, light_seqs):
        manual_seed(42)
        np.random.seed(42)
        X = self.create_inputs_antiBERTy(heavy_seqs, light_seqs)
        SGDClassifier_model = self.loaded_model
        predictions = SGDClassifier_model.predict(X)
        return predictions