from torch import manual_seed, no_grad
import numpy as np
import esm.pretrained

class Classifier:
    def create_inputs_ESM(self, heavy_seqs, light_seqs):
        manual_seed(42)
        np.random.seed(42)
        model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        batch_converter = alphabet.get_batch_converter()

        def seqs_to_numpy_array(seqs):
            if isinstance(seqs, str):
                seqs = [seqs]
            batch_converter_input = [("antibody", seq) for seq in seqs]

            model.eval()
            batch_labels, batch_strs, batch_tokens = batch_converter(batch_converter_input)

            batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

            with no_grad():
                results = model(batch_tokens, repr_layers=[33], return_contacts=True)
            token_representations = results["representations"][33]

            sequence_representations = [
                token_representations[i, 1 : tokens_len - 1].mean(0)
                for i, tokens_len in enumerate(batch_lens)
            ]

            np_arrays = [tensor.numpy() for tensor in sequence_representations]
            return np.vstack(np_arrays)

        X1 = seqs_to_numpy_array(heavy_seqs)
        X2 = seqs_to_numpy_array(light_seqs)

        return np.hstack((X1, X2))

    def run_clf(self, heavy_seqs, light_seqs):
        manual_seed(42)
        np.random.seed(42)

        X = self.create_inputs_ESM(heavy_seqs, light_seqs)

        predictions = self.loaded_model.predict(X)

        return predictions