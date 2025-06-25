# ----- Package Import ----- #
import joblib
import numpy as np
import pandas as pd
import os
# from copyreg import pickle # Not needed, `pickle` is imported directly
from sklearn.metrics import accuracy_score
from torch import manual_seed
# Assuming antiberty is installed and available in your Render environment
from antiberty import AntiBERTyRunner 

# --- Classifier Class Definition ---
# This class MUST be defined in the same file or a module imported by the file
# where the model "Final_Saved_Model_LinearSVC.joblib" is loaded.
class Classifier:
    """
    A wrapper class for loading a pre-trained machine learning model
    and preparing inputs using AntiBERTy embeddings.
    """
    def __init__(self):
        """
        Initializes the Classifier by loading the actual machine learning model.
        Assumes "LinearSVC_Classifier.joblib" is the path to the trained sklearn model.
        """
        # Ensure this path is correct relative to where your app runs
        # or use an absolute path if necessary.
        model_path_inner = "SVC_Classifier.joblib" 
        try:
            self.loaded_model = joblib.load(model_path_inner)
            print(f"Internal model '{model_path_inner}' loaded successfully within Classifier.")
        except FileNotFoundError:
            print(f"Error: Internal model file '{model_path_inner}' not found.")
            raise
        except Exception as e:
            print(f"Error loading internal model '{model_path_inner}': {e}")
            raise

    def create_inputs_antiBERTy(self, heavy_seqs, light_seqs):
        """
        Creates the needed inputs using AntiBERTy, a PyTorch-based model.
        Embeds heavy and light sequences and combines their averaged embeddings.
        """
        manual_seed(42) # For reproducibility with AntiBERTyRunner
        
        # Initialize AntiBERTyRunner for embedding sequences
        antiberty = AntiBERTyRunner()

        # Create embeddings (tensors) for both light and heavy sequences
        heavy_embeddings = antiberty.embed(heavy_seqs)
        light_embeddings = antiberty.embed(light_seqs)

        def tensors_to_numpy_average(tensors):
            """
            Converts a list of PyTorch tensors to a 2D NumPy array
            by averaging over the 512 dimension for each tensor.
            """
            # Convert each tensor to a NumPy array
            np_arrays = [tensor.numpy() for tensor in tensors]
            
            # Sum over the 512 dimension and divide by 512 to get average of sequence lengths
            # This transforms a list of 2D arrays (seq_len, 512) to a list of 1D arrays (512,)
            avg_arrays = [np.sum(np_array, axis=0) / 512 for np_array in np_arrays]
            
            # Stack arrays to form the final 2D array (num_sequences, 512)
            final_array = np.vstack(avg_arrays)
            return final_array

        # Create NumPy arrays for both light and heavy sequences' embeddings
        X1 = tensors_to_numpy_average(heavy_embeddings)
        X2 = tensors_to_numpy_average(light_embeddings)
        
        # Combine the light and heavy arrays to form a (dataset_size, 1024) shape
        X = np.hstack((X1, X2))
        return X

    def run_clf(self, heavy_seqs, light_seqs):
        """
        Runs the classifier on the given heavy and light sequences.
        """
        manual_seed(42) # For reproducibility
        np.random.seed(42) # For reproducibility

        # Prepare input features using the internal AntiBERTy embedding method
        X = self.create_inputs_antiBERTy(heavy_seqs, light_seqs)

        # Make predictions using the loaded scikit-learn model
        predictions = self.loaded_model.predict(X)
        
        return predictions

# --- Function for processing the file in your web application ---
def process_file(filepath):
    """
    Processes an input CSV file containing heavy and light chain sequences,
    uses the pre-trained model to make predictions, and saves the predictions
    to a new CSV file.
    """
    print("check point 1: Starting file processing.")

    # Load the input dataset
    try:
        dataset = pd.read_csv(filepath)
        name = dataset['Name']
        heavy_seqs = dataset['Heavy_Chain']
        light_seqs = dataset['Light_Chain']
    except KeyError as e:
        print(f"Error: Missing expected column in CSV: {e}. Please ensure 'Name', 'Heavy_Chain', and 'Light_Chain' columns exist.")
        raise RuntimeError("Input CSV format error.")
    except pd.errors.EmptyDataError:
        print(f"Error: CSV file '{filepath}' is empty.")
        raise RuntimeError("Input CSV file is empty.")
    except Exception as e:
        print(f"Error reading CSV file '{filepath}': {e}")
        raise RuntimeError("Failed to read input CSV file.")

    print("check point 2: Input data loaded from CSV.")

    # Define the path to your main saved Classifier instance model
    # This path must be correct within your Render deployment environment.
    model_path = '/app/SubQAvail_model/Final_Saved_Model_LinearSVC.joblib'
    
    # Load the Classifier instance
    try:
        # joblib.load will now find the Classifier class because it's defined above
        model_instance = joblib.load(model_path)
        print(f"check point 3: Model '{model_path}' loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Main model file '{model_path}' not found.")
        raise RuntimeError("Failed to load the model. Please check the model file path.")
    except Exception as e:
        print(f"Error loading main model from {model_path}: {e}")
        raise RuntimeError(f"Failed to load the model. Error: {e}")

    # Make predictions using the loaded Classifier instance
    try:
        # Use the run_clf method of the loaded Classifier instance
        predictions = model_instance.run_clf(heavy_seqs, light_seqs)
        # Note: The original code had `Predictions = model.predict(X)` and `Predictions = model.run_clf(...)` commented out.
        # Since `model_instance` is a `Classifier` object, `run_clf` is the correct method to call.
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise RuntimeError("Prediction failed. Please check input sequences or model compatibility.")

    print("check point 4: Predictions generated.")

    # Create a DataFrame for predictions
    df_predictions = pd.DataFrame({
        'Name': name,
        'High/Low Bioavailability': predictions
    })

    # Save predictions to a CSV file
    FOLDER = 'uploads'
    if not os.path.exists(FOLDER):
        os.makedirs(FOLDER) # Create directory if it doesn't exist
    predictions_path = os.path.join(FOLDER, 'BioAvail_Prediction.csv')
    try:
        df_predictions.to_csv(predictions_path, index=False)
        print(f"Predictions saved to {predictions_path}")
    except Exception as e:
        print(f"Error saving predictions to CSV: {e}")
        raise RuntimeError("Failed to save prediction results.")

    # Clean up the input file
    input_data_path = os.path.join(FOLDER, 'input_data.csv') # Assuming this is the input file name in 'uploads'
    try:
        if os.path.exists(input_data_path):
            os.remove(input_data_path)
            print(f"{input_data_path} has been deleted.")
        else:
            print(f"{input_data_path} not found. File might have been deleted already or named differently.")
    except Exception as e:
        print(f"Error deleting {input_data_path}: {e}")

    return predictions_path

# --- Example Usage (for local testing, if needed) ---
# This part is for saving the model initially (e.g., on your local machine)
# It's important that 'LinearSVC_Classifier.joblib' exists when you run this
# and that 'mAb_inputdata.csv' exists for testing the full flow.
if __name__ == "__main__":
    # --- Part 1: Initial Model Saving (Do this once to create the joblib files) ---
    # This block would typically be run as a separate script to train and save models.
    # For demonstration, we'll simulate saving the Classifier instance.
    print("--- Simulating initial model saving ---")
    try:
        # Make sure 'LinearSVC_Classifier.joblib' exists for the Classifier.__init__
        # For a real scenario, this would be your trained sklearn model.
        # For this example, let's create a dummy one if it doesn't exist.
        if not os.path.exists("LinearSVC_Classifier.joblib"):
            from sklearn.svm import LinearSVC
            dummy_model = LinearSVC(random_state=42, dual=False, max_iter=1000)
            # You would train this model with some dummy data here
            # For this example, we just save an untrained instance to allow Classifier to load it.
            joblib.dump(dummy_model, "LinearSVC_Classifier.joblib")
            print("Created a dummy 'LinearSVC_Classifier.joblib' for demonstration.")

        # Save an instance of the Classifier. This is your "Final_Saved_Model_LinearSVC.joblib"
        joblib.dump(Classifier(), "Final_Saved_Model_LinearSVC.joblib")
        print("Classifier instance saved as 'Final_Saved_Model_LinearSVC.joblib'.")

        # --- Part 2: Testing the prediction pipeline ---
        print("\n--- Testing the prediction pipeline with dummy data ---")
        # Create a dummy CSV for testing the process_file function
        dummy_csv_content = """Name,Heavy_Chain,Light_Chain
Sample1,QVQLQESGPGLVKPSQTLSLTCTVSGGSISSGFYYWSWIRQPPGKGLQEWIGRIYPGDGTNYNQKFNGRLTLTVDTSTSTAYMELSSLRSEDTAVYYCARDGGYGSGTTVTVSS,EIVLTQSPATLSLSPGERATLSCRASQSVSSSYLAWYQQKPGQAPRLLIYGASSRATGIPDRFSGSGSGTDFTLTISSLEPEDFAVYYCQQYGSSLSSRTFGGGTKVEIK
Sample2,QVQLQESGPGLVKPSQTLSLTCTVSGGSISSGFYYWSWIRQPPGKGLQEWIGRIYPGDGTNYNQKFNGRLTLTVDTSTSTAYMELSSLRSEDTAVYYCARDGGYGSGTTVTVSS,EIVLTQSPATLSLSPGERATLSCRASQSVSSSYLAWYQQKPGQAPRLLIYGASSRATGIPDRFSGSGSGTDFTLTISSLEPEDFAVYYCQQYGSSLSSRTFGGGTKVEIK
"""
        # Create 'uploads' directory if it doesn't exist for testing
        test_folder = 'uploads'
        if not os.path.exists(test_folder):
            os.makedirs(test_folder)

        test_input_csv_path = os.path.join(test_folder, "input_data.csv")
        with open(test_input_csv_path, "w") as f:
            f.write(dummy_csv_content)
        print(f"Created dummy input CSV: {test_input_csv_path}")

        # Run the processing function
        output_csv_path = process_file(test_input_csv_path)
        print(f"Processing complete. Predictions saved to: {output_csv_path}")

        # Verify output
        df_output = pd.read_csv(output_csv_path)
        print("\n--- Output Predictions (first 5 rows) ---")
        print(df_output.head())

        # Clean up test files
        # os.remove("LinearSVC_Classifier.joblib") # Uncomment if you want to remove the dummy internal model
        # os.remove("Final_Saved_Model_LinearSVC.joblib") # Uncomment if you want to remove the saved Classifier instance
        # os.remove(output_csv_path) # Uncomment to remove the prediction output file
        # This will be handled by the `process_file` function for `input_data.csv`
        # if os.path.exists(test_input_csv_path):
        #     os.remove(test_input_csv_path)

    except ImportError as e:
        print(f"\nImportError: {e}. Please ensure 'antiberty' and other required packages are installed.")
        print("You might need to install them using: pip install numpy pandas scikit-learn torch antiberty")
    except Exception as e:
        print(f"\nAn error occurred during example usage: {e}")
        # To get more detailed traceback for debugging locally
        import traceback
        traceback.print_exc()