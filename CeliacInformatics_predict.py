import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import PandasTools
from rdkit.Chem import Descriptors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib



def getFingerprints(mol):
    rdkbi = {}
    all_drops = [330, 342, 2, 3, 16, 36, 40, 51, 64, 81, 87, 103, 112, 117, 141, 142, 161, 166, 186, 194, 209, 233, 236, 238, 248, 265, 280, 284, 290, 294, 297, 312, 341, 351, 369, 371, 372, 379, 386, 402, 403, 410, 416, 417, 428, 439, 443, 447, 460, 479, 480, 491, 504, 506, 509]
    
    fingerprints = pd.DataFrame(np.array(AllChem.RDKFingerprint(mol, maxPath=5, fpSize=512, bitInfo=rdkbi)))
    cop_fing = fingerprints.copy(deep = True)
    fingerprints.drop(all_drops, inplace = True)
    fingerprints_model_input = np.array(fingerprints)
    fingerprints_model_input = fingerprints_model_input.reshape(1, 457)
    return fingerprints_model_input, rdkbi, cop_fing

def getDescriptors(mol, missingVal = None):
    descriptor_dict = {}
    for nm,fn in Descriptors._descList:
        try:
            val = fn(mol)
        except:
            import traceback
            traceback.print_exc()
            val = missingVal
        descriptor_dict[nm] = val
    desc_df_full = pd.DataFrame(descriptor_dict, index=['Values'])
    descriptors_df = pd.DataFrame({'Values': pd.Series(list(descriptor_dict.values()))})
    descriptors_model_input = np.array(descriptors_df)
    descriptors_model_input = descriptors_model_input.reshape((1, 210))
    return descriptors_model_input, descriptor_dict, desc_df_full

def getFeatures(mol):
    fing_input, rdkbi = getFingerprints(mol)[0:2]
    desc_input = getDescriptors(mol)[0]
    features_model_input = np.concatenate([fing_input, desc_input], axis = 1)
    return features_model_input, rdkbi

def make_prediction(smiles):
    mol = Chem.MolFromSmiles(smiles)
    smiles = Chem.MolToSmiles(mol)
    mol = Chem.MolFromSmiles(smiles)
    df = pd.read_csv('models/data/clf_celiac_desc_fing_data.csv')
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=1)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)  
    
    features, rdkbi = getFeatures(mol)
    features = sc.transform(features) 
    
    model = joblib.load(open('models/celiac_clf.joblib', 'rb'))

    prediction = model.predict(features)
    if prediction[0] == 1:
        prediction = "Active"
    else:
        prediction = "Inactive"
    return prediction

if __name__ == "__main__":
    #One Molecule
    smiles = "CCCCCCCCCCCCCC"
    print(smiles + ": " + str(make_prediction(smiles)))

    #Multiple Molecules
    # df_smiles = pd.read_csv("test.csv")
    # for smiles_input in df_smiles["SMILESE"]:
    #     print(smiles_input + ": " + str(make_prediction(smiles_input)))
