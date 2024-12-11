from rdkit.Chem import Descriptors
def check_smiles(mol, smiles):
    if mol is None:
        return False, "Invalid input (SMILES does not exist): " + str(smiles)
    return True, ""

def check_organic(mol, smiles):
    if 'C' not in smiles:
        return False, "Invalid input (Inorganic): " + str(smiles)
    return True, ""

def check_weight(mol, smiles):
    if Descriptors.MolWt(mol) < 60:
        return False, "Invalid input (Low Molecular Weight): " + str(smiles)
    return True, ""