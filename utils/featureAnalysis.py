import numpy as np
import os
import io
from flask import Response
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import PandasTools
from rdkit.Chem import Descriptors
from rdkit.Chem import Draw
from PIL import Image
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import Figure
import catboost
from utils.features import getDescriptors, getFeatures, getFingerprints

def get_important_fingerprints(mol, rdkbi):
    bit_list = [495, 433, 382, 338, 350, 193, 74, 417, 263]
    bit_keys = ["green", "red", "green", "red", "red", "green", "green", "grey", "green"]
    substructure_list = []
    substructure_numbers = []
    key_colors = []
    for i in bit_list:
        if i in rdkbi:
            substructure = Draw.IPythonConsole.DrawRDKitBit(mol, i ,rdkbi)
            substructure_list.append(substructure)
            substructure_numbers.append(i)
            key_colors.append("green") if i in [495, 382, 193, 74, 263] else None
            key_colors.append("red") if i in [433, 338, 350] else None
            key_colors.append("grey") if i in [417] else None
    sub_file_names = []
    for index, substructure_img in enumerate(substructure_list):
        sub_file_name = 'sub' + str(index) + '.png'
        sub_file_names.append(sub_file_name)
        sub_path = os.path.join('static', 'imgs', sub_file_name)
        substructure_img.save(sub_path)
    if len(substructure_numbers) < 4:
        img_width = 175
    elif len(substructure_numbers) < 11:
        img_width = 115
    return sub_file_names, substructure_numbers, img_width, len(substructure_numbers), key_colors


def graph_important_descriptors(smiles, mol, rdkbi):
    all_descriptors = getDescriptors(mol)[1]
    important_descriptor_names = [
        "NumSaturatedHeterocycles",
        "NumSaturatedRings",
        "SMR_VSA6",
        "VSA_EState3",
        "BCUT2D_CHGHI", 
        "SlogP_VSA2",
        "BCUT2D_CHGLO",
        "PEOE_VSA12",
        "BCUT2D_MRHI",
        "PEOE_VSA8"
    ]
    molecular_weight = Descriptors.MolWt(mol)
    important_descriptors = [int(all_descriptors[descriptor]) for descriptor in important_descriptor_names]
    
    plt.figure(figsize=(10, 6))
    plt.bar(important_descriptor_names, important_descriptors, color='blue')
    for i, value in enumerate(important_descriptors):
        plt.text(i, value + 1, str(value), ha='center', va='bottom', rotation=45)
    plt.xlabel('Molecular Descriptors')
    plt.ylabel('Descriptor Values')
    plt.title(f"{smiles}\nMolecular Weight: {molecular_weight} g/mol")
    plt.ylim(0, max(important_descriptors) + 20) 
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('static/imgs/descriptor_plot.png')

def graph_spearman_ranking(smiles, mol):
    df = pd.read_csv('models/data/ic50_full_data.csv')
    df_fing = getFingerprints(mol)[2]
    df_fing = df_fing.transpose()
    df_desc = getDescriptors(mol)[2]
    feat_needed = ['EState_VSA2', 'BalabanJ', 'BCUT2D_MRHI', 'EState_VSA5', 'NumSaturatedHeterocycles', 'VSA_EState5', 'EState_VSA4', 'PEOE_VSA8', 'EState_VSA3', 'VSA_EState4', 'PEOE_VSA7', 'Chi3v', 'EState_VSA6', 'MinEStateIndex', 'HallKierAlpha', 'Chi2v', 'EState_VSA7', 'EState_VSA8', 'SlogP_VSA4', '143', 'PEOE_VSA6', 'SlogP_VSA2', '454', 'SlogP_VSA6', '390', 'SlogP_VSA5', 'SlogP_VSA1', 'PEOE_VSA3', 'SMR_VSA1', 'Chi1v', 'PEOE_VSA14', 'EState_VSA1', 'PEOE_VSA1', 'PEOE_VSA2', '336', 'PEOE_VSA10', '355', 'EState_VSA9', 'SlogP_VSA12', 'SlogP_VSA10', 'PEOE_VSA9', '275', 'HeavyAtomMolWt', 'NumHeteroatoms', 'PEOE_VSA5', 'fr_COO', 'Kappa1', '213', '43', 'Chi0v', 'SlogP_VSA3', 'PEOE_VSA12', 'Chi1', 'NHOHCount', 'Chi3n', '165', '7', 'LabuteASA', '267', 'Chi0n', '229', 'MolWt', 'PEOE_VSA11', '234', '84', 'ExactMolWt', '61', '477', '88', 'fr_COO2', 'NumValenceElectrons', '219', '325', '392', '291', '457', '464', 'fr_amide', '425', '124', 'PEOE_VSA13', 'Chi0', '158', '199', '377', '98', '500', '190', '356', '489', 'fr_NH2', 'NumAliphaticHeterocycles', 'NOCount', '411', '491', '305', 'fr_ester', '204', 'HeavyAtomCount', 'fr_C_O', 'NumHDonors', 'SlogP_VSA8', '57', '151', '442', '376', '372', '136', '320', '114', '451', '498', 'PEOE_VSA4', '60', '362', 'NumHAcceptors', 'NumAromaticHeterocycles', '94', '321', '369', '276', '105', '33', '357', '264', '96', '26', '400', 'NumAromaticCarbocycles', 'fr_Ndealkylation2', '44', '11', '274', '223', '19', '345', '398', '479', '361', 'SlogP_VSA11', 'fr_alkyl_halide', '272', '473', '492', '403', '215', '196', '431', 'fr_C_O_noCOO', '240', '383', '499', '254', '82', '38', '52', '387', 'fr_benzene', '49', '47', '494', '354', '181', '106', '184', '235', '332', '37', '50', '318', '367', '120', '257', '113', '283', '160', 'fr_allylic_oxid', '155', '18', '406', '316', '30', '6', '266', '97', '281', '146', '328', '303', 'fr_piperzine', 'SlogP_VSA7', '115', '180', '58', '483', '168', '256', '183', '510', '507']
    feat_needed = [int(item) if item.isdigit() else item for item in feat_needed]
    df_feat = pd.concat([df_fing, df_desc], axis = 1)
    df_feat_chosen = df_feat[feat_needed]
    df_feat_arr = df_feat_chosen.to_numpy().flatten()
    df_feat_arr = df_feat_arr [~np.isnan(df_feat_arr )].flatten()
    df_feat_arr = df_feat_arr.reshape((1, 200))
    df2 = pd.read_csv('models/data/reg_fing_desc_clean.csv')
    X = df2.iloc[:, :-1].values
    y = df2.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    df_feat_arr = sc.transform(df_feat_arr)
    model2 = pickle.load(open('models/REG_catboost.pkl', 'rb'))
    prediction = model2.predict(df_feat_arr)
    ic50_val = prediction[0]
    new_row = pd.DataFrame({df.columns[0]: [ic50_val]})
    df_with_pred = pd.concat([df, new_row], ignore_index=True)
    df_sorted = df_with_pred.sort_values(by=df.columns[0])
    df_sorted.reset_index(drop=True, inplace=True)
    pred_index = df_sorted.index[df_sorted[df.columns[0]] == ic50_val][0]
    plt.figure(figsize=(10, 6))
    plt.scatter(df_sorted.index, df_sorted[df.columns[0]], color='blue', label='Relative IC50s of Dataset')
    plt.scatter(pred_index, ic50_val, color='red', zorder=5, label='Predicted Relative IC50')
    plt.text(pred_index, ic50_val, f'{ic50_val:.2f}', color='red', fontsize=12, ha='right', va='bottom')
    plt.ylabel('Relative IC50 Value [µM]')
    plt.title(smiles)
    plt.xticks([])
    plt.legend()
    plt.tight_layout()
    plt.savefig('static/imgs/sp_rank.png')
    filtered_df = df_sorted[(df_sorted[df_sorted.columns[0]] >= ic50_val - 0.1) & (df_sorted[df_sorted.columns[0]] <= ic50_val + 0.1)]
    return filtered_df[[filtered_df.columns[0]]], filtered_df["Smiles"]
    
