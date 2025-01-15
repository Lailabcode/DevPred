#   /Website
#       app.py
#       /templates
#               index.html
#       /static
#           /css
#               style.css
#           /image
#               logo.png
#       /uploads
#           csv generate output: DeepSP_descriptors.csv

from flask import Flask, request, redirect, url_for, render_template, flash, send_from_directory, abort
import os
from urllib.parse import quote as url_quote
import csv
from DeepSP_main import process_file as deep_sp_process_file  
from DeepViscosity_main import process_file as deep_viscosity_process_file  
from AbDev_main import process_file as ab_dev_process_file
from SubQAvail_main import process_file as SubQAvail_process_file ############################
import numpy as np
import pandas as pd

# Celiac Informatics Imports
from flask import Flask, render_template, url_for, request
from flask_material import Material
import os
import uuid
import numpy as np
import pandas as pd
import io

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import Figure
from rdkit.Chem.Draw import IPythonConsole

import joblib
from PIL import Image
import pickle

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import PandasTools
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from rdkit.Chem import Crippen
import catboost

from utils.moleculeInputs import check_smiles, check_organic, check_weight
from utils.features import getDescriptors, getFingerprints, getFeatures, getMoleculeInfo
from utils.prediction import predict_activity
from utils.featureAnalysis import get_important_fingerprints, graph_important_descriptors, graph_spearman_ranking
from utils.drugLikeness import lipinski_report, muegge_report, ghose_report, veber_report, egan_report, check_num_violations

app = Flask(__name__)
Material(app)

app.secret_key = 'pkl'
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'txt', 'csv'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('ABout.html')

# DeepSP 
@app.route('/DeepSP', methods=['GET', 'POST'])
def deep_sp():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and allowed_file(file.filename):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

        mab_data = {
            'Name': request.form.get('mab_name', ''),
            'Heavy_Chain': request.form.get('heavy_chain', ''),
            'Light_Chain': request.form.get('light_chain', '')
        }
        filepath = write_to_csv(mab_data, 'input_data.csv')

        try:
            predictions_path = deep_sp_process_file(filepath)
            
            with open(predictions_path, 'r', newline='') as csvfile:
                reader = csv.reader(csvfile)
                predictions_data = list(reader)  
            return render_template('DeepSP.html', 
                                   predictions_data=predictions_data, 
                                   predictions_path=os.path.basename(predictions_path))
        except Exception as e:
            flash(f'Error processing file: {e}')
            return redirect(request.url)
    return render_template('DeepSP.html')

# DeepViscosity 
@app.route('/DeepViscosity', methods=['GET', 'POST'])
def deep_viscosity():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and allowed_file(file.filename):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
        mab_data = {
            'Name': request.form.get('mab_name', ''),
            'Heavy_Chain': request.form.get('heavy_chain', ''),
            'Light_Chain': request.form.get('light_chain', '')
        }
        filepath = write_to_csv(mab_data, 'input_data.csv')

        try:
            descriptors_path, predictions_path = deep_viscosity_process_file(filepath)
            
            with open(descriptors_path, 'r', newline='') as csvfile:
                reader = csv.reader(csvfile)
                descriptors_data = list(reader)  

            with open(predictions_path, 'r', newline='') as csvfile:
                reader = csv.reader(csvfile)
                predictions_data = list(reader)  
            
            return render_template('DeepViscosity.html', 
                                   descriptors_data=descriptors_data,  
                                   descriptors_path=os.path.basename(descriptors_path), 
                                   predictions_data=predictions_data,  
                                   predictions_path=os.path.basename(predictions_path))
        except Exception as e:
            flash(f'Error processing file: {e}')
            return redirect(request.url)
    return render_template('DeepViscosity.html')

# AbDev 
@app.route('/AbDev', methods=['GET', 'POST'])
def ab_dev():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and allowed_file(file.filename):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

        mab_data = {
            'Name': request.form.get('mab_name', ''),
            'Heavy_Chain': request.form.get('heavy_chain', ''),
            'Light_Chain': request.form.get('light_chain', '')
        }
        filepath = write_to_csv(mab_data, 'input_data.csv')
         
        try:
            
            descriptors_path, predictions_path = ab_dev_process_file(filepath)
              
            with open(descriptors_path, 'r', newline='') as csvfile:
                reader = csv.reader(csvfile)
                descriptors_data = list(reader)  

            with open(predictions_path, 'r', newline='') as csvfile:
                reader = csv.reader(csvfile)
                predictions_data = list(reader)
 
            return render_template('AbDev.html', 
                                   descriptors_data=descriptors_data,  
                                   descriptors_path=os.path.basename(descriptors_path), 
                                   predictions_data=predictions_data,  
                                   predictions_path=os.path.basename(predictions_path))
        except Exception as e:
            flash(f'Error processing file: {e}')
            return redirect(request.url)
            
    return render_template('AbDev.html')

def write_to_csv(data, filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    with open(filepath, 'w', newline='') as csvfile:
        fieldnames = ['Name', 'Heavy_Chain', 'Light_Chain']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(data)
    return filepath


@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)
    except FileNotFoundError:
        abort(404)

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import PandasTools
from rdkit.Chem import Descriptors


@app.route('/celiac_informatics', methods=['GET', 'POST'])
def celiac_informatics():
    if request.method == 'POST':
        smiles = request.form['smiles_name']
        smiles = str(smiles)
        mol = Chem.MolFromSmiles(smiles)

        smiles = Chem.MolToSmiles(mol)
        mol = Chem.MolFromSmiles(smiles)

        molecular_formula = getMoleculeInfo(mol)
        
        molecule_img = Draw.MolToImage(mol)
        img_path = os.path.join('static', 'imgs', 'molecule.png')
        molecule_img.save(img_path)
        
        isValidSmiles, message = check_smiles(mol, smiles)
        if not isValidSmiles:
            error_message = message 
            return render_template('CeliacInformatics.html', activity_result = error_message, smiles_input=smiles, molecular_formula = molecular_formula)
        
        isOrganic, message = check_organic(mol, smiles)
        if not isOrganic:
            error_message = message
            return render_template('CeliacInformatics.html', activity_result = error_message, smiles_input=smiles, molecular_formula = molecular_formula)
            
        isWeight, message = check_weight(mol, smiles)
        if not isWeight:
            error_message = message
            return render_template('CeliacInformatics.html', activity_result = error_message, smiles_input=smiles, molecular_formula = molecular_formula)
        

        lipinski, color_lip = lipinski_report(smiles)
        muegge, color_mue = muegge_report(smiles)
        ghose, color_gho = ghose_report(smiles)
        veber, color_veb = veber_report(smiles)
        egan, color_eg = egan_report(smiles)
        violation_report, viol_report_color = check_num_violations(smiles)

        features, rdkbi = getFeatures(mol)
        activity_result, pred_color = predict_activity(features)
        isRanked = False
        if activity_result == "Active":
            isRanked =  True
        ic50_arr, smiles_arr = graph_spearman_ranking(smiles, mol)

        if viol_report_color == "red":
            activity_result = "Molecule is not Drug-Like"
            num_of_sub = 0
            sub_file_names = 0
            substructure_numbers = 0
            img_width = 0
            key_colors = 0
            isGraph = False
            isRanked = False
        else:
            sub_file_names, substructure_numbers, img_width, num_of_sub, key_colors = get_important_fingerprints(mol, rdkbi)
            graph_important_descriptors(smiles, mol, rdkbi)
            isGraph = True
        
        return render_template(
            'CeliacInformatics.html', 
            smiles_input=smiles, 
            activity_result = activity_result, 
            pred_color = pred_color, 
            sub_file_names = sub_file_names, 
            substructure_numbers = substructure_numbers,
            img_width = img_width,
            num_of_sub = num_of_sub,
            molecular_formula = molecular_formula,
            key_colors = key_colors,
            lipinski = lipinski,
            muegge = muegge,
            ghose = ghose,
            veber = veber,
            egan = egan,
            color_lip = color_lip,
            color_mue = color_mue,
            color_gho = color_gho,
            color_veb = color_veb,
            color_eg = color_eg,
            violation_report = violation_report,
            viol_report_color = viol_report_color,
            isGraph = isGraph,
            isRanked = isRanked
        )
    
    return render_template('CeliacInformatics.html')

# SubQAvail 
@app.route('/SubQAvail', methods=['GET', 'POST'])
def SubQAvail():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and allowed_file(file.filename):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

        mab_data = {
            'Name': request.form.get('mab_name', ''),
            'Heavy_Chain': request.form.get('heavy_chain', ''),
            'Light_Chain': request.form.get('light_chain', '')
        }
        filepath = write_to_csv(mab_data, 'input_data.csv')
         
        try:
            
            predictions_path = SubQAvail_process_file(filepath) 

            with open(predictions_path, 'r', newline='') as csvfile:
                reader = csv.reader(csvfile)
                predictions_data = list(reader)
 
            return render_template('SubQAvail.html',  
                                   predictions_data=predictions_data,  
                                   predictions_path=os.path.basename(predictions_path))
        except Exception as e:
            flash(f'Error processing file: {e}')
            return redirect(request.url)
            
    return render_template('SubQAvail.html')

if __name__ == '__main__':
    app.run(debug=True)


