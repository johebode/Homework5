import pandas as pd
import os
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from rdkit import Chem
from rdkit.Chem.AllChem import GetMorganGenerator, GetRDKitFPGenerator
from rdkit.Chem import MACCSkeys
import argparse

# Set up argument parser for hyperparameters
def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a fingerprint-based regression model")
    parser.add_argument('--max_iter', type=int, default=1000, help='Maximum number of iterations for MLPRegressor')
    parser.add_argument('--hidden_layer_sizes', type=int, nargs='+', default=(100,), help='Number of neurons in each hidden layer')
    return parser.parse_args()

# Get hyperparameters from command line arguments
args = parse_arguments()

# Fingerprint generator setup
mfgen = GetMorganGenerator(radius=2, fpSize=1024)
data_path = 'Lipophilicity.csv'  # Update path to your dataset

# Read the dataset
esol_data = pd.read_csv(data_path)
print(os.path.exists(data_path))
train_data, test_data = train_test_split(esol_data, test_size=0.2)

# Feature and target variables
X_train = train_data["smiles"]
y_train = train_data["exp"]
X_test = test_data["smiles"]
y_test = test_data["exp"]

# Morgan Fingerprint features for training and testing
train_fp1 = [mfgen.GetFingerprint(Chem.MolFromSmiles(s)) for s in X_train]
test_fp1 = [mfgen.GetFingerprint(Chem.MolFromSmiles(s)) for s in X_test]

# MACCS Fingerprint features for training and testing
train_fp2 = [MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(s)) for s in X_train]
test_fp2 = [MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(s)) for s in X_test]

# Convert to numpy arrays
train_fp1 = np.array(train_fp1)
test_fp1 = np.array(test_fp1)
train_fp2 = np.array(train_fp2)
test_fp2 = np.array(test_fp2)

# Scale the target values
scaler = StandardScaler()
y_train_scaled = scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()

# Train the models with user-defined hyperparameters
morgan_model = MLPRegressor(random_state=42, max_iter=args.max_iter, hidden_layer_sizes=tuple(args.hidden_layer_sizes))
maccs_model = MLPRegressor(random_state=42, max_iter=args.max_iter, hidden_layer_sizes=tuple(args.hidden_layer_sizes))

# Fit the models
morgan_model.fit(train_fp1, y_train_scaled)
maccs_model.fit(train_fp2, y_train_scaled)

# Make predictions
y_pred_morgan_scaled = morgan_model.predict(test_fp1)
y_pred_maccs_scaled = maccs_model.predict(test_fp2)

# Inverse scale the predictions
y_pred_morgan = scaler.inverse_transform(y_pred_morgan_scaled.reshape(-1, 1)).flatten()
y_pred_maccs = scaler.inverse_transform(y_pred_maccs_scaled.reshape(-1, 1)).flatten()

# Calculate RMSE
rmse_morgan = np.sqrt(mean_squared_error(y_test, y_pred_morgan))
rmse_maccs = np.sqrt(mean_squared_error(y_test, y_pred_maccs))

# Print RMSE results
print(f'RMSE for Morgan Fingerprints: {rmse_morgan}')
print(f'RMSE for MACCS Keys: {rmse_maccs}')

# Save results to a text file
env_name = os.getenv("CONDA_DEFAULT_ENV")
with open("results.txt", "w") as f:
    f.write(f"RMSE Morgan: {rmse_morgan}\n")
    f.write(f"RMSE MACCS: {rmse_maccs}\n")
    f.write(f"Conda Environment: {env_name}\n")
    f.write(f"Hyperparameters:\n")
    f.write(f"  max_iter: {args.max_iter}\n")
    f.write(f"  hidden_layer_sizes: {args.hidden_layer_sizes}\n")
