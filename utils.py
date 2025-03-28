import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing, metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

eps = torch.tensor(1.e-8)

# ------------------------------------------------------------------------------
# Data Loading and Preprocessing Functions
# ------------------------------------------------------------------------------

def load_data(file_path, names=None):
    """
    Load dataset from a CSV file.
    """
    data = pd.read_csv(file_path, names=names)
    return data

def one_hot_encoding(data, cols):
    """
    Perform one-hot encoding on specified categorical columns.
    """
    encoded_data = pd.DataFrame()
    for col in cols:
        dummies = pd.get_dummies(data[col], prefix=col)
        encoded_data = pd.concat([encoded_data, dummies], axis=1)
    data.drop(cols, axis=1, inplace=True)
    data = pd.concat([data, encoded_data], axis=1)
    return data

def label_encoding(data, cols):
    """
    Perform label encoding on specified categorical columns.
    """
    encoded_data = pd.DataFrame()
    encoder = preprocessing.LabelEncoder()
    for col in cols:
        encoded_col = encoder.fit_transform(data[col])
        encoded_data = pd.concat([encoded_data, pd.DataFrame(encoded_col, columns=[col])], axis=1)
    data.drop(cols, axis=1, inplace=True)
    data = pd.concat([data, encoded_data], axis=1)
    return data

def get_labels(data, name):
    """
    Extract and encode labels based on the dataset type.
    For 'IDS2018': map 'Benign' to 0 and others to 1.
    For 'kdd': map "normal" to 0 and others to 1.
    """
    if name == 'IDS2018':
        label = data['Label']
        label = np.where(label == "Benign", 0, 1)
        data.drop(['Label'], axis=1, inplace=True)
    elif name == 'kdd':
        label = data[41]
        label = np.where(label == "normal", 0, 1)
        data.drop([41], axis=1, inplace=True)
    elif name == 'USNW_NB15':
        label = data['label']
        label = np.where(label == 0, 0, 1)
        data.drop(['label'], axis=1, inplace=True)
    else:
        label = None
    return label

# ------------------------------------------------------------------------------
# Evaluation Metrics
# ------------------------------------------------------------------------------

def get_scores(y_pred, y):
    """
    Compute common classification metrics: accuracy, precision, recall, F1 score.
    """
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='binary')
    recall = recall_score(y, y_pred, average='binary')
    f1 = f1_score(y, y_pred, average='binary')
    return accuracy, precision, recall, f1

def get_confusion_matrix(y_pred, y):
    """
    Returns the confusion matrix components: TN, FP, FN, TP.
    """
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    return tn, fp, fn, tp

# ------------------------------------------------------------------------------
# Data Normalization and Column Operations
# ------------------------------------------------------------------------------

def normalize_cols(data, scaler=None):
    """
    Normalize columns of the DataFrame using MinMaxScaler.
    If a scaler is provided, use it; otherwise, fit a new one.
    """
    if data.empty:
        return data, scaler
    
    if scaler is None:
        scaler = MinMaxScaler().fit(data)
    
    data_scaled = scaler.transform(data)
    return pd.DataFrame(data_scaled, columns=data.columns), scaler

def merge_cols(data1, data2):
    """
    Concatenate two DataFrames column-wise.
    """
    return pd.concat([data1, data2], axis=1)

def remove_cols(data, cols):
    """
    Remove columns from the DataFrame based on their index positions.
    """
    col_names = data.columns[cols]
    data.drop(col_names, axis=1, inplace=True)
    return data

def split_data(data, y, split=0.7):
    """
    Split data and labels into training and testing sets based on a given ratio.
    """
    split_index = int(split * len(data))
    train = data.iloc[:split_index]
    test = data.iloc[split_index:]
    y_train = y[:split_index]
    y_test = y[split_index:]
    return train, test, y_train, y_test

def fill_na(data):
    """
    Fill NaN values in the DataFrame with zeros.
    """
    data = data.fillna(value=0, axis=1)
    return data

# ------------------------------------------------------------------------------
# Utility Functions for Model Inputs
# ------------------------------------------------------------------------------

def relative_euclidean_distance(x1, x2, eps=eps):
    """
    Compute the relative Euclidean distance between two tensors.
    """
    num = torch.norm(x1 - x2, p=2, dim=1)
    denom = torch.norm(x1, p=2, dim=1)
    return num / torch.max(denom, eps)

def cosine_similarity(x1, x2, eps=eps):
    """
    Compute the cosine similarity between two tensors.
    """
    dot_prod = torch.sum(x1 * x2, dim=1)
    norm1 = torch.norm(x1, p=2, dim=1)
    norm2 = torch.norm(x2, p=2, dim=1)
    return dot_prod / torch.max(norm1 * norm2, eps)

def normalize_tuple(x, norm_val):
    """
    Normalize a tuple of two values by a given normalization value.
    """
    a, b = x
    return (a / norm_val, b / norm_val)

def process_features(data, categorical_cols, features, embed):
    """
    Selects features based on type and applies encoding if required.
    - 'categorical' selects only categorical features.
    - 'numerical' removes categorical features.
    - Embedding can be one-hot or label encoding.
    """
    if features == "categorical":
        data = data[categorical_cols]
    elif features == "numerical":
        data = remove_cols(data, categorical_cols)
    if embed == 'one_hot':
        data = one_hot_encoding(data, categorical_cols)
    elif embed == 'label_encode':
        data = label_encoding(data, categorical_cols)
    return data