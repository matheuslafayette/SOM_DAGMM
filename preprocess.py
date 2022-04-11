import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import (load_data, get_labels, fill_na, 
                  normalize_cols, process_features)

SEED = 42

def preprocess_data(dataset_name, features, embed, output_dir):
    if dataset_name == 'kdd':
        names = list(range(43))
        data = load_data('data/KDDTrain+.txt', names)
        categorical_cols = [1, 2, 3, 4]
        dataB = data[data[41] == "normal"]
        dataM = data[data[41] != "normal"]
        YB = get_labels(dataB, dataset_name)
        YM = get_labels(dataM, dataset_name)
    elif dataset_name == 'IDS2018':
        data = load_data('data/CSE-CIC-IDS2018-25k.csv')
        categorical_cols = []
        dataB = data[data['Label'] == "Benign"]
        dataM = data[data['Label'] != "Benign"]
        YB = get_labels(dataB, dataset_name)
        YM = get_labels(dataM, dataset_name)
    elif dataset_name == 'IDS2019':
        data = load_data('data/CSE-CIC-IDS2019_23.csv')
        data = data.drop(['Flow ID', 'Source IP', 'Source Port', 'Destination IP',
                          'Destination Port', 'Protocol', 'Timestamp', 'SimillarHTTP'], axis='columns')
        categorical_cols = []
        dataB = data[data['Label'] == "BENIGN"]
        dataM = data[data['Label'] != "BENIGN"]
        YB = get_labels(dataB, dataset_name)
        YM = get_labels(dataM, dataset_name)
    else:
        raise ValueError("Invalid dataset name")

    def _base_process(df):
        df = process_features(df, categorical_cols, features, embed)
        df = fill_na(df.replace([np.inf, -np.inf], np.nan))
        return df

    # Processa os dados
    dataM = _base_process(dataM)
    dataB = _base_process(dataB)

    # Divisão dos dados benignos
    train_data, temp_data, Y_train, Y_temp = train_test_split(
        dataB, YB, test_size=0.5, random_state=SEED
    )
    val_benign, test_benign, Y_val_benign, Y_test_benign = train_test_split(
        temp_data, Y_temp, test_size=0.5, random_state=SEED
    )

    # Divisão dos dados malignos
    val_malignant, test_malignant, Y_val_malignant, Y_test_malignant = train_test_split(
        dataM, YM, test_size=0.5, random_state=SEED
    )

    # Normalização
    train_data_normalized, scaler = normalize_cols(train_data)
    val_benign_normalized, _ = normalize_cols(val_benign, scaler)
    test_benign_normalized, _ = normalize_cols(test_benign, scaler)
    val_malignant_normalized, _ = normalize_cols(val_malignant, scaler)
    test_malignant_normalized, _ = normalize_cols(test_malignant, scaler)

    # Criação dos conjuntos finais
    val_data = pd.concat([val_benign_normalized, val_malignant_normalized])
    Y_val = np.concatenate([Y_val_benign, Y_val_malignant])

    test_data = pd.concat([test_benign_normalized, test_malignant_normalized])
    Y_test = np.concatenate([Y_test_benign, Y_test_malignant])

    # Salva os dados
    os.makedirs(output_dir, exist_ok=True)
    train_data_normalized.to_csv(f"{output_dir}/train_data_kdd.csv", index=False)
    val_data.to_csv(f"{output_dir}/val_data_kdd.csv", index=False)
    test_data.to_csv(f"{output_dir}/test_data_kdd.csv", index=False)
    np.save(f"{output_dir}/Y_train_kdd.npy", Y_train)
    np.save(f"{output_dir}/Y_val_kdd.npy", Y_val)
    np.save(f"{output_dir}/Y_test_kdd.npy", Y_test)

if __name__ == '__main__':
    preprocess_data(
        dataset_name='kdd',
        features='numerical',
        embed='',
        output_dir='processed_data/'
    )