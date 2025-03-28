import numpy as np
from minisom import MiniSom

def som_train(data, x=10, y=10, sigma=1, learning_rate=0.8, iters=10000, neighborhood_function='bubble'):
    """
    Trains a Self-Organizing Map (SOM) using the MiniSom library.
    
    Parameters:
      - data: Input data array.
      - x, y: Dimensions of the SOM grid.
      - sigma: Spread of the neighborhood function.
      - learning_rate: Learning rate for weight updates.
      - iters: Number of iterations for training.
      - neighborhood_function: Function to determine neighborhood influence.
      
    Returns:
      - A trained SOM model with learned spatial representation of the data.
    """
    input_len = data.shape[1]
    print("SOM training started:")
    som = MiniSom(x=x, y=y, input_len=input_len, sigma=sigma, learning_rate=learning_rate, neighborhood_function=neighborhood_function)
    som.random_weights_init(data)
    som.train_random(data, iters)
    return som

def som_pred(som_model, data, outlier_percentage):
    """
    Detects anomalies based on quantization error thresholds derived from the SOM.
    
    Parameters:
      - som_model: Trained SOM model.
      - data: Input data (as a tensor) to evaluate.
      - outlier_percentage: Proportion of data expected to be outliers.
      
    Returns:
      - y_pred: An array indicating anomalies (1 for anomaly, 0 for normal).
    """
    data_np = data.numpy()
    quantization_errors = np.linalg.norm(som_model.quantization(data_np) - data_np, axis=1)
    error_threshold = np.percentile(quantization_errors, 100 * (1 - outlier_percentage) + 5)
    is_anomaly = quantization_errors > error_threshold
    y_pred = np.multiply(is_anomaly, 1)
    return y_pred
