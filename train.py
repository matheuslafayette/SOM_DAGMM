import os
import numpy as np
import torch
import pandas as pd
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

from utils import get_scores
from som import som_train
from models import (CompressionNetwork, EstimationNetwork, 
                   GMM, Mixture, DAGMM, SOM_DAGMM)

def prepare_dataloaders(data_dir, batch_size):
    train_data = pd.read_csv(f"{data_dir}/train_data.csv")
    val_data = pd.read_csv(f"{data_dir}/test_data.csv")
    Y_val = np.load(f"{data_dir}/Y_test.npy")

    train_tensor = torch.tensor(train_data.values.astype(np.float32))
    val_tensor = torch.tensor(val_data.values.astype(np.float32))
    
    train_loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=batch_size,
        shuffle=True
    )
    
    return train_loader, train_tensor, val_tensor, Y_val

def initialize_model(data_tensor):
    pretrained_som = som_train(
        data=data_tensor.numpy(), 
        x=10, y=10, sigma=1, 
        learning_rate=0.8, 
        iters=10000, 
        neighborhood_function='bubble'
    )
    compression = CompressionNetwork(data_tensor.shape[1])
    estimation = EstimationNetwork()
    gmm = GMM(2, 6)
    mix = Mixture(6)
    dagmm = DAGMM(compression, estimation, gmm)
    return SOM_DAGMM(dagmm, pretrained_som), compression, mix

def train_model(net, compression, mix, optimizer, 
               train_loader, val_tensor, Y_val, 
               epochs, save_path):
    best_val_score = -np.inf
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        error_count = 0
        
        for batch in train_loader:
            data = batch[0]
            optimizer.zero_grad()
            
            out = net(data)
            L_loss = compression.reconstruction_loss(data)
            G_loss = mix.gmm_loss(out=out, L1=0.1, L2=0.005)
            loss = L_loss + G_loss
            
            loss.backward()
            
            optimizer.step()
            
            running_loss += loss.item() if torch.isfinite(loss) else 0
            error_count += 0 if torch.isfinite(loss) else 1

        print(f"Epoch {epoch+1}: Loss={running_loss:.4f}, Errors={error_count}")
        
        if (epoch+1) % 5 == 0:
            net.eval()
            with torch.no_grad():
                out_val = net(val_tensor)
            
            threshold = np.percentile(out_val.cpu().numpy(), 20)
            pred_val = (out_val.cpu().numpy() > threshold).astype(int)
            a, p, r, f = get_scores(pred_val, Y_val)
            print("Validation Accuracy:", a, "Precision:", p, "Recall:", r, "F1 Score:", f)
            
            if f > best_val_score:
                best_val_score = f
                torch.save(net.state_dict(), f"{save_path}/best.pt")
                print(f"New best model saved with F1={f:.4f}")

def main():
    config = {
        "epochs": 50,
        "batch_size": 1024,
        "data_dir": "processed_data/",
        "save_path": "save/",
        "learning_rate": 0.0001
    }
    
    train_loader, train_tensor, val_tensor, Y_val = prepare_dataloaders(
        config['data_dir'], 
        config['batch_size']
    )

    unique, counts = np.unique(Y_val, return_counts=True)
    percentages = counts / counts.sum() * 100

    print("Class distribution in Y_val:")
    for cls, pct, count in zip(unique, percentages, counts):
        print(f"Class {cls}: {pct:.2f}% ({count} samples)")
    
    net, compression, mix = initialize_model(train_tensor)
    
    optimizer = optim.Adam(net.parameters(), lr=config['learning_rate'])
    
    os.makedirs(config['save_path'], exist_ok=True)
    train_model(
        net, compression, mix, optimizer,
        train_loader, val_tensor, Y_val,
        config['epochs'], config['save_path']
    )

if __name__ == '__main__':
    main()