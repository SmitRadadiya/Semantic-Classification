import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchsummary import summary
from utils import get_data,get_tokenize_data,MakeData
from model import FCNetwork

device = torch.device('cuda' if  torch.cuda.is_available() else 'cpu' )

def training(data, model : FCNetwork, num_epoch = 10):
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    criterian = nn.CrossEntropyLoss()
    
    for epoch in range(num_epoch):
        model.train()
        for i, (x,y) in enumerate(data):
            x = x.to(device)
            y_hate = model(x)
            y = y.to(device)
            loss = criterian(y_hate, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
           
        print(f'Epoch: [{epoch+1}/{num_epoch}], Loss: {loss.item():.6f}')


def validation(data, model, total_samples):
    model.eval()
    acc = 0
    for _, (x,y) in enumerate(data):
        x = x.to(device)
        y = y.to(device)
        y_pred = model(x)
        y_pred_cls = torch.argmax(y_pred, dim=1)
        acc += y_pred_cls.eq(y).sum()

    acc = acc/total_samples
    print(acc)

def main():
    path = '../Data/ClassificationDataset-train0.xlsx'

    data = get_data(path)
    dim = 100
    data_vec = get_tokenize_data(data,dim)

    train_data = MakeData(data_vec, data['class'])

    train_loader = DataLoader(dataset=train_data, shuffle=True)

    input_dim = dim
    output_dim = 3
    hidden_dim1 = 32
    hidden_dim2 = 16
    model = FCNetwork(input_dim, hidden_dim1, hidden_dim2, output_dim).to(device)
    print("Model Summary :")
    print(summary(model, (1, 100), device=str(device).lower()))

    epoch = 10
    training(train_loader, model, num_epoch=epoch)

    test_dataset = get_data('../Data/ClassificationDataset-valid0.xlsx')
    test_data_vec = get_tokenize_data(test_dataset,dim)
    test_data = MakeData(test_data_vec, test_dataset['class'])
    test_loader = DataLoader(dataset=test_data)

    validation(test_loader, model, test_dataset.shape[0])


if __name__ == '__main__':
    main()
