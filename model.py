import torch
import torch.nn as nn

class FCNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(FCNetwork, self).__init__()
        self.L1 = nn.Linear(input_dim, hidden_dim1)
        self.dropout = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()
        self.L2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.L3 = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        x = self.L1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.L2(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.L3(x)  # No softmax as loss is crossentropy
        return x
    

# class RNN(nn.Module):
#     def __init__(self, input_size, hidden_size) -> None:
#         super(RNN, self).__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.LSTM = nn.LSTM()
#         self.l1 = nn.Linear()