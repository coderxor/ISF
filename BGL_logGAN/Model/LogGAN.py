import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class G_Model(nn.Module):
    def __init__(self,input_size, hidden_size, num_layers, num_keys,emb_dimension):
        super(G_Model,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_keys=num_keys
        self.emb_dimension=emb_dimension
        self.emb = nn.Embedding(num_keys, emb_dimension)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_keys)

    def forward(self,input_tensor,train_flag):
        if train_flag:
            h0 = torch.randn(self.num_layers, input_tensor.size(0), self.hidden_size).to(device)
            c0 = torch.randn(self.num_layers, input_tensor.size(0), self.hidden_size).to(device)
        else:
            h0 = torch.zeros(self.num_layers, input_tensor.size(0), self.hidden_size).to(device)
            c0 = torch.zeros(self.num_layers, input_tensor.size(0), self.hidden_size).to(device)
        input_tensor=input_tensor.to(device)
        input_tensor=self.emb(input_tensor)
        out, _ = self.lstm(input_tensor, (h0, c0))
        out = self.fc(out[:, -1, :])
        out=F.sigmoid(out)
        return out



class D_Model(nn.Module):
    def __init__(self,input_size, hidden_size, num_layers, num_keys,emb_dimension):
        super(D_Model,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_keys = num_keys
        self.emb_dimension = emb_dimension
        self.emb = nn.Embedding(num_keys, emb_dimension)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, num_keys)
        self.fc2 = nn.Linear(2*num_keys, 100)
        self.fc3=nn.Linear(100,1)

    def forward(self, input_tensor,label):
        h0 = torch.zeros(self.num_layers, input_tensor.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, input_tensor.size(0), self.hidden_size).to(device)
        input_tensor = input_tensor.to(device)
        input_tensor = self.emb(input_tensor)
        out, _ = self.lstm(input_tensor, (h0, c0))
        out = self.fc1(out[:, -1, :])
        out = F.sigmoid(out)
        out=torch.cat((out,label),1)
        out=F.relu(self.fc2(out))
        out=F.sigmoid(self.fc3(out))
        return out


