import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from Model.DeepLog import DeepLog_Model
from utils import load_data
import csv
import random


def fenge(listTemp, n):
    res=[]
    for i in range(0, len(listTemp), n):
        res.append(listTemp[i:i + n])
    return res

def train(h_window):

    Sessions = list(train_data.keys())


    for session in Sessions :
        train_loss = 0
        patten=train_data[session].keys()
        input_tensor = torch.zeros(1, len(session), 394)
        for k in range(len(session)):
            input_tensor[0][k][session[k]]=1

        input_tensor = Variable(torch.LongTensor([session]))

        for j in patten:
            label = torch.LongTensor([j]).cuda()
            output = model(input_tensor)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        print(train_loss / len(patten))


    print(str(h_window) + "训练完成")
    filename = 'Checkpoint/DeepLog_' + str(h_window) + '.pkl'
    torch.save(model.state_dict(), filename)



if __name__=="__main__":
    h_window = 4
    hidden_size = 128
    num_layers = 2
    num_keys = 394
    emb_dimension = 200
    batch_size=30
    torch.cuda.set_device(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model = DeepLog_Model(emb_dimension, hidden_size, num_layers, num_keys, emb_dimension).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    train_data=load_data(h_window)

    train(h_window)



