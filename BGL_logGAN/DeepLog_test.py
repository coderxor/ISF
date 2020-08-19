import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from Model.DeepLog import DeepLog_Model
import csv
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
h_window=3
hidden_size=128
num_layers=2
num_keys=394
emb_dimension=200

def load_data(file,file_label):
    session_data = []
    f = open(file, 'r')
    reader = csv.reader(f)
    for line in reader:
        if len(line) < h_window + 1:
            continue
        data = []
        for v in line:
            v = int(v) - 1
            data.append(v)
        session_data.append(data)
    session_label = []
    f = open(file_label, 'r')
    reader = csv.reader(f)
    for line in reader:
        if len(line) < h_window + 1:
            continue
        data = []
        for v in line:
            v = int(v)
            data.append(v)
        session_label.append(data)

    test_tuple = {}
    for i in range(len(session_data)):
        session = session_data[i]
        for j in range(len(session) - h_window - 1):
            input = session[j:j + h_window]
            label = session_label[i][j + h_window + 1]
            target = session[j + h_window + 1]
            key = tuple(input + [target])
            test_tuple.setdefault(key, {'frequence': 0, 'normal': 0, 'anormal': 0})
            test_tuple[key]['frequence'] += 1
            if label:
                test_tuple[key]['normal'] += 1
            else:
                test_tuple[key]['anormal'] += 1

    return test_tuple



def Test(test_tuple,train_tuple,flag):
    with torch.no_grad():
        model = DeepLog_Model(emb_dimension, hidden_size, num_layers, num_keys, emb_dimension).cuda()
        model.load_state_dict(torch.load('Checkpoint/DeepLog_4.pkl'))
        print('模型初始化完成')
        topK = 10
        TP = []
        FP = []
        FN = []
        TN = []
        precision = []
        recall = []
        for i in range(topK):
            TP.append(0)
            FP.append(0)
            FN.append(0)
            TN.append(0)
            precision.append(0)
            recall.append(0)
        for key in test_tuple.keys():
            if flag:
                if key not in train_tuple:
                    continue
            else:
                if key in train_tuple:
                    continue
            data = list(key)
            input = data[0:h_window]
            target = data[-1]
            input_tensor = Variable(torch.LongTensor([input]))
            y = model(input_tensor)
            y = y.view(-1)
            y = y.cpu()
            _, index = torch.topk(y, topK)
            index = index.data.numpy()
            index = index.tolist()
            if target not in index:
                for k in range(topK):
                    FP[k] += test_tuple[key]['normal']

                for k in range(topK):
                    TP[k] += test_tuple[key]['anormal']
            else:
                tmp_index = index.index(target)
                for k in range(topK):
                    if k < tmp_index:
                        FP[k] += test_tuple[key]['normal']
                        TP[k] += test_tuple[key]['anormal']
                    else:
                        TN[k] += test_tuple[key]['normal']
                        FN[k] += test_tuple[key]['anormal']

        F1 = []
        TNR = []
        for k in range(len(TP)):
            precision[k] = float(TP[k]) / max((TP[k] + FP[k]), 1)
            recall[k] = float(TP[k]) / max((TP[k] + FN[k]), 1)
            F1.append(2 * precision[k] * recall[k] / max((precision[k] + recall[k]), 0.001))
            TNR.append(float(TN[k]) / max((TN[k] + FN[k]), 1))

    print(precision)
    print(recall)
    print(F1)
    print(TNR)


if __name__=="__main__":
    train_data_file='Dataset/train_data_tmp.csv'
    train_label_file = 'Dataset/train_data_label_tmp.csv'
    test_data_file = 'Dataset/test_data_session_tmp.csv'
    test_label_file = 'Dataset/test_data_label_tmp.csv'

    train_tuple=load_data(train_data_file,train_label_file)
    test_tuple=load_data(test_data_file,test_label_file)

    Test(test_tuple,train_tuple,flag=True)

    Test(test_tuple, train_tuple, flag=False)

