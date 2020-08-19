import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from Model.LogGAN_mask import G_Model,D_Model

def Data(file,file_label):
    session_data = []
    f = open(file, 'r')
    reader = csv.reader(f)
    for line in reader:
        data = []
        if len(line) < h_window + 1:
            continue
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

            label = session_label[i][j + h_window]
            target = session[j + h_window ]
            if input==[36,36,361] and target==210:
                index="break"
            key = tuple(input + [target])
            test_tuple.setdefault(key, {'frequence': 0, 'normal': 0, 'anormal': 0})
            test_tuple[key]['frequence'] += 1
            if label:
                test_tuple[key]['normal'] += 1
            else:
                test_tuple[key]['anormal'] += 1

    return test_tuple

def load_data(h_window,data_file,data_label):

    session_data = []
    f = open(data_file, 'r')
    reader = csv.reader(f)
    for line in reader:
        if len(line) < h_window + 1:
            continue
        data = []
        for v in line:
            v = int(v) - 1
            data.append(v)
        session_data.append(data)

    session_data_label = []
    f = open(data_label, 'r')
    reader = csv.reader(f)
    for line in reader:
        if len(line) < h_window + 1:
            continue
        data = []
        for v in line:
            v = int(v)
            data.append(v)
        session_data_label.append(data)

    train_data = {}

    for i in range(len(session_data)):
        session = session_data[i]
        labels = session_data_label[i]
        for j in range(len(session) - h_window):
            context = session[j:j + h_window]
            context = tuple(context)
            l = session[j + h_window]
            train_data.setdefault(context, {})

            if labels[j] == 1:
                train_data[context].setdefault(l, {'label': 1, 'Count': 0})
                train_data[context][l]['label'] = 1
                train_data[context][l]['Count'] = train_data[context][l]['Count'] + 1

            else:
                train_data[context].setdefault(l, {'label': 0, 'Count': 0})
                train_data[context][l]['label'] = 0
                train_data[context][l]['Count'] = train_data[context][l]['Count'] + 1

    print('数据加载完成')
    return  train_data

def divide():
    Test_see = []
    Test_notsee = []
    for patten in test_dataset:
        if patten in train_dataset:
            # patten_pre = test_dataset[i]
            target_test = list(test_dataset[patten].keys())
            tartet_train = list(train_dataset[patten].keys())

            for t in target_test:
                if t in tartet_train:
                    temp1 = []
                    temp1.append(list(patten))
                    temp1.append(t)
                    temp1.append(test_dataset[patten][t]['label'])
                    temp1.append(test_dataset[patten][t]['Count'])
                    Test_see.append(temp1)
                else:
                    temp2 = []
                    temp2.append(list(patten))
                    temp2.append(t)
                    temp2.append(test_dataset[patten][t]['label'])
                    temp2.append(test_dataset[patten][t]['Count'])
                    Test_notsee.append(temp2)

        else:

            target_test = list(test_dataset[patten].keys())
            for t in target_test:
                temp3 = []
                temp3.append(list(patten))
                temp3.append(t)
                temp3.append(test_dataset[patten][t]['label'])
                temp3.append(test_dataset[patten][t]['Count'])
                Test_notsee.append(temp3)

    print(len(Test_see))
    Count1=0
    Count2=0
    for i in Test_see:
        if i[2]==0:
            Count1+=1
        else:
            Count2+=1
    print(Count1,Count2)

    print(len(Test_notsee))
    Count1=0
    Count2=0
    for i in Test_notsee:
        if i[2]==0:
            Count1+=1
        else:
            Count2+=1
    print(Count1,Count2)

    rate = len(Test_see) / len(Test_notsee)

    print(rate)

    return Test_see,Test_notsee

def Test(Flag,Test_tuple,Train_tuple):
    if Flag:
        with torch.no_grad():
            model = G_Model(emb_dimension, hidden_size, num_layers, num_keys, emb_dimension).cuda()
            model.load_state_dict(torch.load('Checkpoint/G_model_mask_tmp.pkl'))
            save(model, Test_tuple,Train_tuple,True, "Result/LogGan_Testsee.csv")
            #save(model, Test_tuple,Train_tuple,False, "Result/LogGan_Testnosee.csv")

    else:
        with torch.no_grad():
            model = G_Model(emb_dimension, hidden_size, num_layers, num_keys, emb_dimension).cuda()
            model.load_state_dict(torch.load('Checkpoint/DeepLog_3.pkl'))
            save(model, Test_tuple,Train_tuple,True, "Result/Deeplog_Testsee.csv")

            save(model, Test_tuple,Train_tuple,False, "Result/Deeplog_Testnosee.csv")


def save(model,test_tuple,Train_data,flag,savefile):
    print(len(test_tuple))
    TP = []
    FP = []
    FN = []
    TN = []
    precision = []
    recall = []
    tt=0
    for i in range(len(threshold)):
        TP.append(0)
        FP.append(0)
        FN.append(0)
        TN.append(0)
        precision.append(0)
        recall.append(0)
    for key in test_tuple.keys():

        if flag:
            if key not in Train_data:
                continue
        else:
            if key in Train_data:
                continue
        data = list(key)
        input = data[0:h_window]
        target = data[-1]



        input_tensor = Variable(torch.LongTensor([input]))
        input_tensor.cuda()
        y = model(input_tensor, train_flag)
        y = y.view(-1)
        y = y.cpu()
        r = float(y[target])

        # if test_tuple[key]['anormal']!=0:
        #     if Train_data[key]['anormal'] == 0 and test_tuple[key]['anormal'] != 0:
        #             index = "break"
        #             continue
        #     elif Train_data[key]['anormal'] != 0 and test_tuple[key]['anormal'] != 0:
        #         index='break'
        if Train_data[key]['anormal'] == 0 and test_tuple[key]['anormal'] != 0 or Train_data[key]['anormal'] != 0 and test_tuple[key]['anormal'] == 0:
            index="break"



        for k in range(len(threshold)):
            if r >= threshold[k]:
                if test_tuple[key]['anormal']!=0:
                    print(key,r )
                FN[k] += test_tuple[key]['anormal']
                TN[k] += test_tuple[key]['normal']
            else:

                # if test_tuple[key]['normal']!=0:
                #     print(key, r)
                TP[k] += test_tuple[key]['anormal']
                FP[k] += test_tuple[key]['normal']
        tt+=1
    print(tt)
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
    print("***************")


if __name__=="__main__":
    hidden_size = 128
    num_layers = 3
    h_window=2
    num_keys = 394
    emb_dimension = 200
    topK = 10
    train_flag = False
    threshold = [1, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.40, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1]
    torch.cuda.set_device(1)


    train_data_file='Dataset/train_data_tmp.csv'
    train_label_file = 'Dataset/train_data_label_tmp.csv'
    test_data_file = 'Dataset/test_data_session_tmp.csv'
    test_label_file = 'Dataset/test_data_label_tmp.csv'
    train_dataset=load_data(h_window,train_data_file,train_label_file)
    test_dataset=load_data(h_window,test_data_file,test_label_file)
    Test_see,Test_notsee=divide()

    normal=0
    abnormal=0
    for patten in train_dataset:
        for key in train_dataset[patten]:
            if train_dataset[patten][key]['label']==0:
                abnormal+=1
            else:
                normal+=1
    print(normal)
    print(abnormal)


    # Traind_tuple=Data(train_data_file,train_label_file)
    #
    # Test_tuple=Data(test_data_file,test_label_file)
    #
    # Test(True,Test_tuple,Traind_tuple)

    #Test(False, Test_t uple, Traind_tuple)





























