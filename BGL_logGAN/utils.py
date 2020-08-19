import csv
import json
import itertools
import copy


def load_data(h_window):

    session_data = []
    f = open('Dataset/train_data_tmp.csv', 'r')
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
    f = open('Dataset/train_data_label_tmp.csv', 'r')
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

            if labels[j+h_window] == 1:
                train_data[context].setdefault(l, {'label': 1, 'Count': 0})
                train_data[context][l]['label'] = 1
                train_data[context][l]['Count'] = train_data[context][l]['Count'] + 1

            else:
                train_data[context].setdefault(l, {'label': 0, 'Count': 0})
                train_data[context][l]['label'] = 0
                train_data[context][l]['Count'] = train_data[context][l]['Count'] + 1

    print('数据加载完成')
    return  train_data


def Permutation(train_data_temp):
    train_data=copy.deepcopy(train_data_temp)
    for patten in train_data_temp:
        listO1=list(itertools.permutations(patten,len(patten)))[1:]
        dict_temp=train_data[patten]

        for new_patten in listO1:
            if new_patten in train_data:
                label1=list(train_data[new_patten].keys())  # 原有的
                label2=list(dict_temp.keys())    #准备新增的
                for l in label2:
                    if l in label1:
                        if train_data[patten][l]['label']==train_data[new_patten][l]['label']:
                            train_data[patten][l]['Count']=train_data[patten][l]['Count']+1
                        else:
                            print(patten)

                    else:
                        if dict_temp[l]['label'] == 1:
                            train_data[new_patten].setdefault(l, {'label': 1, 'Count': 0})
                            train_data[new_patten][l]['label'] = 1
                            train_data[new_patten][l]['Count'] = train_data[new_patten][l]['Count'] + 1

                        else:
                            train_data[new_patten].setdefault(l, {'label': 0, 'Count': 0})
                            train_data[new_patten][l]['label'] = 0
                            train_data[new_patten][l]['Count'] = train_data[new_patten][l]['Count'] + 1
            else:
                train_data.setdefault(new_patten, dict_temp)

    return  train_data












