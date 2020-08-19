import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from Model.LogGAN_mask import G_Model, D_Model
import csv
import os
import random
import math
from utils import load_data,Permutation


def fenge(listTemp, n):
    res=[]
    for i in range(0, len(listTemp), n):
        res.append(listTemp[i:i + n])
    return res

def negative_sampling(listTemp,negative_rate,num_all):
    candidate_list=[i for i in range(num_all)]
    candidate=set(candidate_list)
    candidate=candidate-set(listTemp)
    candidate=list(candidate)
    res=listTemp
    #n=int(len(candidate)*negative_rate)
    n=5
    while(n):
        i=random.randint(0,len(candidate)-1)
        tmp=candidate[i]
        candidate.pop(i)
        res.append(tmp)
        n-=1
    return res

def positive_sampling(listTemp,positive_rate,num_all):
    candidate_list = [i for i in range(num_all)]
    candidate = set(candidate_list)
    candidate = candidate - set(listTemp)
    candidate = list(candidate)
    res = []
    n=int(len(candidate)*positive_rate)
    #n = 5
    while (n):
        i = random.randint(0, len(candidate) - 1)
        tmp = candidate[i]
        candidate.pop(i)
        res.append(tmp)
        n -= 1
    return res





def train(num_epochs):
    for epoch in range(num_epochs):
        # minibatch
        # minibatch
        Sessions = list(train_data.keys())
        random.shuffle(Sessions)
        G_Session = fenge(Sessions, batch_size)
        random.shuffle(Sessions)
        D_Session = fenge(Sessions, batch_size)
        G_train_loss = []
        D_train_loss = []
        for i in range(len(G_Session)):  # minibatch
            # G_step
            g_session = G_Session[i]
            input_tensor = Variable(torch.LongTensor(g_session))
            fake_pro = generate_model(input_tensor, train_flag)
            mask = torch.zeros(len(g_session), num_keys).cuda()
            PS = []
            for j in range(len(g_session)):
                label = list(train_data[g_session[j]].keys())
                # label=negative_sampling(label,negative_rate,num_keys)
                for k in label:
                    mask[j][k] = 1.0
                ps = positive_sampling(label, positive_rate, num_keys)
                PS.append(ps)
                for k in ps:
                    mask[j][k] = 1.0
            # mask2 = torch.zeros(len(g_session), num_keys).cuda()#add to construction error
            real_pro = torch.zeros(len(g_session), num_keys).cuda()
            for j in range(len(g_session)):
                label = list(train_data[g_session[j]].keys())
                for k in label:
                    # mask2[j][k]=1.0
                    if train_data[g_session[j]][k]['label'] == 1:
                        real_pro[j][k] = 1.0
                ps = positive_sampling(label, positive_rate, num_keys)
                for k in ps:
                    # mask2[j][k] = 1.0
                    real_pro[j][k] = 1.0

            fake_D_1 = discriminate_model(input_tensor, torch.mul(fake_pro, mask))
            tmp = torch.zeros(len(g_session), 1).cuda()
            # G_loss = -1*criterion(fake_D_1, tmp)+alpha*MSE(torch.mul(fake_pro,mask2),real_pro)/float(real_pro.size()[0])
            G_loss = -1 * criterion(fake_D_1, tmp) + alpha * MSE(fake_pro, real_pro)
            G_train_loss.append(G_loss.item())
            G_optimizer.zero_grad()
            G_loss.backward()
            G_optimizer.step()

            # D_step
            # for i in range(len(D_Session)):
            d_session = D_Session[i]
            input_tensor = Variable(torch.LongTensor(d_session))
            fake_pro = generate_model(input_tensor, train_flag)
            mask = torch.zeros(len(d_session), num_keys).cuda()
            for j in range(len(d_session)):
                label = list(train_data[d_session[j]].keys())
                # label = negative_sampling(label, negative_rate, num_keys)
                for k in label:
                    mask[j][k] = 1.0
                    # ps=positive_sampling(label,positive_rate,num_keys)
                    # PS.append(ps)
                    # for k in ps:
                    #     mask[j][k] = 1.0
            fake_D_2 = discriminate_model(input_tensor, torch.mul(fake_pro.detach(), mask))
            real_pro = torch.zeros(len(d_session), num_keys).cuda()
            for j in range(len(d_session)):
                label = list(train_data[d_session[j]].keys())
                for k in label:
                    if train_data[d_session[j]][k]['label'] == 1:
                        real_pro[j][k] = 1.0
                        # for k in PS[j]:
                        #     real_pro[j][k] = 1.0
            real_pro = Variable(real_pro)
            real_D = discriminate_model(input_tensor, torch.mul(real_pro, mask))
            tmp1 = torch.zeros(len(d_session), 1).cuda() + 1
            tmp2 = torch.zeros(len(d_session), 1).cuda()

            D_loss = criterion(real_D, tmp1) + criterion(fake_D_2, tmp2)
            D_train_loss.append(D_loss.item())
            D_optimizer.zero_grad()
            D_loss.backward()
            D_optimizer.step()

        print('epoch:%d  G_loss:%f' % (epoch,(sum(G_train_loss) / len(G_train_loss))),
              'D_loss:%f' % (sum(D_train_loss) / len(D_train_loss)))

    print("训练完成")


if __name__=="__main__":
    h_window = 3
    hidden_size = 128
    num_layers = 2
    num_keys = 394
    emb_dimension = 200
    train_flag = True
    num_epochs = 300
    batch_size = 50
    negative_rate = 0.3
    positive_rate = 0.1
    alpha = 0.5
    torch.cuda.set_device(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data = load_data(h_window)
    print(len(train_data))
    train_data =Permutation(train_data)
    print(len(train_data))

    generate_model = G_Model(emb_dimension, hidden_size, num_layers, num_keys, emb_dimension).cuda()
    discriminate_model = D_Model(emb_dimension, hidden_size, num_layers, num_keys, emb_dimension).cuda()
    print('模型加载完成')
    criterion = nn.BCELoss()
    # MSE=nn.MSELoss(size_average=False)
    MSE = nn.MSELoss()
    G_optimizer = optim.Adam(generate_model.parameters())
    D_optimizer = optim.Adam(discriminate_model.parameters())
    cnt = 0
    train(num_epochs)

    filename1 = 'Checkpoint/G_model_mask_tmp_window4_Permutation.pkl'
    torch.save(generate_model.state_dict(), filename1)
    filename2 = 'Checkpoint/D_model_mask_tmp_window4_Permutation.pkl'
    torch.save(discriminate_model.state_dict(), filename2)





