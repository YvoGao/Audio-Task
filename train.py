import copy
import math
from sklearn.mixture import GaussianMixture
import torch
import os
import pandas as pd
from loader import ProcessDataset
import os
import random
import numpy as np
import datetime
import pytz
from tqdm import tqdm
import torch
import torch.nn as nn
from loader import FSC89Dataset, FSCILDataset, LBS100Dataset, Ny100Dataset, ProcessDataset
from utils import get_loss, get_parse, get_scores, print_scores, set_seed, setup_logging, get_record
import models
from pengi import pengi
import torch.nn.functional as F
from sklearn.preprocessing import normalize

class Record(object):
    def __init__(self, args):
        self.args = args
        self.old = []
        self.novel = []
        self.overall = []
    def print(self):
        print(f"*"*20+"Final Results"+"*"*20)
        str_head = "{:<9}".format('Metric')
        str_acc = "{:<9}".format("Acc")
        str_acc_old = "{:<9}".format("Old")
        str_acc_novel = "{:<9}".format("Novel")
        str_f1 = "{:<9}".format("F1")
        str_f1_old = "{:<9}".format("Old")
        str_f1_novel = "{:<9}".format("Novel")
        str_pre = "{:<9}".format("Pre")
        str_pre_old = "{:<9}".format("Old")
        str_pre_novel = "{:<9}".format("Novel")
        str_rec = "{:<9}".format("Rec")
        str_rec_old = "{:<9}".format("Old")
        str_rec_novel = "{:<9}".format("Novel")
        for i in range(self.args.session):
            str_head = str_head + "{:<9}".format('Sess {}'.format(i + 1))
            str_acc = str_acc + "{:<9}".format("{:.1f}".format(self.overall[i][0] *100.0))
            str_acc_old = str_acc_old +  "{:<9}".format("{:.1f}".format(self.old[i][0] *100.0))
            str_acc_novel =str_acc_novel + "{:<9}".format("{:.1f}".format(self.novel[i][0] *100.0))
            str_f1 = str_f1 + "{:<9}".format("{:.1f}".format(self.overall[i][1] *100.0))
            str_f1_old = str_f1_old + "{:<9}".format("{:.1f}".format(self.old[i][1] *100.0))
            str_f1_novel = str_f1_novel + "{:<9}".format("{:.1f}".format(self.novel[i][1] *100.0))
            str_pre = str_pre + "{:<9}".format("{:.1f}".format(self.overall[i][2] *100.0))
            str_pre_old = str_pre_old + "{:<9}".format("{:.1f}".format(self.old[i][2] *100.0))
            str_pre_novel = str_pre_novel + "{:<9}".format("{:.1f}".format(self.novel[i][2] *100.0))
            str_rec = str_rec + "{:<9}".format("{:.1f}".format(self.overall[i][3] *100.0))
            str_rec_old = str_rec_old + "{:<9}".format("{:.1f}".format(self.old[i][3] *100.0))
            str_rec_novel =str_rec_novel + "{:<9}".format("{:.1f}".format(self.novel[i][3] *100.0))
            
        print(str_head)
        print(str_acc)
        print(str_acc_old)
        print(str_acc_novel)
        print(str_f1)
        print(str_f1_old)
        print(str_f1_novel)
        print(str_pre)
        print(str_pre_old)
        print(str_pre_novel)
        print(str_rec)
        print(str_rec_old)
        print(str_rec_novel)

            
            
    def add(self, record):
        self.overall.append(record[0])
        self.novel.append(record[1])
        self.old.append(record[2])
        

    def save(self):
        str_head = "{:<9}".format('Metric')
        str_acc = "{:<9}".format("Acc")
        str_acc_old = "{:<9}".format("Old")
        str_acc_novel = "{:<9}".format("Novel")
        str_f1 = "{:<9}".format("F1")
        str_f1_old = "{:<9}".format("Old")
        str_f1_novel = "{:<9}".format("Novel")
        str_pre = "{:<9}".format("Pre")
        str_pre_old = "{:<9}".format("Old")
        str_pre_novel = "{:<9}".format("Novel")
        str_rec = "{:<9}".format("Rec")
        str_rec_old = "{:<9}".format("Old")
        str_rec_novel = "{:<9}".format("Novel")
        for i in range(self.args.session):
            str_head = str_head + ",{:<9}".format('Sess {}'.format(i + 1))
            str_acc = str_acc + ",{:<9}".format("{:.1f}".format(self.overall[i][0] *100.0))
            str_acc_old = str_acc_old +  ",{:<9}".format("{:.1f}".format(self.old[i][0] *100.0))
            str_acc_novel =str_acc_novel + ",{:<9}".format("{:.1f}".format(self.novel[i][0] *100.0))
            str_f1 = str_f1 + "{:<9}".format(",{:.1f}".format(self.overall[i][1] *100.0))
            str_f1_old = str_f1_old + ",{:<9}".format("{:.1f}".format(self.old[i][1] *100.0))
            str_f1_novel = str_f1_novel + ",{:<9}".format("{:.1f}".format(self.novel[i][1] *100.0))
            str_pre = str_pre + ",{:<9}".format("{:.1f}".format(self.overall[i][2] *100.0))
            str_pre_old = str_pre_old + ",{:<9}".format("{:.1f}".format(self.old[i][2] *100.0))
            str_pre_novel = str_pre_novel + ",{:<9}".format("{:.1f}".format(self.novel[i][2] *100.0))
            str_rec = str_rec + ",{:<9}".format("{:.1f}".format(self.overall[i][3] *100.0))
            str_rec_old = str_rec_old + ",{:<9}".format("{:.1f}".format(self.old[i][3] *100.0))
            str_rec_novel =str_rec_novel + ",{:<9}".format("{:.1f}".format(self.novel[i][3] *100.0))

        csv_file_path = os.path.join(self.args.log_dir, f"{self.args.dataset}_{self.args.seed}.csv")
        with open(csv_file_path, 'a') as csv_f:
           csv_f.write(str_head + "\n")
           csv_f.write(str_acc + "\n")
           csv_f.write(str_acc_old + "\n")
           csv_f.write(str_acc_novel + "\n")
           csv_f.write(str_f1 + "\n")
           csv_f.write(str_f1_old + "\n")
           csv_f.write(str_f1_novel + "\n")
           csv_f.write(str_pre + "\n")
           csv_f.write(str_pre_old + "\n")
           csv_f.write(str_pre_novel + "\n")
           csv_f.write(str_rec + "\n")
           csv_f.write(str_rec_old + "\n")
           csv_f.write(str_rec_novel + "\n")
        csv_f.close()

class Sampler(nn.Module):
    def __init__(self, args):
        super(Sampler, self).__init__()
        self.args = args
        self.dim = args.ctx_dim
        # TOP R
        self.k = 5
        # the number of samples per shot
        self.num_sampled = 20
        self.threshold = 0.8

    def calculate_var(self, features):
        v_mean = features.mean(dim=1) 
        v_cov = []
        for i in range(features.shape[0]):
            diag = torch.var(features[i], dim=0)
            v_cov.append(diag)
        v_cov = torch.stack(v_cov)

        return v_mean, v_cov

    def forward(self, prototypes, queries):
        self.nway = prototypes.shape[0]
        self.kshot = 5
        similarity = prototypes / prototypes.norm(dim=-1, keepdim=True) @ (queries / queries.norm(dim=-1, keepdim=True)).t()
        # (N, K, NQ)
        similarity = -similarity.view(prototypes.shape[0], prototypes.shape[1], -1)

        values, indices = similarity.topk(self.k, dim=2, largest=False, sorted=True)     
        nindices = indices.view(-1, self.k)
       
        convex_feat = []
        for i in range(nindices.shape[0]):
            convex_feat.append(queries.index_select(0, nindices[i]))
        convex_feat = torch.stack(convex_feat) # NK, k, 768

        sampled_data = convex_feat.view(prototypes.shape[0], self.kshot * self.k, self.dim)
       
        return sampled_data
    


class ProtoPN(torch.nn.Module):
    def __init__(self, args):
        super(ProtoPN, self).__init__()
        self.args = args
        self.way = args.way
        self.session = args.session
        self.dim = args.ctx_dim
        self.classes = range(args.way * args.session)
        self.memory = {cls_name: np.empty((0, self.dim)) for cls_name in self.classes}
        self.old_prototypes = np.zeros((args.way * args.session, args.ctx_dim))
        # 使用嵌入层来表示原型
        ctx_vectors= torch.empty(self.way, self.dim)
        torch.nn.init.normal_(ctx_vectors, std=0.02)

        self.learned_prototypes  = torch.nn.Parameter(ctx_vectors).to(args.device)
        
        self.reference = nn.Linear(self.dim, args.way * args.session, bias=True).to(args.device)
        nn.init.orthogonal_(self.reference.weight)
        nn.init.constant_(self.reference.bias, 0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
        self.sampler = Sampler(args)

    def add_memory(self, label, features, session):
        for ul in self.classes:
            # 选择具有相同标签的音频特征
            mask = (label == ul)
            same_label_audio = features[mask]
            # 转换为 NumPy 数组
            same_label_audio_np = same_label_audio.cpu().detach().numpy()    
            self.memory[ul] = np.vstack((self.memory[ul], same_label_audio_np))
            # 更新视觉原型
            if self.memory[ul].shape[0] > 0:
                self.old_prototypes[ul] = self.memory[ul].mean(axis=0)

    

    def Transformation_Matrix(self, prototype, session):
        C = prototype # (N, emd_dim)
        eps = 1e-6 #避免除以0的情况
        R = self.reference.weight[:(session+1) * self.way,:] # (emd_dim, N)

        # 标准化
        power_R = ((R * R).sum(dim=1, keepdim=True)).sqrt()
        R = R / (power_R + eps)

        # 标准化
        power_C = ((C * C).sum(dim=1, keepdim=True)).sqrt()
        C = C / (power_C + eps)
        P = torch.matmul(torch.pinverse(C), R)
        P = P.permute(1, 0)
        return P


    def predict(self, audio_features, session):
        # 获取当前session的原型
        prototypes = torch.tensor(self.old_prototypes[: (1+session) * self.way], dtype=torch.float32).to(self.args.device)
        
        prototypes[session * self.way: (1+session) * self.way] += self.learned_prototypes
        
        audio_features = F.normalize(audio_features, dim=-1)
        prototypes = F.normalize(prototypes, dim=-1)
        # 投影特征
        P = self.Transformation_Matrix(prototypes, session)
        weight = P.view(P.size(0), P.size(1), 1)
        prototypes = F.conv1d(prototypes.squeeze(0).unsqueeze(2), weight).squeeze(2)
        audio_features = F.conv1d(audio_features.squeeze(0).unsqueeze(2), weight).squeeze(2)

        # 归一化音频特征和原型
        audio_features = F.normalize(audio_features, dim=-1)
        prototypes = F.normalize(prototypes, dim=-1)
            
        # 计算相似度
        logits = audio_features @ prototypes.t()


        return logits
    
    def infer(self, audio_features, session):
        with torch.no_grad():
            prototypes = torch.tensor(self.old_prototypes[: (1+session) * self.way],dtype=torch.float32).to(self.args.device)
            audio_features = F.normalize(audio_features, dim=-1)
            prototypes = F.normalize(prototypes, dim=-1)
            P = self.Transformation_Matrix(prototypes, session)
            weight = P.view(P.size(0), P.size(1), 1)
            prototypes = F.conv1d(prototypes.squeeze(0).unsqueeze(2), weight).squeeze(2)
            audio_features = F.conv1d(audio_features.squeeze(0).unsqueeze(2), weight).squeeze(2)

            samples = []
            # 得到所有的样本，
            for ul in range((session+1)*self.args.way):
                if self.memory[ul].shape[0] > 0:
                    samples.append(self.memory[ul])
            samples_np = np.stack(samples)
            samples_tensor = torch.tensor(samples_np, dtype=torch.float32).to(audio_features.device).view(-1,self.dim)
            samples_tensor = F.normalize(samples_tensor, dim=-1)
            samples_tensor = F.conv1d(samples_tensor.squeeze(0).unsqueeze(2), weight).squeeze(2).view(prototypes.shape[0],-1, self.dim)
            # 利用query来计算原型
            sampled_data = self.sampler(samples_tensor, audio_features)
            # import pdb
            # pdb.set_trace()
            c_prototypes = prototypes.view(prototypes.shape[0], -1, self.dim) 
            prototypes = torch.cat((c_prototypes, samples_tensor, sampled_data), dim=1)
            # prototypes = torch.cat((5*c_prototypes, samples_tensor, sampled_data), dim=1)
            prototypes = torch.mean(prototypes, dim=1)

            audio_features = F.normalize(audio_features, dim=-1)
            prototypes = F.normalize(prototypes, dim=-1)
            
            logits = audio_features @ prototypes.t()
            
        return logits



def init_epoch(model, dataloader, optimizer, criterion, device, epoch, args, session):

    model['encoder'].eval()
    model['classifier'].eval()
    for batch_idx, (data, label) in enumerate(tqdm(dataloader)):
        data = data.to(device).squeeze(1)
        label = label.to(device)     
        features = model['encoder'](data)
        model['classifier'].add_memory(label, features, session)
    
        

def train_epoch(model, dataloader, optimizer, criterion, device, epoch, args, session):
    """
    在一个训练周期中训练模型。
    
    Args:
        model (dict): 包含模型组件的字典。
        dataloader (torch.utils.data.DataLoader): 包含训练数据的迭代器。
        optimizer (torch.optim.Optimizer): 用于训练模型的优化器。
        criterion (torch.nn.Module): 计算损失函数的模块。
        device (torch.device): 模型和数据所在的设备（CPU或GPU）。
        epoch (int): 当前训练周期数。
        args (object): 参数对象。
        session (int): 当前会话编号。
    
    Returns:
        tuple: 包含平均损失、实际标签和预测标签的元组。
            - avg_loss (float): 平均损失。
            - actual_labels (list): 实际标签列表。
            - predicted_labels (list): 预测标签列表。
    """
    # model['encoder'].train()
    model['classifier'].train()
    losses = []
    actual_labels = []
    predicted_labels = []
    for batch_idx, (data, label) in enumerate(tqdm(dataloader)):
        data = data.to(device).squeeze(1)
        label = label.to(device)     
        features = model['encoder'](data)
        logits = model['classifier'].predict(features, session)
        loss = criterion(logits, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        actual_labels.extend(label.cpu().numpy())
        predicted_labels.extend(torch.argmax(logits, dim=1).cpu().numpy())

    avg_loss = sum(losses) / len(losses)
    return avg_loss, actual_labels, predicted_labels


def train_session(model, data, optimizer, criterion, device, epoch, args, session):
    """
    训练一个会话的数据。
    
    Args:
        model (dict): 包含模型组件的字典。
        data (object): 数据对象。
        optimizer (torch.optim.Optimizer): 用于训练模型的优化器。
        criterion (torch.nn.Module): 计算损失函数的模块。
        device (torch.device): 模型和数据所在的设备（CPU或GPU）。
        epoch (int): 总训练周期数。
        args (object): 参数对象。
        session (int): 当前会话编号。
    """
    train_data = data.train_sessions[session]
    train_data.reset_index(drop=True)
    train_dataset = ProcessDataset(train_data, data.label_mapping)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers)
    
    # 初始化一个原型
    init_epoch(model, train_dataloader, optimizer, criterion, device, epoch, args, session)
    print(f"\n\n-------------------------------\nInit (Session {session})\n-------------------------------\n")
    for ep in range(epoch):
        train_loss, actual_labels, predicted_labels = train_epoch(model, train_dataloader, optimizer, criterion, device, ep, args, session)
        accuracy, f1_score, precision, recall = get_scores(actual_labels, predicted_labels, args.classes[:(session+1)* args.way])
        print(f"\n\n-------------------------------\nTrain (Epoch {epoch}/{ep+1})\n-------------------------------\n")
        print_scores(accuracy, f1_score, precision, recall, train_loss) 

def test_session(model, data, optimizer, criterion, device, epoch, args, session):
    """
    测试一个会话的数据。
    
    Args:
        model (dict): 包含模型组件的字典。
        data (object): 数据对象。
        optimizer (torch.optim.Optimizer): 用于训练模型的优化器。
        criterion (torch.nn.Module): 计算损失函数的模块。
        device (torch.device): 模型和数据所在的设备（CPU或GPU）。
        epoch (int): 总训练周期数。
        args (object): 参数对象。
        session (int): 当前会话编号。
    """
    if session == 0:
        test_data = data.test_sessions[session]
    else:
        test_data = pd.concat(data.test_sessions[:session+1])
        test_data.reset_index(drop=True)
    test_dataset = ProcessDataset(test_data, data.label_mapping)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True, num_workers=args.num_workers)
    with torch.no_grad():
        model['encoder'].eval()
        model['classifier'].eval()
        losses = []
        actual_labels = []
        predicted_labels = []
        for batch_idx, (data, label) in enumerate(tqdm(test_dataloader)):
            data = data.to(device).squeeze(1)
            label = label.to(device)     
            features = model['encoder'](data)
            logits = model['classifier'].infer(features, session)
            loss = criterion(logits, label)
            losses.append(loss.item())
            actual_labels.extend(label.cpu().numpy())
            predicted_labels.extend(torch.argmax(logits, dim=1).cpu().numpy())

    avg_loss = sum(losses) / len(losses)
    return avg_loss, actual_labels, predicted_labels

def train(model, data, optimizer, criterion, device, epoch, args):
    """
    训练整个模型。
    
    Args:
        model (dict): 包含模型组件的字典。
        data (object): 数据对象。
        optimizer (torch.optim.Optimizer): 用于训练模型的优化器。
        criterion (torch.nn.Module): 计算损失函数的模块。
        device (torch.device): 模型和数据所在的设备（CPU或GPU）。
        epoch (int): 总训练周期数。
        args (object): 参数对象。
    """
    record = Record(args)
    for session in range(args.session):
        print("*"*20 + "Session {}".format(session + 1) + "*"*20)
        train_session(model, data, optimizer, criterion, device, epoch, args, session)
        # 使用训练好的原型更新原型
        model['classifier'].old_prototypes[session*args.way:(session+1)*args.way] = model['classifier'].learned_prototypes.detach().cpu().numpy() + model['classifier'].old_prototypes[session*args.way:(session+1)*args.way]
        test_loss, actual_labels, predicted_labels = test_session(model, data, optimizer, criterion, device, epoch, args, session)
        accuracy, f1_score, precision, recall = get_scores(actual_labels, predicted_labels, args.classes[:(session+1)* args.way])
        print(f"\n\n-------------------------------\nTest (Session {session})\n-------------------------------\n")
        print_scores(accuracy, f1_score, precision, recall, test_loss) 
        record.add(get_record(actual_labels, predicted_labels, session, args))
        # model['classifier'].soft_calibration(args, session)
    record.print() 
    record.save()

if __name__ == '__main__':
    date_now = datetime.datetime.now(pytz.timezone('Asia/Shanghai'))
    print(f'Time & Date = {date_now.strftime("%I:%M %p")}, {date_now.strftime("%d_%b_%Y")} \n')
    print(f"{'*'*20} Begin experiment {'*'*20}")
    
    args = get_parse()
    
    if args.cuda != -1:
        args.device = torch.device(f'cuda:{args.cuda}')
    else:
        args.device = torch.device('cpu')
    set_seed(args.seed)
    log_file = setup_logging(args)
    
    # 加载数据集
    print(f'Time & Date = {date_now.strftime("%I:%M %p")}, {date_now.strftime("%d_%b_%Y")} \n')
    print(f"{'*'*20} Create Dataset {'*'*20}")
    
    if args.dataset == 'FSC':
        dataset = FSC89Dataset()
    elif args.dataset == 'LBS':
        dataset = LBS100Dataset()
    elif args.dataset == 'Ny':
        dataset = Ny100Dataset()
    data = FSCILDataset(args, dataset)
    args.classnames = data.selected_classnames
    args.classes = data.classes

    # 加载模型
    print(f'Time & Date = {date_now.strftime("%I:%M %p")}, {date_now.strftime("%d_%b_%Y")} \n')
    print(f"{'*'*20} Create Model {'*'*20}")

    model = {}
    model['encoder'] = models.Encoder(args, pengi).to(args.device)
    model['classifier'] = ProtoPN(args)

    # 设置优化器
    optimizer = torch.optim.Adam([
        {'params': model['classifier'].parameters()}
    ], lr=args.lr)

    # 开始训练
    train(model, data, optimizer, get_loss(args), args.device, args.n_epochs, args)

