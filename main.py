import os
import random
import numpy as np
import datetime
import pandas as pd
import pytz
from tqdm import tqdm

import torch
import torch.nn as nn
from loader import FSC89Dataset, FSCILDataset, LBS100Dataset, Ny100Dataset, ProcessDataset
from trainer import run_evaluation, run_training
from utils import get_loss, get_model, get_parse, get_scores, print_scores, set_seed, setup_logging
import models
from train import train

def main():
    date_now = datetime.datetime.now(pytz.timezone('Asia/Shanghai'))
    print(f'Time & Date = {date_now.strftime("%I:%M %p")} , {date_now.strftime("%d_%b_%Y")} \n')
    print(f"*"*20 + "Begin experiment" + "*"*20)
    args = get_parse()
    
    if args.cuda != -1:
        args.device = torch.device('cuda:{}'.format(args.cuda))
    else:
        args.device = torch.device('cpu')
    set_seed(args.seed)
    log_file = setup_logging(args)
    
    
    # 加载数据集
    print(f'Time & Date = {date_now.strftime("%I:%M %p")} , {date_now.strftime("%d_%b_%Y")} \n')
    print(f"*"*20 + "Create Dataset" + "*"*20)
    
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
    print(f'Time & Date = {date_now.strftime("%I:%M %p")} , {date_now.strftime("%d_%b_%Y")} \n')
    print(f"*"*20 + "Create Model" + "*"*20)
    model = get_model(args)
    model.to(args.device)
    loss_function = get_loss(args)
    
    # if args.model_name == 'palm':
    #     traindata = pd.concat(data.train_sessions)
    #     traindata.reset_index(drop=True)
    #     train_dataset = ProcessDataset(traindata, data.label_mapping)
    #     testdata = pd.concat(data.test_sessions)
    #     testdata.reset_index(drop=True)
    #     test_dataset = ProcessDataset(testdata, data.label_mapping)
    #     optimizer = torch.optim.SGD(model.prompt_learner.parameters(), lr=args.lr, momentum=0.9)
    #     train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers)
    #     test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=args.num_workers)
    #     run_training(model, train_dataloader, test_dataloader, optimizer, loss_function, args.device, epochs=args.n_epochs, args=args)
    #     test_loss, actual_labels, predicted_labels = run_evaluation(model, test_dataloader, loss_function, args.device)
    #     accuracy, f1_score, precision, recall =  get_scores(actual_labels, predicted_labels, args.classnames)
    #     print_scores(accuracy, f1_score, precision, recall, test_loss)

    if args.model_name == 'zeroshot':
        testdata = pd.concat(data.test_sessions)
        testdata.reset_index(drop=True)
        test_dataset = ProcessDataset(testdata, data.label_mapping)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=args.num_workers)
        test_loss, actual_labels, predicted_labels = run_evaluation(model, test_dataloader, loss_function, args.device)
        accuracy, f1_score, precision, recall =  get_scores(actual_labels, predicted_labels, args.classes)
        print_scores(accuracy, f1_score, precision, recall, test_loss)
    else:
        optimizer = torch.optim.SGD(model.prompt_learner.parameters(), lr=args.lr, momentum=0.9)
        train(model, data, optimizer, loss_function, args.device, args.n_epochs, args)

if __name__ == '__main__':
    main()