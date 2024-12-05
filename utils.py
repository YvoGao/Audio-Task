

import argparse
import os
import random
import models
from pengi import pengi
import numpy as np
import torch
import sys
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
# from loader import FSC89Dataset, LBS100Dataset, Ny100Dataset


def get_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='palm', help='model name')
    parser.add_argument('--dataset', type=str, default='FSC', help='dataset name  (LBS,FSC,Ny)')
    parser.add_argument('--metapath', type=str, help='path to FSC-89-meta folder')
    parser.add_argument('--datapath', type=str, help='path to FSD-MIX-CLIPS_data folder)')
    parser.add_argument('--cuda', type=int, default=1, help='cuda device id')
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--save_path', type=str, default='./result/')
    parser.add_argument('--do_logging', help='Disable Logging (default: False)', action='store_true')
    parser.add_argument('--exp_name', type=str, default='test', help='experiment name')

    # dataset setting(way, shot, query)
    parser.add_argument('--way', type=int, default=5, help='class number of per task (default: 5)')
    parser.add_argument('--shot', type=int, default=5, help='shot of per class (default: 5)')
    parser.add_argument('--query', type=int, default=200, help='query of per class (default: 15)')
    parser.add_argument('--test_batch_size', type=int, default=128)
    parser.add_argument('--train_batch_size', type=int, default=5)
    parser.add_argument('--freq_test_model', type=int, default=10, help='Frequency of testing the model (default: 10)')
        
    # hyper option
    parser.add_argument('--lr', type=float, default=0.001, help= 'Learning Rate for Adam Optimizer (default: 0.001)')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers (default: 4)')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate (default: 0.5)')
    parser.add_argument('--weight_decay', type=float, default=0.2, help='weight decay (default: 1e-4)')
    parser.add_argument('--warmup_steps', type=int, default=100)
    parser.add_argument('--n_epochs', type=int, default=5, help='Number of epochs (default: 100)')
    
    parser.add_argument('--sim', type=str, default='cosine', help='similarity measure (default: cosine)')
    parser.add_argument('--T', type=int, default=5, help='scale factor (default: 5)')
    parser.add_argument('--session', type=int, default=5, metavar='N',
                        help='num. of sessions, including one base session and n incremental sessions (default:10)')
    parser.add_argument('--ctx_dim', type=int, default=1024, help='Dimension of the context vector (default: 512)')
    parser.add_argument('--n_ctx', type=int, default=16)
    parser.add_argument('--prompt_prefix', type=str, default='The is a audio of ', help='Prompt Prefix (default: The is a recording of )')
    parser.add_argument('--spec_aug', help='Apply Spectrogram Augmentation (default: False)', action='store_true')
    
    parser.add_argument('--shift_weight', type=float, default=0.5)
    parser.add_argument("--softmax_t", type=int, default=16)
    return parser.parse_args()
    
def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    
def setup_logging(args):
    log_dir = os.path.join('logs', args.model_name) # log file dir
    args.log_dir = log_dir
    
    if args.do_logging:
        if not os.path.exists(log_dir): 
            os.makedirs(log_dir)
        log_file_path = os.path.join(log_dir, f"{args.exp_name+'-SEED'+str(args.seed)}.log")
        if os.path.exists(log_file_path): 
            os.remove(log_file_path)
        json_file_path = os.path.join(log_dir, f"{args.exp_name}.json")
        args.json_file_path = json_file_path
        print(f"\nLogging to '{log_file_path}'\n")
        log_file = redirect_output_to_log(log_file_path) # redirect terminal output to log file
    else:
        log_file =None

    return log_file


def redirect_output_to_log(log_file):
    # Open the log file in append mode
    log = open(log_file, 'a')
 
    # Duplicate stdout and stderr
    sys.stdout = Tee(sys.stdout, log)
    sys.stderr = Tee(sys.stderr, log)

    return log

class Tee:
    def __init__(self, *files):
        self.files = files
 
    def write(self, text):
        for file in self.files:
            file.write(text)
            file.flush()
 
    def flush(self):
        for file in self.files:
            file.flush()

METHODS = ['zeroshot', 'coop', 'cocoop', 'palm', 'fscilpalm']
def get_model(args):
    print(f"Using Method: '{args.model_name.upper()}'\n")

    if args.model_name == 'zeroshot':
        model = models.ZeroShot(args, pengi)
    elif args.model_name == 'fscilcoop':
        model = models.COOP(args, pengi)
    elif args.model_name == 'fscilcocoop':
        model = models.COCOOP(args, pengi)
    elif args.model_name == 'palm':
        model = models.PALM(args, pengi)
    elif args.model_name == 'fscilpalm':
        model = models.FSCILPALM(args, pengi)
        # raise NotImplementedError("Model 'palm' is not implemented yet.")
    else:
        raise ValueError(f"Model '{args.model_name}' is not supported. Choose from: [{', '.join(METHODS)}]")
    
    return model

def get_loss(args):
    # 基础损失函数
    criterion = torch.nn.CrossEntropyLoss()
    # 利用LM头的损失函数
    
    # data_free的损失函数
    return criterion


def get_scores(actual_labels, predicted_labels, classnames):
    # import pdb;pdb.set_trace()
    # try:
    #     cls_report = classification_report(actual_labels, predicted_labels, target_names=classnames, output_dict=True)
    # except:
    #     import pdb;pdb.set_trace()
    accuracy = accuracy_score(actual_labels, predicted_labels)
    f1 = f1_score(actual_labels, predicted_labels, average='micro')
    precision = precision_score(actual_labels, predicted_labels, average='micro', zero_division=0)
    recall = recall_score(actual_labels, predicted_labels, average='micro', zero_division=0)
    
    
    return accuracy, f1, precision, recall

def print_scores(accuracy, f1_score, precion, recall, avg_loss):
    print(f"{'Accuracy':<15} = {accuracy:0.4f}")
    print(f"{'F1-Score':<15} = {f1_score:0.4f}")
    print(f"{'Precision':<15} = {precion:0.4f}")
    print(f"{'Recall':<15} = {recall:0.4f}")
    print(f"{'Average Loss':<15} = {avg_loss:0.4f}\n\n")
    
    
    
def get_record(label, pre, session, args):
    """
    根据给定的实际标签、预测标签、会话和参数，计算并存储各项准确率。
    
    Args:
        actual_labels (list or array): 实际标签列表或数组。
        predicted_labels (list or array): 预测标签列表或数组。
        session (int): 当前会话的索引。
        args (argparse.Namespace): 包含配置参数的命名空间对象。
    
    Returns:
        None
    
    """
    # 计算整体准确率
    total_correct = sum(1 for l, p in zip(label, pre) if l == p)
    overall_accuracy = total_correct / len(label)
    categories = np.arange((session + 1) * args.way)
    overall = get_scores(label, pre, categories)

    if session == 0:
        old = [0, 0, 0, 0 ]
        novel = [0, 0, 0, 0 ]
        return overall, novel, old
    
    
    # 计算基类的准确率       
    # 过滤出特定类别的样本
    old_labels = [l for l in label if l in categories[:session * args.way]]
    old_predictions = [p for l, p in zip(label, pre) if l in categories[:session * args.way]]
    # 计算准确率
    correct_predictions = sum(1 for l, p in zip(old_labels, old_predictions) if l == p)
    old_accuracy = correct_predictions / len(old_labels)
    old = get_scores(old_labels, old_predictions, categories[:session * args.way])
    
    # 计算新类的准确率
    # 过滤出特定类别的样本
    novel_labels = [l for l in label if l in categories[session * args.way:]]
    novel_predictions = [p for l, p in zip(label, pre) if l in categories[session * args.way:]]
    # 计算准确率
    correct_predictions = sum(1 for l, p in zip(novel_labels, novel_predictions) if l == p)
    novel_accuracy = correct_predictions / len(novel_labels)
    novel = get_scores(novel_labels, novel_predictions, categories[session * args.way:])
    # return old_accuracy, novel_accuracy, overall_accuracy
    
    return overall, novel, old