
import argparse
import json
import os
import pandas as pd
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset
from pengi import pengi
from utils import get_parse


FSC89_datapath = "/data/gaoyunlong/dataset/Audio/dataset_FSC89/"
FSC89_metapath = "/data/gaoyunlong/dataset/Audio/dataset_FSC89/meta/"

Ny100_datapath = "/data/gaoyunlong/dataset/Audio/dataset_Nsynth/"
Ny100_metapath = "/data/gaoyunlong/dataset/Audio/dataset_Nsynth/"

LBS100_datapath = "/data/gaoyunlong/dataset/Audio/datasets_LibiriSpeech/"
LBS100_metapath = "/data/gaoyunlong/dataset/Audio/ls100-meta/"



class FSC89Dataset(object):
    def __init__(self):
        # 获得所有的列表
        meta_info_train = pd.read_csv(os.path.join(FSC89_metapath, 'setup1', 'Fsc89-' + 'setup1' + '-fsci_train.csv'))
        meta_info_test = pd.read_csv(os.path.join(FSC89_metapath, 'setup1',  'Fsc89-' + 'setup1' + '-fsci_test.csv'))
        meta_info_val = pd.read_csv(os.path.join(FSC89_metapath, 'setup1', 'Fsc89-' + 'setup1' + '-fsci_val.csv'))
        train_labels = np.array(meta_info_train['label'])
        test_labels = np.array(meta_info_test['label'])
        train_start_time = meta_info_train['start_time']
        test_start_time = meta_info_test['start_time']
        
        self.train_data = [os.path.join(FSC89_datapath, 'audio', meta_info_train['data_folder'][index], meta_info_train['FSD_MIX_SED_filename'][index].replace('.wav', '_' + str(int(train_start_time[index] * 44100)) + '.wav')) for index in range(len(meta_info_train['FSD_MIX_SED_filename']))]
        self.train_labels = [train_labels[index] for index in range(len(meta_info_train['label']))]
        
        self.test_data = [os.path.join(FSC89_datapath, 'audio', meta_info_test['data_folder'][index], meta_info_test['FSD_MIX_SED_filename'][index].replace('.wav', '_' + str(int(test_start_time[index] * 44100)) + '.wav')) for index in range(len(meta_info_test['FSD_MIX_SED_filename']))]
        self.test_labels = [test_labels[index] for index in range(len(meta_info_test['label']))]
        self.classes = np.max(train_labels)
        self.label_map = json.load(open('labelmap/FSC_89.json'))
       
    # def __len__(self):
    #     return len(self.train_data)
        
class Ny100Dataset(object):
    def __init__(self):
        self.audio_dir = Ny100_datapath
        self.meta_dir = os.path.join(Ny100_metapath, 'nsynth-' + str(100) + '-fs-meta')
        
        # 载入词汇
        with open(os.path.join(self.meta_dir, 'nsynth-' + str(100) + '-fs_vocab.json')) as vocab_json_file:
            label_to_ix = json.load(vocab_json_file)
        # 获得所有的列表
        meta_info_train = pd.read_csv(os.path.join(self.meta_dir, 'nsynth-' + str(100) + '-fs_train.csv'))
        meta_info_test = pd.read_csv(os.path.join(self.meta_dir, 'nsynth-' + str(100) + '-fs_test.csv'))
        meta_info_val = pd.read_csv(os.path.join(self.meta_dir, 'nsynth-' + str(100) + '-fs_val.csv'))
        
        train_filenames = meta_info_train['filename']
        train_labels = meta_info_train['instrument']
        train_audio_source = meta_info_train['audio_source']
        # 将文本编码转成数字
        train_label_code = []
        for i in range(len(train_labels)):
            train_label_code.append(label_to_ix[train_labels[i]])
        self.train_data = [os.path.join(self.audio_dir, train_audio_source[index], 'audio', train_filenames[index] + '.wav') for index in range(len(train_filenames))]
        self.train_labels = [train_label_code[index] for index in range(len(train_filenames))]
        
        test_filenames = meta_info_test['filename']
        test_labels = meta_info_test['instrument']
        test_audio_source = meta_info_test['audio_source']
        # 将文本编码转成数字
        test_label_code = []
        for i in range(len(test_labels)):
            test_label_code.append(label_to_ix[test_labels[i]])
        self.test_data = [os.path.join(self.audio_dir, test_audio_source[index], 'audio', test_filenames[index] + '.wav') for index in range(len(test_filenames))]
        self.test_labels = [test_label_code[index] for index in range(len(test_filenames))]
        self.classes = np.max(self.train_labels)
        self.label_map = json.load(open('labelmap/nsynth_100.json'))
            
        
class LBS100Dataset(object):
    def __init__(self):
        self.data_dir = LBS100_datapath
        self.metapath = LBS100_metapath

        meta_info_train = pd.read_csv(os.path.join(self.metapath, "librispeech_fscil_train.csv"))
        meta_info_val = pd.read_csv(os.path.join(self.metapath, "librispeech_fscil_val.csv"))
        meta_info_test = pd.read_csv(os.path.join(self.metapath, "librispeech_fscil_test.csv"))

        
        train_filenames = meta_info_train['filename']
        train_labels = meta_info_train['label']
        train_labels = np.array(train_labels)  # 所有标签
        self.train_data = [os.path.join(self.data_dir, train_filenames[index]) for index in range(len(train_filenames))]
        self.train_labels = [train_labels[index] for index in range(len(train_filenames))]
        
        test_filenames = meta_info_test['filename']
        test_labels = meta_info_test['label']
        test_labels = np.array(test_labels)  # 所有标签
        self.test_data = [os.path.join(self.data_dir, test_filenames[index]) for index in range(len(test_filenames))]
        self.test_labels = [test_labels[index] for index in range(len(test_filenames))]
        self.classes = np.max(train_labels)
        self.label_map = json.load(open('labelmap/libriSpeech_100.json'))


class ProcessDataset(Dataset):
    def __init__(self, data, label_mapping, resample=True):
        super().__init__()
        self.data = data
        self.process_audio_fn = pengi.preprocess_audio
        self.resample = resample
        self.label_mapping = label_mapping
        
            
    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, idx):
        
        audio_path = self.data.iloc[idx]['data']
        audio = self.process_audio_fn([audio_path], self.resample) # [1,n_samples]
        label = self.label_mapping[self.data.iloc[idx]['label']]
        # return audio, label, audio_path, row['classname']
        
        return audio, label
           


class FSCILDataset(object):
    def __init__(self, args, dataset):
        # self.dataset = dataset
        self.args = args
        
        self.train_df = pd.DataFrame({'data':dataset.train_data, 'label':dataset.train_labels})
        self.test_df = pd.DataFrame({'data':dataset.test_data, 'label':dataset.test_labels})
        self.class_num = dataset.classes
        self.class2label = dataset.label_map
        self.label2class = {v: k for k, v in self.class2label.items()}
        self.selected_classes = np.random.choice(self.class_num, self.args.session*self.args.way, replace=False)
        self.selected_classnames = [self.label2class[classname] for classname in self.selected_classes]
        self.train_sessions = []
        self.test_sessions = []
        self.genetate_data()
        self.label_mapping = {old_label: new_label for new_label, old_label in enumerate(self.selected_classes)}
        self.classes = [self.label_mapping[c] for c in self.selected_classes]
        
    def genetate_data(self):
        print("Selected classes",self.selected_classes)
        # N个会话
        for i in range(self.args.session):
            # 每个会话5个类
            train_df_subset_i = pd.DataFrame(columns=['data', 'label'])
            test_df_subset_i = pd.DataFrame(columns=['data', 'label'])
            for classname in self.selected_classes[i*5: i*5+5]:
                # 训练集
                train_df_class = self.train_df[self.train_df['label'] == classname]
                if len(train_df_class) >= self.args.shot:
                    train_df_subset_i = pd.concat([train_df_subset_i, train_df_class.sample(self.args.shot)])
                else:
                    train_df_subset_i = pd.concat([train_df_subset_i, train_df_class])
                
                # 测试集
                test_df_class = self.test_df[self.test_df['label'] == classname]
                if len(test_df_class) >= self.args.query:
                    test_df_subset_i = pd.concat([test_df_subset_i, test_df_class.sample(self.args.query)])
                else:
                    test_df_subset_i = pd.concat([test_df_subset_i, test_df_class])
            train_df_subset_i = train_df_subset_i.reset_index(drop=True)
            test_df_subset_i = test_df_subset_i.reset_index(drop=True)
            
            self.train_sessions.append(train_df_subset_i)
            self.test_sessions.append(test_df_subset_i)
    
    
if __name__ == '__main__':
    dataset1 = LBS100Dataset()
    dataset2 = FSC89Dataset()
    dataset3 = Ny100Dataset()
    
    print(dataset1.train_data[0])
    print(dataset2.train_data[0])
    print(dataset3.train_data[0])
    
    parser = get_parse()
    FSICLdata = FSCILDataset(parser, dataset3)
    for session in range(parser.session):
        
        dataset = ProcessDataset(FSICLdata.train_session[session])
        train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)    
        for item in train_dataloader:
            # import pdb
            # pdb.set_trace()
            print(item)
            
    
    # torchaudio.load(dataset1.train_data[0])
    # torchaudio.load(dataset2.train_data[0])
    # torchaudio.load(dataset3.train_data[0])