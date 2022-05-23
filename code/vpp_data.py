import os
import torch
import random
import subprocess
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import datasets


class VPPDataset(Dataset):
    def __init__(self, df, cont_cols, max_seq_len=80, times=1, augmentation=False):
        self.features = df[cont_cols].values
        self.rc_index = df['rc_index'].values
        self.r_index = df['r_index'].values
        self.c_index = df['c_index'].values
        self.u_out = df['u_out'].values        
        self.breath_id = df['breath_id'].values
        
        groups = df.groupby('breath_id').groups
        self.indices_by_breath_id = list(groups.values())
        #self.breath_ids = list(groups.keys())
        self.times = times
        self.augmentation=augmentation
         
        if 'pressure' not in df:
            df['pressure'] = 0
            df['pressure_diff1'] = 0
            df['pressure_diff2'] = 0
            df['pressure_diff3'] = 0
            df['pressure_diff4'] = 0                                            
        self.pressure = df['pressure'].values
        self.target = df[['pressure', 'pressure_diff1', 'pressure_diff2', 'pressure_diff3', 'pressure_diff4']].values
        
        self.max_seq_len = max_seq_len
    
    def __len__(self):
        """
        :return: returns the number of datapoints in the dataframe
        """
        return len(self.indices_by_breath_id)*self.times
    
    def get_sample(self, idx):
        indices = self.indices_by_breath_id[idx]
        r_index = self.r_index[indices]
        c_index = self.c_index[indices]
        u_out = self.u_out[indices]
        breath_id = self.breath_id[indices]
        
        r_code = F.one_hot(torch.LongTensor(r_index), num_classes=3).float()
        c_code = F.one_hot(torch.LongTensor(c_index), num_classes=3).float()
        
        features = self.features[indices]        
        pressure = self.pressure[indices]
        target = self.target[indices]
        ret = {            
            "r_code": r_code[:self.max_seq_len],
            "c_code": c_code[:self.max_seq_len],
            "features": torch.tensor(features, dtype=torch.float)[:self.max_seq_len],
            "u_out": torch.tensor(u_out, dtype=torch.long)[:self.max_seq_len],
            "pressure": torch.tensor(pressure, dtype=torch.float)[:self.max_seq_len],
            "target": torch.tensor(target, dtype=torch.float)[:self.max_seq_len],
            'breath_id': torch.tensor(breath_id, dtype=torch.long)[:self.max_seq_len]
        }
        return ret
    
    def __getitem__(self, idx):
        """
        Returns the excerpt text and the targets of the specified index
        :param idx: Index of sample excerpt
        :return: Returns the dictionary of excerpt text, input ids, attention mask, target
        """
        idx = idx % len(self.indices_by_breath_id)
        
        one_sample = self.get_sample(idx)
        
        if self.augmentation:
            if random.random() < 0.4:
                idx = np.random.randint(len(self.indices_by_breath_id))
                sampled_sample = self.get_sample(idx)
                mix_cols = ['r_code', 'c_code', 'features', 'pressure', 'target']                
                k = 0.5
                for col in mix_cols:
                    one_sample[col] = (one_sample[col]*k + (1-k)*sampled_sample[col])
                        
            # shuffling augmentation
            if random.random() < .2:
                ws = np.random.choice([2, 4, 5])
                num = self.max_seq_len // ws

                idx = np.arange(0, self.max_seq_len)
                for i in range(num):
                    np.random.shuffle(idx[i * ws:(i + 1) * ws])
                
                shuffle_cols = ['r_code', 'c_code', 'features', 'u_out', 'pressure', 'target']
                for col in shuffle_cols:
                    one_sample[col] = one_sample[col][idx]
            
            if random.random() < .2:
                if random.random() < .5:
                    # maskring R
                    one_sample['r_code'][:] = 0
                else:
                    # masking C
                    one_sample['c_code'][:] = 0
                        
        return one_sample        


if __name__== '__main__':
    RAW_DATA_DIR = '../../input'
    train_df = pd.read_csv(os.path.join(RAW_DATA_DIR, 'train.csv'))
    #test_df = pd.read_csv(os.path.join(RAW_DATA_DIR, 'test.csv'))
    #submission_df = pd.read_csv(os.path.join(RAW_DATA_DIR, 'sample_submission.csv'))
    
    train_db = VPPDataset(train_df)
    
    for res in train_db:
        #print(res)
        a = 0        
