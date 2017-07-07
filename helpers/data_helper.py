import numpy as np
import torch
from torch.utils.data import Dataset

DATA_PATH = '../data/final/'

class InstacartDataset(Dataset):
    """Custom class for our data."""
    def __init__(self, data_filename:str, labels_filename=None, transform=None):
        self.data = np.load(DATA_PATH+data_filename,mmap_mode='r')
        if labels_filename:
            self.labels = np.load(DATA_PATH+labels_filename, mmap_mode = 'r')
        else:
            self.labels = labels_filename
        self.transform = transform
        
    def __len__(self):
        return self.data.shape[1]
    
    def __getitem__(self, index:int):
        try:
            sample = {'features': self.data[:,index,2:], 'target': self.labels[index,1:]}
        except TypeError:
            sample = {'features': self.data[:,index,2:], 'target': self.labels}
        if self.transform:
            sample = self.transform(sample)
        return sample

class ToTensor(object):
    """Transforms ndarrays in sample dict to torch.Tensor objects"""
    def __call__(self, sample):
        return {'features': torch.from_numpy(sample['features']).type(torch.FloatTensor), 
                'target': torch.from_numpy(sample['target']).type(torch.FloatTensor)}
#        except RuntimeError:
#            return {'features': torch.from_numpy(sample['features']), 
#                'target': torch.Tensor()}

