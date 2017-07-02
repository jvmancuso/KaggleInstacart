import numpy as np
from sklearn.metrics import fbeta_score
import torch
from torch.utils.data import Dataset
import shutil

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
        return len(self.data)
    
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
        try:
            return {'features': torch.from_numpy(sample['features']), 
                'target': torch.from_numpy(sample['target']).type(torch.FloatTensor)}
        except RuntimeError:
            return {'features': torch.from_numpy(sample['features']), 
                'target': torch.Tensor()}

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """ Saves most recent model and best model. """
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')
        
def fbeta(preds, labels, beta=1):
    """Computes F-beta score.
    preds: torch.cuda.ByteTensor
        ---Class predictions in {0,1} for each possible label.---
    labels: torch.cuda.ByteTensor
        ---Ground truth value in {0,1} for each possible label.---
    beta: float
        ---How much to weight recall, always 2 for our cases.---
    """
    return fbeta_score(labels.cpu().numpy(), preds.cpu().numpy(), beta=beta, average='weighted')