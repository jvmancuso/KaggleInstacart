import numpy as np
from sklearn.metrics import fbeta_score, accuracy_score
import shutil
import torch

def classify_and_score(preds, labels):
    """Computes validation accuracy and F-1 score."""
    rounded = np.around(preds.cpu().data.numpy())
    binary_preds, binary_labels = get_binary_preds(rounded, labels)
    F = fbeta(binary_preds, binary_labels)
    acc = accuracy_score(rounded, labels)
    return F, acc

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
        ---How much to weight recall.---
    """
    return fbeta_score(labels.cpu().data.numpy(), preds.cpu().numpy(), beta=beta, average='weighted')

    
def get_binary_preds(preds, labels):
    labels = labels.cpu().data.numpy()
    