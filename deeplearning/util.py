import copy
import torch
import numpy as np
from .model import FSTabularModel
from scipy.stats import zscore
from sklearn.base import BaseEstimator, TransformerMixin

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_model = None
        self.best_loss = None
        self.counter = 0
        self.status = ""

    def __call__(self, model, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model.state_dict())
        elif self.best_loss - val_loss >= self.min_delta:
            self.best_model = copy.deepcopy(model.state_dict())
            self.best_loss = val_loss
            self.counter = 0
            self.status = f"Improvement found, counter reset to {self.counter}"
        else:
            self.counter += 1
            self.status = f"No improvement in the last {self.counter} epochs"
            if self.counter >= self.patience:
                self.status = f"Early stopping triggered after {self.counter} epochs."
                if self.restore_best_weights:
                    model.load_state_dict(self.best_model)
                return True
        return False
    
def make_model(n_cont, model_args):    
    model = FSTabularModel(n_cont=n_cont, **model_args)
    return model

class ZScoreScaler(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return zscore(X, axis=0, nan_policy='omit')
    
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100 
    return acc

def normalize(x, means, stds):
    """Z-Normalization: Subtracts the mean from x and then divides by standard deviation (STD). 
    
    Arguments:
        x {list-like or pd.DataFrame} -- a matrix or dataframe of continuous input values
        means {np.ndarray} -- array containing the mean of every column in x. (Must be in same order as x's columns.)
        stds {np.ndarray} -- array containing the STD of every column in x. (Must be in same order as x's columns.)
    
    Returns:
        {list-like or pd.DataFrame} -- a normalized copy of x 
    """
    norm_stds = np.asarray(stds)
    norm_means = np.asarray(means)
    norm_stds[norm_stds==0.0] = 1e-6
        
    return (x - norm_means) / norm_stds # epsilon here prevents division by 0 if x.std()==0

def split_loss_weight(y_true:torch.Tensor) -> tuple:
    """Extracts loss weight from y_true if it's included

    Args:
        y_true (torch.Tensor): labels, possibly with a second dimension for the loss weights

    Returns:
        tuple: (labels, loss weights). If y_true was already just the labels w/o loss weights, loss weights will be None.
    """    
    w = None
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        w = y_true[:,1]
        y_true = y_true[:,0].long()
    return y_true, w


# ------------------------------------------------------------ Model Ops -------------------------------------------------------
def load_from_ckpt(ckptpath:str, model=None, opt=None, strict:bool=True, ret_ckpt=False):
    """Loads a model's and/or opt's statedict(s) from a checkpoint file, and optionally also returns the checkpoint object

    Args:
        ckptpath (str): pathh to the saved checkpoint object
        model (torch.nn.Module, optional): model object. Defaults to None.
        opt (torch.optim.optimizer.Optimizer, optional): torch optimizer object. Defaults to None.
        strict (bool, optional): strict loading of model state dict. Defaults to True.
        ret_ckpt (bool, optional): if True, this function returns the checkpoint object. If False, this function has no return object. Defaults to False.

    Returns:
        If ret_ckpt: returns the checkpoint dict object saved at ckptpath
        Else: none 
    """    
    checkpoint = torch.load(ckptpath)
    if model is not None: model.load_state_dict(checkpoint['model_sd'], strict=strict)
    if opt is not None: opt.load_state_dict(checkpoint['opt'])
    if ret_ckpt: return checkpoint


# ------------------------------------------------------------- Misc --------------------------------------------------------
def protect(*protected):
    """ Returns a metaclass that protects all attributes (given as strings) from being overriden

    Args:
        *protected (str): arbitrary number of str args that are the attributes of a parent class which you want to prevent from being overriden

    Returns:
        class: the Protect metaclass
    """    
    class Protect(type):
        has_base = False
        def __new__(meta, name, bases, attrs):
            if meta.has_base:
                for attribute in attrs:
                    if attribute in protected:
                        raise AttributeError('Overriding of attribute "%s" not allowed.'%attribute)
            meta.has_base = True
            klass = super().__new__(meta, name, bases, attrs)
            return klass
    return Protect
