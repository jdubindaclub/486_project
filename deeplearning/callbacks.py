from abc import abstractmethod
from typing import List

import os, operator
import numpy as np
from warnings import warn
import torch
from .metrics import Loss
from .metrics import Metric
import pandas as pd
import optuna

class ManualCB:
    """Base class for callbacks when training without using fastai. 
    All methods should receive a fitter object as first param.
    Add methods here like 'on_epoch_begin' as they are needed.
    """
    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def on_epoch_end(self, fitter, epoch, **kwargs):
        pass

    @abstractmethod
    def on_train_end(self, fitter, **kwargs):
        pass
        

class MetricTrackerCB(ManualCB):
    """a callback that will calculated score based on the input metric"""
    def __init__(self, metric_obj:Metric, d_out_val:pd.DataFrame=None, **kwargs):
        """
        Args:
            metric_obj (Metric): the metric that is used to calculate scores
            d_out_val (pd.DataFrame, optional): out/TGT columns for the validation set. NOTE THAT YOUR VAL DATALOADER MUST RESULT IN SAME ORDER AS d_out_val. Defaults to None.
                                                This is used for calculating metrics that require a d_out. If your metric doesn't require it, no need to include it

        """
        super().__init__(**kwargs)
        self.metric_obj, self.d_out_val = \
            metric_obj, d_out_val
        self.metric_vs_epoch = []
        self.best = np.NINF if metric_obj.op == operator.gt else np.PINF if metric_obj.op == operator.lt else metric_obj.best 
        self.metric_lbl = metric_obj.metric_lbl
        self.metric_improved = False
        
    def on_epoch_end(self, epoch_info_val:dict=None, val_preds:List=None, val_lbls:List=None, *args, **kwargs)->None:
        """Calculate metric and keep track of it, noting if it's currently the best score so far.

        Args:
            epoch_info_val (dict): if included, val preds, lbls, and loss are extracted from here and overwrite what was passed in for them
            val_preds (List, optional): predictions on the validation set. These are calculated in fitter and passed from there. Defaults to None.
            val_lbls (List, optional): labels on the validation set, in same order as val_preds. Defaults to None.
        """        
        if epoch_info_val is not None:
            val_preds, val_lbls, vloss = epoch_info_val.get('preds', None), epoch_info_val.get('labels', None), epoch_info_val.get('loss', None)
            
        self.metric = vloss if isinstance(self.metric_obj, Loss) else self.metric_obj(val_preds, val_lbls, d_out=self.d_out_val)
        self.metric_vs_epoch.append(self.metric)
        current = self.metric
        if isinstance(current, torch.Tensor): current = current.cpu()
        if current is not None and self.metric_obj.op(current, self.best):
            self.best = current
            self.metric_improved=True
        else:
            self.metric_improved=False

class OptunaCB(ManualCB):
    def __init__(self, metric_cb:ManualCB, trial:optuna.trial.Trial, prune_mode:str='current', single_objective:bool=True):
        """ a callback for handling the requirement and need of optuna HPO

        Args:
            metric_cb(ManualCB): the metric tracker cb that stores and calculate scores. 
            trial(optuna.trial.Trial): the parameter passed in from objective function defined in the notebook. Read more on the optuna.trial.
            prune_mode(str, optional): the setting determines how optuna.Trial is reporting to the pruner. Current supporting two options:'best' or 'current'.
                If set to 'best', the optuna.Trial will only report the best score in this round of trial, it may save some trials that hits a high score early 
                from being pruned (they just goes down hill from that one hit, it will still get pruned if it does not catch to the stats of other trials. 
                If set to 'current', the optuna.Trial will report the score from the MetricTrackerCB on the current epoch, this could mean it is favoring the 
                trial that reaches peak later in the run. Default to 'current'
            single_objective(bool): whether this trial is using multiple objectives for optimizing. Note: when you are using multiple objective, Optuna pruner would not be working properly.
        """ 
        if prune_mode not in ['best','current']: raise ValueError('only accept best or current.')
        self.metric_cb, self.trial, self.prune_mode, self.single_objective = metric_cb, trial, prune_mode, single_objective
        
    def on_epoch_end(self, epoch:int, *args, **kwargs)->None:
        """handle the need for optuna pruner, which requires the current trial to "report" the score at the end of each training epoch.

        Args:
            epoch (int): the current epoch index.
        """  
        metric_score = self.metric_cb.metric if self.prune_mode is 'current' else self.metric_cb.best
        if self.single_objective: 
            self.trial.report(metric_score, epoch)
            if self.trial.should_prune():
                raise optuna.TrialPruned()
    
    def get_final_score(self):
        """ used in external call in the objective function. It will return the best score from all the epochs within this trial. 
        So sampler will use this score to study and draw samples for future trials, while the pruner will collect a general stats from each trial using this score.
        """
        return self.metric_cb.best

class SaveModelCBExt(ManualCB):
    """a callback that handles how to save after each epoch"""
    def __init__(self, metric_cb:ManualCB, every:str='best_only', name:str='epoch', parent_dir='.', quiet=True):
        """Instantiates callback to be used with a fitting function/object that follows the callback API of ManualCB

        Args:
            metric_cb(ManualCB): the metric tracker cb that is used for recording scores after each epoch
            every (str, optional): When to save the model. Either 'epoch' or 'best_only'. Defaults to 'best_only'.
            name (str, optional): Name to save model as. See `update_save_name` for where the model will actually get saved to. Defaults to 'epoch'.
            parent_dir (str, optional): Used for saving the model to the appropriate directory. See `update_save_name` for where the model will actually get saved to. Defaults to '.'.
            quiet (bool, optional): if False, adds more print statements. Defaults to True.
        """         
        self.metric_cb, self.every, self.name, self.quiet = \
            metric_cb, every, name, quiet
        self.save_name, self.metric_vs_epoch = None, []
        self.best = np.NINF if metric_cb.metric_obj.op == operator.gt else np.PINF if metric_cb.metric_obj.op == operator.lt else metric_cb.metric_obj.best 
        self.metric_lbl = metric_cb.metric_obj.metric_lbl
        self.metric_dir = f"{parent_dir}/{self.metric_lbl}" 
        if self.every not in ['best_only', 'improvement', 'epoch']:
            warn(f"SaveModel every '{self.every}' is invalid, falling back to every='epoch'.")
            self.every = 'epoch'
        os.makedirs(self.metric_dir, exist_ok=True)

    @classmethod
    def from_model_wrapper(cls, mw, metric_obj, **kwargs):
        return cls(metric_obj, d_out_val=mw.d.iloc[mw.val_i][mw.out_cols + mw.y_cols], **kwargs)

    def update_save_name(self, score, save_epoch:int):
        "Sets self.save_name appropriately given the current context. Save_epoch is the epoch being saved."
        self.save_name = f'{self.metric_dir}/{self.name}_{save_epoch}_{self.metric_lbl}={score:.4f}.pth'
        
    def on_epoch_end(self, fitter:object, epoch:int, *args, **kwargs)->None:
        """based on the save mode, determine the saving scheme.

        Args:
            fitter (object): an object that follows the Fitting API. It must have load/save functions
            epoch (int): the epoch that is ending (0-based, so the first epoch should be epoch 0)
        """      
        ### Save accordingly
        if self.every=="epoch": 
            self.update_save_name(self.metric_cb.metric , epoch)
            fitter.save(self.save_name)
        else: #every='best_only' or "improvement"
            if self.metric_cb.metric_improved:
                if not self.quiet: 
                    print('===========================================================================')
                    print(f'Better model found at epoch {epoch} with {self.metric_lbl} of: {self.metric_cb.best:.4f}.')
                    print('===========================================================================')
                if self.every=='best_only' and self.save_name is not None and os.path.exists(self.save_name): 
                    os.remove(self.save_name)
                self.update_save_name(self.metric_cb.best, epoch)
                fitter.save(self.save_name)

    def on_train_end(self, fitter:object, **kwargs):
        "Load the best model."
        if self.every=='best_only' or self.every=='improvement':
            if os.path.isfile(self.save_name):
                fitter.load(self.save_name)
                print('===========================================================================')
                print('===========================================================================')
                print(f"Training done, loaded best model: {self.save_name}")
                print('===========================================================================')
                print('===========================================================================')
            else:
                warn(f"Training done, but could not load best model because file {self.save_name} doesn't exist")

