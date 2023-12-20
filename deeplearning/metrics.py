from abc import abstractmethod
from warnings import warn
from types import FunctionType
from typing import Iterable

from . import util
from .loss_fxs import Loss as LossModule

import operator
import torch
import numpy as np
import sklearn.metrics as sk_metrics

def calc_xentropy_loss(preds:torch.Tensor, y:torch.LongTensor, w:torch.Tensor=None, mode:str='prob'):
    """Calculates cross-entropy loss either using probs or logits. The loss returned will be equal to that of 
    CrossEntropyLoss, but CrossEntropyLoss can only receive logits.
    
    This is a convenience function for when you want to calculate loss for models - it's not for being used
    for a model's loss function (since it is not a torch.nn.Module). 

    Args:
        preds (torch.Tensor): predictions (either logits or probabilities). Size should be (len(y), n_classes)
        y (torch.LongTensor): class labels.
        w (torch.Tensor, optional): per-sample weights. If None, no sample-wise weighting will be performed. Defaults to None.
        mode (str, optional): Whether `preds` are logits or probabilities. Either 'prob' or 'logit'; defaults to 'prob'.

    Raises:
        ValueError: If mode is not one of 'prob' or 'logit'.

    Returns:
        torch.Tensor: The mean cross-entropy loss of `preds` given class labels `y`
    """    
    if w is None: w = torch.ones(len(y))
    if mode == 'prob':
        return (torch.nn.NLLLoss(reduction='none')(torch.log(preds+1e-20), y) * w).mean()
    elif mode == 'logit':
        return (torch.nn.CrossEntropyLoss(reduction='none')(preds, y) * w).mean()
    else:
        raise ValueError('Mode must be either prob or logit')

def _check_sizes(y_pred:Iterable, other:Iterable) -> None:
    """Raises and error if len(y_pred) != len(other). 
    This most often would occur if you give the wrong d_out for y_pred (ie you are predictin on test but provided d_val).

    Args:
        y_pred (Iterable): predictions
        other (Iterable): thing to compare predictions with. Usually either labels or d_out

    Raises:
        ValueError: If len(y_pred) != len(other)
    """    
    if len(y_pred) != len(other):
        raise ValueError(f"len(y_pred) = {len(y_pred)} but len(other) = {len(other)}; these must be equal. "+\
                            "Perhaps you're trying to calculate test metrics and did not instantiate a metric object with d_out as the test set?")  

class Metric():
    "Parent class for all Metrics. This defines the Metrics API"
    def __init__(self, op:FunctionType=operator.gt, metric_lbl:str='metric', split_into_loss_weight:bool=True, **kwargs) -> None:
        """Instantiates metric object.

        Args:
            op (FunctionType, optional): What operator to associate with the metric. If higher is better, this should be operator.gt ('>'). 
                                         If lower is better, this should be operator.lt ('<'). Defaults to operator.gt.
            metric_lbl (str, optional): The label or title for this metric. This can be used extensively (for example, for saving best models). Defaults to 'metric'.
        """        
        self.op, self.metric_lbl, self.split_into_loss_weight = op, metric_lbl, split_into_loss_weight

    def calc_metric(self, y_pred:Iterable, y_true:Iterable, **kwargs) -> float:
        """Calculates the metric in question. This is also the object's __call__ function.

        This is more of a wrapper around the actual self._calc_metric private function. This wrapper function just does some prep and checks before 
        calling self._calc_metric. This follows the template method (https://en.wikipedia.org/wiki/Template_method_pattern).

        Specifically, this function converts y_pred to a float tensor and y_true to a long tensor, extracts the loss weights from y_true if they are there,
        checks that y_pred and y_true are the same length, and then calls self._calc_metric with these updated y_pred, y_true, and w values.

        Args:
            y_pred (torch.FloatTensor or np.ndarray): predictions
            y_true (torch.LongTensor or np.ndarray): labels

        Returns:
            (float): The value of the metric
        """        
        y_pred, y_true = torch.as_tensor(y_pred, dtype=torch.float), torch.as_tensor(y_true)
        if self.split_into_loss_weight: 
            y_true, w = util.split_loss_weight(y_true)
            kwargs['w'] = w
        _check_sizes(y_pred, y_true)
        return self._calc_metric(y_pred, y_true, **kwargs)

    @abstractmethod
    def _calc_metric(self, y_pred:torch.FloatTensor, y_true:torch.LongTensor, **kwargs):
        """Every child class must implement this function, and it is what actually calculates the specific metric.
        
        It's best to not have any **kwargs that are critical to _calc_metric's operation; any info other than y_pred, y_true, or w that is critical to 
        calculating the metric should be added to the metric's __init__ function and accessed here using class attributes. This is because such additional args 
        do not follow the API, and other places that use metrics are not guaranteed to pass any such **kwargs to calc_metric. (Such as SaveModelCBExt.on_epoch_end.)
        
        See TrimAPR as an example. It's _calc_metric function requires knowing d_out and long_short. Rather than passing these to _calc_metric,
        they are provided in the metric's init function and accessed with self.<var>. You'll notice _calc_metric does have a kwarg `ret_all`, 
        but this is non-critical and just provides a non-default way to return things, and thus is acceptable.

        Args:
            y_pred (torch.FloatTensor): predictions (logits)
            y_true (torch.LongTensor): labels
        """        
        pass

    def pred_to_probs_class1(self, y_pred:torch.FloatTensor) -> torch.FloatTensor:
        """Converts 2-dimensional logits predictions to 1-dimensional probabilities for the positive class (class1).
        If y_pred is already 1-d, this function assumes y_pred is already probs_class1 and therefore returns y_pred as it was passed in.

        Args:
            y_pred (torch.FloatTensor): 2-dimensional logits predictions

        Returns:
            torch.FloatTensor: 1-dimensional probability predictions for the positive class
        """        
        if y_pred.ndim == 1: # assumes y_pred is already probs_class1 in this case  
            if (y_pred > 1.0).any() or (y_pred < 0.0).any():
                raise ValueError("y_pred is one-dimensional and is not in probabilities. Y_pred should either be 2-d and be the pre-softmax predictions or be 1-d and be the post-softmax class1 probabilities")
            return y_pred 
        if not hasattr(self, 'softmax'): self.softmax = torch.nn.Softmax(dim=1) 
        probs_class1 = self.softmax(y_pred)[:,1]
        return probs_class1

    def __call__(self, y_pred, y_true, **kwargs):
        "Executes self.calc_metric"
        return self.calc_metric(y_pred, y_true, **kwargs)

class Loss(Metric):
    "Calculates loss given a Loss module object (see training.loss_fxs.Loss)"
    def __init__(self, loss_module:LossModule, split_into_loss_weight:bool=True, strip_d_out:bool=False, **kwargs) -> None:
        """Calculates any loss given a loss_module that inherits from Loss.

        Args:
            loss_module (LossModule): the loss module object used to calculate the desired loss from y_pred and y_true
            split_into_loss_weight(bool, optional): If True, then util.split_loss_weight is used in an attempt to split y_true into label and loss weight
            strip_d_out(bool, optional): If True, then 'd_out' is removed from the kwargs passed to _calc_metric before sending them to self.loss_module
        """        
        metric_lbl = loss_module.__class__.__name__
        super().__init__(op=operator.lt, metric_lbl=metric_lbl, split_into_loss_weight=split_into_loss_weight, **kwargs)
        self.loss_module, self.strip_d_out = loss_module, strip_d_out
        
    def _calc_metric(self, y_pred: torch.FloatTensor, y_true: torch.LongTensor, **kwargs):
        if self.strip_d_out:
            kwargs.pop('d_out', None)
        return self.loss_module(y_pred, y_true, **kwargs).item()


class CELoss(Metric):
    "Cross Entropy Loss metric, optionally with loss weighting"
    def __init__(self, weighted:bool=False, **kwargs) -> None:
        """Instantiates CELoss metric

        Args:
            weighted (bool, optional): If true, will do sample-wise loss weighting. Defaults to False.
        """        
        warn("CELoss is outdated. You should be able to use the more generic Loss metric for CELoss")
        metric_lbl = 'w_CELoss' if weighted else 'CELoss'
        super().__init__(op=operator.lt, metric_lbl=metric_lbl, **kwargs)
        self.weighted = weighted

    def _calc_metric(self, y_pred:torch.FloatTensor, y_true:torch.LongTensor, w:torch.FloatTensor=None, **kwargs) -> float:
        """Caculates the CELoss metric, optionally with loss weighting


        Args:
            y_pred (torch.FloatTensor): predictions (logits)
            y_true (torch.LongTensor): labels
            w (torch.FloatTensor, optional): loss weights per sample. Defaults to None.

        Returns:
            float: cross-entropy loss, optionally weighted.
        """        
        if not self.weighted: w = None
        return calc_xentropy_loss(y_pred, y_true, w=w, mode='logit').item()


class Accuracy(Metric):
    "Accuracy metric"
    def __init__(self, threshold:float=0.5) -> None:
        """Instantiates accuracy metric object

        Args:
            threshold (float, optional): class probability threshold. Defaults to 0.5.
        """        
        self.threshold = threshold
        super().__init__(metric_lbl='accuracy')

    def _calc_metric(self, y_pred:torch.FloatTensor, y_true:torch.LongTensor, **kwargs) -> float:
        """Calculates accuracy metric

        Args:
            y_pred (torch.FloatTensor): predictions (logits)
            y_true (torch.LongTensor): labels

        Returns:
            float: accuracy metric, as defined by sk_metrics.accuracy_score
        """        
        probs_class1 = self.pred_to_probs_class1(y_pred)
        return sk_metrics.accuracy_score(y_true, probs_class1 > self.threshold)

class AUC(Metric):
    "Area under the ROC Metric"
    def __init__(self) -> None:
        super().__init__(metric_lbl='AUC')
    
    def _calc_metric(self, y_pred:torch.FloatTensor, y_true:torch.LongTensor, **kwargs) -> float:
        """Calculates AUC metric

        Args:
            y_pred (torch.FloatTensor): predictions (logits)
            y_true (torch.LongTensor): labels

        Returns:
            float:AUC metric, as defined by sk_metrics.roc_auc_score
        """         
        probs_class1 = self.pred_to_probs_class1(y_pred)
        return sk_metrics.roc_auc_score(y_true, probs_class1)
    
class Last(Metric):
    """Simply saves the last model during training. This saves a model at every epoch.
    """
    def __init__(self, **kwargs) -> None:
        """ No args required. 
        """             
        super().__init__(op=operator.eq, metric_lbl='last', **kwargs)
        self.best=0.0
        
    def _calc_metric(self, y_pred:torch.FloatTensor, y_true:torch.LongTensor,  **kwargs) -> float:
        """Saves the model at every epoch.

        Args:
            y_pred (torch.FloatTensor): _description_
            y_true (torch.LongTensor): _description_

        Returns:
            float: 0.0, this value does not change.
        """        
        return 0.0
    
class Entropy(Metric):
    "This is the Shannon information entropy calculation for all data samples. The lower the expected/average entropy the more 'certain' the model is about the data."
    def __init__(self, **kwargs) -> None:
        """Instantiates CELoss metric

        Args:
            
        """        
        metric_lbl = 'entropy'
        super().__init__(op=operator.lt, metric_lbl=metric_lbl, **kwargs)

    def _calc_metric(self, y_pred:torch.FloatTensor, y_true:torch.LongTensor, **kwargs) -> float:
        """Calculates the expected entropy

        Args:
            y_pred (torch.FloatTensor): _description_
            y_true (torch.LongTensor): _description_

        Returns:
            float: Shannon entropy as a measure of certainty
        """      
        
        attn_wt_probs = torch.softmax(y_pred, -1)
        entropy = -torch.sum((attn_wt_probs)*torch.log2(attn_wt_probs), -1)
        return torch.mean(entropy).item()