# pylint: disable=import-error
from types import FunctionType
from warnings import warn
from . import models
from . import util
from typing import List

import time, gc
import torch, numpy as np
from transformers import AdamW

from . import util
from typing import Union
from contextlib import nullcontext

class Predictor():
    """A wrapper around a model that handles predicting on novel data. This can be instantiated after a model has been fitted. 
    Do not use this class for any operations requiring updates to the model (e.g. when training).
    """
    def __init__(self, model, device=None) -> None:
        self.model = model
        self.to(device)

    def switch_device(self, device):
        "Switches self.model to device and updates self.device"
        if not isinstance(device, torch.device):
            device = util.pick_device(device)
        self.device = device
        self.model.to(device)
        return self # to enable chaining
        
    def to(self, device): # left 'switch_device' method for now for backwards comapatibility
        return self.switch_device(device)

    def predict(self, x:Union[dict, torch.Tensor, np.ndarray], softmax=False, ret_np:bool=True, with_dropout=False, unpack_x=False, **forward_kwargs) -> Union[torch.Tensor, np.ndarray]:
        """Gets model predictions given some inputs. Ensures model is in eval mode, that the context is with torch.no_grad, 
        and that model and input are on same device. 

        Args:
            x (Union[torch.Tensor, np.ndarray]): feature values to predict on. If not a Tensor, will be cast to one and sent to self.device
            softmax (bool, optional): If True, softmaxes predictions before returning them. Defaults to False.
            ret_np (bool, optional): If True, will detach predictions and convert them to numpy. Defaults to True.
            with_dropout (bool, optional): If True, predictions are run with all dropout modules set to training mode (ie, with dropout on). 
                                            After predicting, dropout is immediately turned back off to ensure no accidental persistence of dropout.
                                            Predicting with dropout on at inference time is useful if trying to do a bayesian approximation.
            unpack_x (bool, optional): If True, then the call to self.model(x, ...) will actually be self.model(**x, ...)

        Returns:
            Union[torch.Tensor, np.ndarray]: model predictions on x (which are probabilities if `softmax` otherwise they're logits)
        """        
        self.model.eval()
        with torch.no_grad():
            if type(x) != dict and type(x) != torch.Tensor: x = torch.Tensor(x)
            x = util.move_obj_to_device(x, self.device)
            context_manager = util.DropoutOnCM(self.model) if with_dropout else nullcontext()
            with context_manager:
                preds = self.model(**x, **forward_kwargs) if unpack_x else self.model(x, **forward_kwargs)
        if softmax: preds = torch.nn.Softmax(dim=1)(preds)
        if ret_np: preds = preds.detach().cpu().numpy()
        return preds

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def load_model(self, ckpt:Union[str, dict], strict:bool=True):
        """Loads model state dict from a checkpoint object or checkpoint file at ckpt into self.model

        Args:
            ckpt (Union[str, dict]): either a checkpoint dictionary object or a path to a saved checkpoint object. This checkpoint dict must have at least the key 'model_sd'.
            strict (bool, optional): if strict loading (see pytorch's load_state_dict) should be used for the model's state dict. Defaults to True.
        """        
        if isinstance(ckpt, str): ckpt = torch.load(ckpt, map_location=self.device)
        self.model.load_state_dict(ckpt['model_sd'], strict=strict)
        return self

class EvidentialPredictor(Predictor):
    def predict(self, x: Union[torch.Tensor, np.ndarray], ret_np: bool=True, evidence_fx=torch.nn.functional.relu) -> Union[torch.Tensor, np.ndarray]:
        self.model.eval()
        with torch.no_grad():
            if type(x) != torch.Tensor: x = torch.Tensor(x)
            if x.device != self.device: x = x.to(self.device)
            preds = self.model(x)

        # Probabilities are treated as the expected value of a parametrized Dirichlet distribution, where the alpha parameter is derived from NN outputs
        evidence = evidence_fx(preds)
        alpha = evidence + 1
        num_classes=alpha.shape[-1]
        uncertainty = num_classes / torch.sum(alpha, dim=1, keepdim=True)
        probs = alpha / torch.sum(alpha, dim=1, keepdim=True)
        if ret_np: 
            probs = probs.detach().cpu().numpy()
            uncertainty = uncertainty.detach().cpu().numpy()
        return probs, uncertainty


class Fitter:
    """Class used to fit a model. Extensible callback system used.
    
    Returns:
        {object} -- Instantiated fitter object
    """    
    def __init__(self, model:torch.nn.Module, train_dl:torch.utils.data.DataLoader, val_dl:torch.utils.data.DataLoader, loss_fx:torch.nn.Module, grad_clip:float, \
                device:torch.device, lr_sched_factory:FunctionType, lr_kwargs:dict={}, callbacks:List[object]=[], opt_factory=None, opt_kwargs={}, del_opt=True, quiet:bool=True):
        """Prepares class attributes. Sends model to device and then creates optimizer.

        Args:
            model (torch.nn.Module): model to fit.
            train_dl (torch.utils.data.DataLoader): train dataloader. Each row from the loader must be a tuple of 3 elements - x,y, and z. Only x and y are used
            val_dl (torch.utils.data.DataLoader): validation dataloader. Each row from the loader must be a tuple of 3 elements - x,y, and z. Only x and y are used
            loss_fx (torch.nn.Module): Loss function / criterion for backprop
            grad_clip (float): What to clip the gradients to when training.
            device (torch.device): device to perform operations on
            lr_sched_factory (func): factory function that returns a lr scheduler with a step() function. This is used to generate a scheduler every time self.fit is called. 
                                     Called like so: lr_sched_factory(self.opt, **lr_kwargs).
            lr_kwargs (dict, optional): kwargs for lr_sched_factory. 
            callbacks (List[object], optional):  List of ManualCB objects. Defaults to [].
            opt_factory (func, optional): factory function for creating an optimizer if you do not want the one created by self.default_opt_factory. 
                                          Dev note: The only reason for a factory function instead of just passing an instantiated opt is because according to PyTorch docs the optimizer
                                          needs to be made *after* sending the model to device, which we handle inside of fitter. This may not be true (see link below), but better safe than sorry.
                                          https://discuss.pytorch.org/t/effect-of-calling-model-cuda-after-constructing-an-optimizer/15165
            opt_kwargs (dict, optional): if opt_factory isn't None, these are all of the args for opt except the first arg (model). If opt_factory is None, these are additional
                                         kwargs for self.default_opt_factory.
            del_opt (bool, optional): If True, self.opt is deleted after calling the fit function (to save space). Set to False if you need access to the optimizer post-fitting
            quiet (bool, optional): Run in quiet mode. If True, no epoch losses will be printed throughout training. Defaults to True.
        """        
        self.model, self.train_dl, self.val_dl, self.loss_fx, self.grad_clip, self.device, self.callbacks, self.lr_sched_factory, self.lr_kwargs, self.del_opt, self.quiet = \
            model, train_dl, val_dl, loss_fx, grad_clip, device, callbacks, lr_sched_factory, lr_kwargs, del_opt, quiet
        self.checkpoint, self.train_loss_per_epoch, self.val_loss_per_epoch = {}, [], []
        self.opt_factory = opt_factory if opt_factory is not None else self.default_opt_factory
        self.to(device)
        self.opt = self.opt_factory(self.model.parameters(), **opt_kwargs)
        
    def restore_fitter(self, ckptpath, model:torch.nn.Module, train_dl:torch.utils.data.DataLoader, val_dl:torch.utils.data.DataLoader, loss_fx:torch.nn.Module,\
            del_opt=True, opt_kwargs={}, device=None, quiet=True):
        """Restores everything that was deleted at end of fit function so that you can continue training if desired. 
        self.opt is re-created using self.opt_factory. Everything else must be passed in. State dicts for model and optimizer are loaded from ckptpath
        
        Example of how to use:
            predictor = fitter.fit(n_epochs=n_epochs)
            # you realize after training for n_epochs that you want to train for more than n_epochs but don't want to start over
            ckptpath = '/.../AUC/iter0_epoch_8_AUC=0.5502.pth' # epoch to continue from
            epoch_offset = fitter.restore_fitter(ckptpath, model, train_dl, val_dl, loss_fx, opt_kwargs=opt_kwargs, quiet=False)
            additonal_epochs = 5
            lr_kwargs['n_epochs']=additional_epochs
            predictor = fitter.fit(n_epochs=additional_epochs, epoch_offset=epoch_offset, lr_kwargs=lr_kwargs)

        Args:
            ckptpath (str): path to saved fitter checkpoint file
            model (torch.nn.Module): instantiated model object. State dict in checkpoint file will be loaded in
            train_dl (torch.utils.data.DataLoader): train dataloader
            val_dl (torch.utils.data.DataLoader): val dataloader
            loss_fx (torch.nn.Module): instantiated loss object for fitting
            opt_kwargs (dict, optional): kwargs for optimizer instantation. Defaults to {}.

        Returns:
            int: epoch number, to be used as epoch_offset for fitter.fit
        """        
        warn("WARNING: This function is experimental and not totally tested. User beware")
        self.model, self.train_dl, self.val_dl, self.loss_fx, self.del_opt, self.quiet = model, train_dl, val_dl, loss_fx, del_opt, quiet
        if device is None: device = self.device
        self.to(device)
        self.opt = self.opt_factory(self.model.parameters(), **opt_kwargs)
        self.load(ckptpath)
        epoch = self.checkpoint['epoch']
        # TODO - reset trackers to end at the epoch you load in so you continue where you left off
        self.train_loss_per_epoch = self.train_loss_per_epoch[:epoch]
        self.val_loss_per_epoch = self.val_loss_per_epoch[:epoch]
        ### TODO - figure out how to reset metric callbacks ('cb.metric_vs_epoch')
        return epoch

    def to(self, device):
        "Moves self.model to the device if it's not already there"
        self.model.to(device)
        self.device = device
                
    def default_opt_factory(self, model_params, **upd_kwargs):
        """Returns an AdamW optimizer.

        Args:
            upd_kwargs (dict, optional): Parameters for AdamW constructor different/in addition to our defaults. 
                Our defaults are betas=(0.9,0.99), weight_decay=1e-2, correct_bias=True. Defaults to {}.

        Returns:
            torch.optim: insantiated AdamW optimizer
        """        
        opt_kwargs=dict(betas=(0.9,0.99), weight_decay=1e-2, correct_bias=True)
        opt_kwargs.update(upd_kwargs)
        return AdamW(model_params, **opt_kwargs)

    def _predictor_factory(self, model, device):
        "Factory function for creating the appropriate predictor object"
        return Predictor(model, device)

    def _batch_to_device(self, batch):
        "How to send a batch to the device"
        return util.move_obj_to_device(batch, self.device)

    def _batch_to_model(self, batch):
        "Predicts on a batch. B/c batches might need to be extracted differently, override this function when necessary"
        return self.model(**batch)

    def _labels_from_batch(self, batch):
        "Extracts labels from a batch. B/c batches might need to be extracted differently, override this function when necessary"
        return batch['label']
    
    def _define_batch_info(self, loss:torch.Tensor, preds:torch.Tensor, labels:torch.Tensor, *args, **kwargs):
        """defines what information from a batch to collect across batches for the whole epoch.

        Args:
            loss (torch.Tensor): loss tensor for the current batch (has not yet called .item() on it).
            preds (torch.Tensor): predictions for the current
            labels (torch.Tensor): labels for the current batch

        Returns:
            dict: dictionary defining the batch. MUST have at least 'loss' key.
        """        
        return {'loss': loss.item(), 'preds': preds, 'labels': labels}
    
    def _collect_batch(self, epoch_info:dict, loss:torch.Tensor, preds:torch.Tensor, labels:torch.Tensor, *args, **kwargs):
        """Updates epoch_info with the current batch's loss, and optionally with its preds and labels as well

        Args:
            epoch_info (dict): the dictionary that contains the epoch's information (loss and optionally preds/labels) from each batch
            loss (torch.Tensor): loss tensor for the current batch.
            preds (torch.Tensor, optional): predictions for the current batch, if you want to collect them.
            labels (torch.Tensor, optional): labels for the current batch.
            *args: other arguments for self.define_batch_info for child classes to use.
            *kwargs: keyword arguments for self.define_batch_info for child classes to use.
            

        Returns:
            dict: updated epoch_info dict with the info from this batch
        """                
        batch_info = self._define_batch_info(loss, preds, labels, *args, **kwargs)
        for k,v in batch_info.items():
            if k not in epoch_info:
                epoch_info[k] = [v]
            else:
                epoch_info[k].append(v)
        return epoch_info
            
    def _finalize_epoch_results(self, epoch_info:dict, **kwargs):
        """Once info across batches has been gathered, perform things across the whole epoch to finalize the epoch results before returning.

        Args:
            epoch_info (dict): dictionary of epoch information. Must have at least 'loss' key

        Returns:
            dict: Finalized epoch results dict. In the parent's case, this is where the loss is the mean, and if preds and labels are present, 
                they are concatenated across dimension zero and are numpy arrays
        """        
        epoch_info['loss'] = np.mean(epoch_info['loss'])
        if 'preds' in epoch_info:
            epoch_info['preds'] = torch.cat(epoch_info['preds']).detach().cpu().numpy() # TODO - is there a reason we typecast to numpy? If not, leave as tensor
        if 'labels' in epoch_info:
            epoch_info['labels'] = torch.cat(epoch_info['labels']).detach().cpu().numpy()
        return epoch_info

    def _process_one_epoch(self, dloader:torch.utils.data.DataLoader, update:bool=False, quiet:bool=True):
        """ Internal function for processing an epoch, batch by batch. If in update mode, this means computed gradients are backpropagated.
            You likely will not need to override this function anymore; instead, you can probably just override `_batch_to_model` and/or `_labels_from_batch`

            Args:
                dloader (torch.utils.data.DataLoader): dataloader to process for the epoch
                update (bool, optional): whether or not to update the model based on the epoch. If False, runs in eval mode. Defaults to False.
                quiet (bool, optional): If False, more print statements. Defaults to True.
                collect_loss_only (bool, optional): If True, then the dictionary returned will only have 'loss'. Otherwise, keys are ['loss', 'preds', 'labels']

            Returns:
                (dict): dictionary of information on the epoch. if collect_loss_only, key will just be 'loss'. if False, keys are ['loss', 'preds', 'labels']
        """
        epoch_info = {}
        for batch in dloader:
            self.opt.zero_grad()
            batch = self._batch_to_device(batch)
            preds = self._batch_to_model(batch)
            labels = self._labels_from_batch(batch)
            # handle loss
            loss = self.loss_fx(preds, labels)
            assert not torch.isnan(loss), f"loss is NaN, batch: \n{batch}\n preds: \n{preds}\n labels: \n{labels}"
            if update: 
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.opt.step()
                self.lr_scheduler.step()
            
            # Collect Batch
            epoch_info = self._collect_batch(epoch_info, loss, preds, labels)
            del preds
            
        epoch_results = self._finalize_epoch_results(epoch_info)
        return epoch_results

    def process_one_epoch(self, dloader:torch.utils.data.DataLoader, update:bool=False, **kwargs):
        """Wrapper for _process_one_epoch so that it runs in the right context (if not training, should run with torch.no_grad)

        Args:
            dloader (torch.utils.data.DataLoader): dataloader to process for the epoch
            update (bool, optional): whether or not to update the model based on the epoch. If False, runs in eval mode and with torch.no_grad context. Defaults to False.
            **kwargs: passed to _process_one_epoch

        Returns:
            If with_loss:
                (np.array, np.array, float): preds, labels, average loss
            else:
                (np.array, np.array): preds, labels.
        """
        if update: 
            self.model.train()
            if hasattr(self.model, 'reset'): self.model.reset()
            return self._process_one_epoch(dloader, update=update, **kwargs)   
        else:
            self.model.eval()
            with torch.no_grad():
                return self._process_one_epoch(dloader, update=update, **kwargs)

    def fit(self, n_epochs:int, epoch_offset:int=0, lr_kwargs={}, ret_self:bool=False):
        """Fits 'self.model' using the optimizer 'self.opt' for 'self.n_epochs'. 

        Args:
            n_epochs (int): how many epochs to train for
            epoch_offset (int, optional): If continuing training from a previous session, this is how many epochs you trained for previously. 
                                          This is mostly for saving checkpoints appropriately. Defaults to 0.
            collect_loss_only (bool, optional): If True, then process_one_epoch will return a dictionary with only 'loss'. 
                Otherwise, keys are ['loss', 'preds', 'labels']. If collect_loss_only, then your callbacks should need only the loss and not the preds/labels.

        Returns:
            object: this fitter object. The only reason this returns anything is in case you need to fit in parallel.
        """        
        self.lr_kwargs.update(lr_kwargs)
        self.lr_scheduler = self.lr_sched_factory(self.opt, **self.lr_kwargs)
        if not self.quiet: print("Beginning fitting loop")
        train_start = time.time()
        for epoch in range(epoch_offset, n_epochs+epoch_offset):
            if not self.quiet: print("Beginning to process training set")
            epoch_info_train = self.process_one_epoch(self.train_dl, update=True, quiet=self.quiet)
            if not self.quiet: print("Beginning to process validation set")
            epoch_info_val = self.process_one_epoch(self.val_dl, update=False, quiet=self.quiet)
            
            self.train_loss_per_epoch.append(epoch_info_train['loss'])
            self.val_loss_per_epoch.append(epoch_info_val['loss'])
            self.checkpoint.update({
                'model_sd': self.model.state_dict(),
                'opt': self.opt.state_dict(),
                'epoch': epoch,
                'vloss': epoch_info_val['loss']
            })
            
            for cb in self.callbacks:
                cb.on_epoch_end(fitter=self, epoch=epoch, epoch_info_val=epoch_info_val)
            
            # Print val and training loss every percent training complete
            if not self.quiet and epoch % (max(1, n_epochs // 100)) == 0:
                print(f"=========== Epoch [{epoch}/{n_epochs+epoch_offset}], TRAIN loss: {epoch_info_train['loss']:.4f}, VAL loss: {epoch_info_val['loss']:.4f} ===========")
            
            del epoch_info_train, epoch_info_val
            gc.collect()

        train_end = time.time()
        for cb in self.callbacks:
            cb.on_train_end(self)
        minutes_trained = (train_end-train_start) / 60
        if not self.quiet:
            print("\n\nTraining done")
            print(f'Elapsed time for training: {minutes_trained:.1f} minutes. Average time per epoch: {minutes_trained / n_epochs:.1f} minutes')
        
        predictor = self._predictor_factory(self.model, self.device)

        del self.model, self.train_dl, self.val_dl, self.checkpoint, self.loss_fx
        if self.del_opt: del self.opt
        gc.collect()
        
        if ret_self: return predictor, self # need to return if doing in parallel and you want access to the fitter object
        else: return predictor

    def save(self, save_name:str):
        "Saves self.checkpoint to a file called save_name"
        torch.save(self.checkpoint, save_name)

    def load(self, ckptpath:str, strict:bool=True):
        """Loads checkpoint, model, and opt into self.checkpoint, self.model's state dict, and self.opt's state dict.

        Args:
            ckptpath (str): path to a saved checkpoint object. This checkpoint dict must have at least the keys 'model_sd' and 'opt'.
            strict (bool, optional): if strict loading (see pytorch's load_state_dict) should be used for the model's state dict. Defaults to True.
        """        
        self.checkpoint = util.load_from_ckpt(ckptpath, self.model, self.opt, strict, ret_ckpt=True)


class EvidentialFitter(Fitter):
    def _one_hot_embedding(self, labels, num_classes):
        y = torch.eye(num_classes)
        return y[labels]

    def _predictor_factory(self, model, device):
        return EvidentialPredictor(model, device)

    def _process_one_epoch(self, dloader:torch.utils.data.DataLoader, update:bool=False, quiet:bool=True):
        """Internal function for processing an epoch, batch by batch. If in update mode, this means computed gradients are backpropagated.

        Args:
            dloader (torch.utils.data.DataLoader): dataloader to process for the epoch
            update (bool, optional): whether or not to update the model based on the epoch. If False, runs in eval mode. Defaults to False.
            quiet (bool, optional): If False, more print statements. Defaults to True.

        Raises:
            NotImplementedError: If y (the labels) from the data loader are multi-dimensional. Still need to figure out how we are going to handle z (d_out) data

        Returns:
            (np.array, np.array): preds, labels.
        """        
        epoch_info = {}
        if not hasattr(self, 'evidential_epoch'): 
            self.evidential_epoch=0
        else:
            self.evidential_epoch+=1
        for x,y,z in dloader:
            self.opt.zero_grad()
            x, y = x.to(self.device), y.to(self.device)
            # Do forward pass, calculate loss
            nn_outputs = self.model(x)
            y_onehot = self._one_hot_embedding(y, num_classes=nn_outputs.shape[-1]) 
            loss = self.loss_fx(nn_outputs, y_onehot.float(), self.evidential_epoch, self.device)
            if update:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.opt.step()
                self.lr_scheduler.step()
            
            # Collect Batch
            epoch_info = self._collect_batch(epoch_info, loss, nn_outputs, y)
            del preds, x, y, z
            
        epoch_results = self._finalize_epoch_results(epoch_info)
        return epoch_results