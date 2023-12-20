from typing import List
import numpy as np
import torch
from torch import nn
import fastai.layers as flayers
from fastcore.utils import listify
from fastcore.utils import ifnone

class TabularModel(nn.Module):
    """Basic model for tabular data. Same as fastai's tab model except forward function expects x_cont first and both x_cont and x_cat are optional.
        Args:
            emb_szs (List[float]): Embedding sizes for categorical features. If empty list is passed in then no embeddings are created.
            n_cont (int): Number of continuous features.
            out_sz (int): output dim size.
            layers (List[int]): List of hidden layer sizes.
            ps (List[float], optional): List of probabilities for each layer's dropout. Defaults to None.
            emb_drop (float, optional): Probability of dropout for embedding layer. Defaults to 0..
            y_range (_type_, optional): Range of y. Defaults to None.
            use_bn (bool, optional): Switch to turn on batchnorm. Defaults to True.
            bn_final (bool, optional): Switch to turn on batchnorm in the final layer. Defaults to False.
            sigma (int, optional): Parameter for stochastic gates (see paper). Defaults to 0.5.
            lam (int, optional): Parameter for stochastic gates (see paper). Defaults to 0.01.
    """
    def __init__(self, emb_szs:List[float], n_cont:int, out_sz:int, layers:List[int], ps:List[float]=None,
                 emb_drop:float=0., y_range=None, use_bn:bool=True, bn_final:bool=False):
        super().__init__()
        if ps is None: ps = [0]*len(layers)
        ps = ifnone(ps, [0]*len(layers))
        ps = listify(ps, layers)
        self.embeds = nn.ModuleList([flayers.embedding(ni, nf) for ni,nf in emb_szs])
        self.emb_drop = nn.Dropout(emb_drop)
        self.bn_cont = nn.BatchNorm1d(n_cont)
        n_emb = sum(e.embedding_dim for e in self.embeds)
        self.n_emb,self.n_cont,self.y_range = n_emb,n_cont,y_range
        sizes = self.get_sizes(layers, out_sz)
        actns = [nn.ReLU(inplace=True) for _ in range(len(sizes)-2)] + [None]
        layers = []
        for i,(n_in,n_out,dp,act) in enumerate(zip(sizes[:-1],sizes[1:],[0.]+ps,actns)):
            layers += flayers.bn_drop_lin(n_in, n_out, bn=use_bn and i!=0, p=dp, actn=act)
        if bn_final: layers.append(nn.BatchNorm1d(sizes[-1]))
        self.layers = nn.Sequential(*layers)

    def get_sizes(self, layers, out_sz):
        return [self.n_emb + self.n_cont] + layers + [out_sz]

    def forward(self, x_cont:torch.Tensor=None, x_cat:torch.Tensor=None, **kwargs) -> torch.Tensor:
        if self.n_emb != 0:
            x = [e(x_cat[:,i]) for i,e in enumerate(self.embeds)]
            x = torch.cat(x, 1)
            x = self.emb_drop(x)
        if self.n_cont != 0:
            x_cont = self.bn_cont(x_cont.float())
            x = torch.cat([x, x_cont], 1) if self.n_emb != 0 else x_cont
        x = self.layers(x)
        if self.y_range is not None:
            x = (self.y_range[1]-self.y_range[0]) * torch.sigmoid(x) + self.y_range[0]
        return x

class FeatureSelector(nn.Module):
    def __init__(self, input_dim, sigma):
        super(FeatureSelector, self).__init__()
        self.mu = nn.Parameter(0.01*torch.randn(input_dim, ), requires_grad=True)
        self.noise = torch.randn(self.mu.size()) 
        self.sigma = sigma
    
    def forward(self, prev_x):
        z = self.mu + self.sigma*self.noise.normal_()*self.training 
        stochastic_gate = self.hard_sigmoid(z)
        new_x = prev_x * stochastic_gate
        return new_x
    
    def hard_sigmoid(self, x):
        return torch.clamp(x+0.5, 0.0, 1.0)

    def regularizer(self, x):
        ''' Gaussian CDF. '''
        return 0.5 * (1 + torch.erf(x / np.sqrt(2))) 

    def get_internal_loss(self):
        return torch.mean(self.regularizer((self.mu + 0.5)/self.sigma)) 

    def _apply(self, fn):
        super(FeatureSelector, self)._apply(fn)
        self.noise = fn(self.noise)
        return self

class FSTabularModel(TabularModel):
    def __init__(self, emb_szs:List[float], n_cont:int, out_sz:int, layers:List[int], ps:List[float]=None,
                 emb_drop:float=0., y_range=None, use_bn:bool=True, bn_final:bool=False, sigma:int=0.5):
        """This is the feature selection verison of the current TabularModel. If using this model then it is necessary to use 
        the FSFitter as alterations to the loss function are made. Based on this paper: 
        https://arxiv.org/pdf/1810.04247.pdf

        Args:
            emb_szs (List[float]): Embedding sizes for categorical features. If empty list is passed in then no embeddings are created.
            n_cont (int): Number of continuous features.
            out_sz (int): output dim size.
            layers (List[int]): List of hidden layer sizes.
            ps (List[float], optional): List of probabilities for each layer's dropout. Defaults to None.
            emb_drop (float, optional): Probability of dropout for embedding layer. Defaults to 0..
            y_range (_type_, optional): Range of y. Defaults to None.
            use_bn (bool, optional): Switch to turn on batchnorm. Defaults to True.
            bn_final (bool, optional): Switch to turn on batchnorm in the final layer. Defaults to False.
            sigma (int, optional): Parameter for stochastic gates (see paper). Defaults to 0.5.
            lam (int, optional): Parameter for stochastic gates (see paper). Defaults to 0.01.
        """
        super().__init__(emb_szs, n_cont, out_sz, layers, ps, emb_drop, y_range, use_bn, bn_final)
        self.feature_selector = FeatureSelector(self.n_emb + self.n_cont, sigma)
        self.sigma = sigma
        self.mu = self.feature_selector.mu

    def get_gates(self, mode):
        if mode == 'raw':
            return self.mu.detach().cpu().numpy()
        elif mode == 'prob':
            return np.minimum(1.0, np.maximum(0.0, self.mu.detach().cpu().numpy() + 0.5)) 
        else:
            raise NotImplementedError()

    def forward(self, x_cont:torch.Tensor=None, x_cat:torch.Tensor=None, **kwargs) -> torch.Tensor:
        x_cont = self.feature_selector(x_cont)
        return super().forward(x_cont, x_cat, **kwargs)
