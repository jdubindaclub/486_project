from abc import abstractmethod
from typing import Iterable
import torch

class Loss(torch.nn.Module):
    "Parent class for all custom Loss modules"
    @abstractmethod
    def forward(self, y_hat, y, **kwargs):
        """the forward call for the loss function. Child classes must override this but follow this function signature.
        Of course, you may not always need y_hat or y, but callers of the loss function will often pass these in so you must be able to accept them

        Args:
            y_hat (Iterable): predictions
            y (Iterable): labels
        """        
        pass

class CombinedSupervisedLosses(Loss):
    "Takes the weighted sum of any number of supervised loss modules, using `lambdas` as the weights"
    def __init__(self, *supervised_losses:Iterable[callable], lambdas:Iterable[float]=None) -> None:
        """
        Args:
            supervised_losses (Iterable[callable]): arbitrary number of supervised loss objects that are callable and whose forward function accepts
                y_hat, y as args.
            lambdas (Iterable[float], optional): weights for each supervised loss in the weighted sum. If None, weights are set to 1.0 for all losses.
                Defaults to None.
        """        
        super().__init__()
        if lambdas is None: lambdas = torch.ones(len(supervised_losses))
        if len(lambdas) != len(supervised_losses): raise ValueError("`lambdas` and `supervised_losses` must be of the same length")
        self.supervised_losses, self.lambdas = supervised_losses, lambdas

    def forward(self, y_hat, y, **kwargs):
        "returns weighted sum of supervised losses using self.lambdas as weights"
        return sum([self.lambdas[i] * loss(y_hat, y) for i,loss in enumerate(self.supervised_losses)])


class CombinedInternalLosses(Loss):
    """Combines a supervised loss with several internal losses from a list of modules that support the `get_internal_loss` property.
    
    This is an interesting one that should be used with care. The modules that support internal_loss must appropriately set the internal loss at some point in
    the fitting process for each batch (e.g., the forward function is an excellent place for internal_loss to be set). Much burden is set on the user for this;
    it is not a terribly robust way of doing it. TODO: future work is required to make this more robust and less error-prone.
    """
    def __init__(self, *modules_with_loss:Iterable[torch.nn.Module], lambdas:Iterable[float]=None, supervised_loss:callable=torch.nn.CrossEntropyLoss()) -> None:
        """
        Args:
            modules_with_loss (Iterable[torch.nn.Module]): arbitrary number of torch modules that each have the `internal_loss` property.
            lambdas (Iterable[float], optional): weights for internal losses from modules_with_loss. If None, defaults to 1.0 for all modules. Defaults to None.
            supervised_loss (callable, optional): loss module for the supervised loss portion. Defaults to torch.nn.CrossEntropyLoss().

        Raises:
            ValueError: all `modules_with_loss` must have the `get_internal_loss` method
            ValueError:  `modules_with_loss` and `lambdas` must have the same length
        """                
        super().__init__()
        for module in modules_with_loss:
            if not hasattr(module, 'get_internal_loss'):
                raise ValueError("Each of `modules` must have the attribute `get_internal_loss`")
        if lambdas is None: lambdas = torch.ones(len(modules_with_loss))
        if len(lambdas) != len(modules_with_loss): raise ValueError("`lambdas` and `modules_with_loss` must be of the same length")
        self.modules_with_loss, self.lambdas, self.supervised_loss = modules_with_loss, lambdas, supervised_loss
        
    def forward(self, y_hat, y, verbose=False, **kwargs):
        "Calculates self.supervised_loss and then adds it to the weighted sum of the internal losses of self.modules_with_loss"
        supe = self.supervised_loss(y_hat, y)
        internal_losses = [module.get_internal_loss() for module in self.modules_with_loss]
        # combined = supe + sum([self.lambdas[i] * l for i,l in enumerate(internal_losses)]) # This is probably slightly slower but i think takes less GPU?
        combined = supe + torch.Tensor(internal_losses).to(supe.device) * torch.Tensor(self.lambdas).to(supe.device)
        if verbose: 
            print('-'*30)
            print('Supervised loss:', supe)
            for i,l in enumerate(internal_losses): print('Lambda: ', self.lambdas[i], '| Internal loss:', l)
            print('combined:', combined)
        return combined

def relu_evidence(y):
    return torch.nn.functional.relu(y)

def softplus_evidence(y):
    return torch.nn.functional.softplus(y)

def kl_divergence(alpha, device='cpu'):
    num_classes=alpha.shape[-1]
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl

def loglikelihood_loss(y, alpha, device='cpu'):
    y = y#.to(device)
    alpha = alpha#.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)
    loglikelihood_err = torch.sum((y - (alpha / S)) ** 2, dim=1, keepdim=True)
    loglikelihood_var = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True
    )
    loglikelihood = loglikelihood_err + loglikelihood_var
    return loglikelihood

def edl_loss(func, y, alpha, epoch_num, annealing_step, device=None):
    y = y#.to(device)
    alpha = alpha#.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)

    A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, device=device)
    return A + kl_div

def mse_loss(y, alpha, epoch_num, annealing_step=10, device='cpu'):
    y = y#.to(device)
    alpha = alpha#.to(device)
    loglikelihood = loglikelihood_loss(y, alpha, device=device)

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, device=device)
    return loglikelihood + kl_div

def edl_log_loss(output, target, epoch_num, annealing_step, device=None):
    evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(
        edl_loss(
            torch.log, target, alpha, epoch_num, annealing_step, device
        )
    )
    return loss

class EvidentialMSELoss():
    """
    I made this a class so that you can give it certain default parameters (like evidence_activation_fx) and can then pass it to a fitter
    Because of the '__call__' method, you can still just call the EdlMSELoss object like you would a normal function
    """
    def __init__(self, evidence_activation_fx=torch.nn.functional.softplus, annealing_step=10, device='cpu'):
        self.evidence_activation_fx, self.annealing_step, self.device = evidence_activation_fx, annealing_step, device
        # todo take out device from loss fxs now that it is in class init
    def _one_hot_embedding(self, labels, num_classes):
        y = torch.eye(num_classes).float().to(self.device) 
        return y[labels]

    def edl_mse_loss(self, output, target, epoch_num, device='cpu', losswts=None, reduction='mean'):
        y_onehot = self._one_hot_embedding(target, num_classes=output.shape[-1])
        evidence = self.evidence_activation_fx(output)
        alpha = evidence + 1
        loss = mse_loss(y_onehot, alpha, epoch_num, self.annealing_step, device=device)
        if losswts is not None: loss*=losswts.view(-1,1)
        if reduction=='mean': loss = torch.mean(loss)
        return loss.flatten()
    
    def edl_digamma_loss(self, output, target, epoch_num, device='cpu', losswts=None, reduction='mean'):
        y_onehot = self._one_hot_embedding(target, num_classes=output.shape[-1])
        evidence = self.evidence_activation_fx(output)
        alpha = evidence + 1
        loss = edl_loss(torch.digamma, y_onehot, alpha, epoch_num, self.annealing_step, device)
        if losswts is not None: loss=loss.view(-1,output.shape[-1])*losswts.view(-1,1)
        if reduction=='mean': loss = torch.mean(loss)
        return loss.flatten()
    
    def __call__(self, *args, **kwargs):
        return self.edl_mse_loss(*args, **kwargs)
        # return self.edl_digamma_loss(*args, **kwargs)