

from abc import ABC, abstractmethod

import torch


class RiskMeasure(ABC, torch.nn.Module):
    pass



class WorstCase(RiskMeasure):

    def forward(self, portfolio_value: torch.Tensor):
        # portifolio_value: P x 1 (final portfolio value for every path)
        # return: 1 x 1
        return portfolio_value.min()


class TailValue(RiskMeasure):

    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = alpha

    def forward(self, portfolio_value: torch.Tensor):
        # portifolio_value: P x 1 (final portfolio value for every path)
        # returns the mean of the worst alpha% of the paths
        # remember that portfolio_value is not sorted and can be negative
        # return: 1 x 1
        k = int(self.alpha * portfolio_value.shape[0])
        return portfolio_value.topk(k, largest=False).values.mean()


class Median(RiskMeasure):

    def forward(self, portfolio_value: torch.Tensor):
        # portifolio_value: P x 1 (final portfolio value for every path)
        # return: 1 x 1
        return portfolio_value.median()




class Expectation(RiskMeasure):

    def forward(self, portfolio_value: torch.Tensor):
        # portifolio_value: P x 1 (final portfolio value for every path)
        # return: 1 x 1
        return portfolio_value.mean()


class Entropy(RiskMeasure):
    def __init__(self, lambd: float):
        super().__init__()
        self.lambd = lambd

    def forward(self, portfolio_value: torch.Tensor):
        # portifolio_value: P x 1 (final portfolio value for every path)
        # return: 1 x 1
        return - 1/self.lambd * torch.log(torch.exp(-self.lambd*portfolio_value).mean())

class CVaR(RiskMeasure):
    def __init__(self, p: float):
        super().__init__()
        self.lambd = 1/p - 1

    def forward(self, portfolio_value: torch.Tensor):
        # portifolio_value: P x 1 (final portfolio value for every path)
        # return: 1 x 1
        # return expected shortfall

        def target_function(omega: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            return omega + self.lambd * torch.clamp(-omega - x, min=0).mean()

        # try different omega's and return the one that gives the max of target_function (expected shortfall)
        omega = torch.linspace(-1, 1, portfolio_value.shape[0], device=portfolio_value.device)
        target = target_function(omega, portfolio_value)
        omega = omega[target.argmax()]

        return -(omega + self.lambd * torch.clamp(-omega - portfolio_value, min=0).mean())


class ExpectationVariance(RiskMeasure):

    def __init__(self, alpha: float) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, portfolio_value: torch.Tensor):

        return portfolio_value.mean() - self.alpha * portfolio_value.var()



class Variance(RiskMeasure):

    def forward(self, portfolio_value: torch.Tensor):
        # portifolio_value: P x 1 (final portfolio value for every path)
        # return: 1 x 1
        return - portfolio_value.var()
        
class CRRA(nn.Module):
    """
    Criterion to be plugged into Agent.fit():
        fit() computes loss = -criterion(pl)
    where pl is terminal P&L from Agent.pl().

    This wrapper maps:
        pl  ->  W_T = w0 + pl  ->  CRRA certainty equivalent (or expected utility)

    IMPORTANT:
    - Requires W_T > 0; we clamp by eps.
    - Returns a scalar to maximize (so fit() can minimize negative).
    """

    def __init__(self, gamma: float, w0: float = 1.0, eps: float = 1e-12, use_ce: bool = True):
        super().__init__()
        if gamma <= 0:
            raise ValueError("gamma must be > 0.")
        self.gamma = float(gamma)
        self.w0 = float(w0)
        self.eps = float(eps)
        self.use_ce = bool(use_ce)

    def forward(self, pl: torch.Tensor) -> torch.Tensor:
        # pl is shape (P,) or (P,1) depending on upstream; normalize to (P,)
        w = pl.squeeze(-1) # the "pl" function is designed to account for the initial wealth if the criterion is CRRA
 
        # log utility case
        if abs(self.gamma - 1.0) < 1e-12:
            #u = torch.log(w)...not used to avoid NaN if at least one component in w is not positive
            u = torch.where(w > 0, torch.log(w), torch.full_like(w, -100000.0)) #huge penalty if wealth is negative
            if self.use_ce:
                return torch.exp(u.mean())  # CE = exp(E[log W])
            else:
                return u.mean() 
        else:
            one_minus_g = 1.0 - self.gamma
            w_pow = torch.where(w>0, w.pow(one_minus_g), torch.full_like(w, -100000.0))
            if self.use_ce:
                # CE = (E[W^(1-gamma)])^(1/(1-gamma))
                m = torch.clamp(w_pow.mean(), min=self.eps)
                return m.pow(1.0 / one_minus_g)
            else:
                # Expected utility = E[ W^(1-gamma)/(1-gamma) ]
                return (w_pow / one_minus_g).mean() 
