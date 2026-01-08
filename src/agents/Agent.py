from abc import ABC, abstractmethod
from typing import List
from matplotlib.animation import FuncAnimation
import torch
from Costs import CostFunction
from instruments.Claims import Claim
from tqdm import tqdm
from instruments.Instruments import Instrument
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F


class Agent(torch.nn.Module, ABC):
    """
    Base class for deep hedge agent
    """
    network: torch.nn.Module
    optimizer: torch.optim.Optimizer

    def __init__(self,
                 criterion: torch.nn.Module,
                 cost_function: CostFunction,
                 hedging_instruments: List[Instrument],
                 interest_rate,
                 lr=0.005,
                 pref_gpu=True):
        """
        :param model: torch.nn.Module
        :param optimizer: torch.optim
        :param criterion: torch.nn
        :param device: torch.device
        """
        super(Agent, self).__init__()
        device: torch.device = torch.device('cpu')
        if pref_gpu:
            if torch.cuda.is_available():
                device = torch.device('cuda')
                print("Running on CUDA GPU")

            elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
                device = torch.device("mps")
                print("Running on MPS GPU")

        self.device = device
        self.lr = lr
        self.criterion = criterion.to(device)
        self.cost_function = cost_function
        self.interest_rate = interest_rate
        self.hedging_instruments = hedging_instruments
        self.N = len(hedging_instruments)
        self.to(device)
        self.training_logs = dict()
        self.portfolio_logs = dict()
        self.validation_logs = dict()

    @abstractmethod
    def forward(self, state: tuple) -> torch.Tensor: # (P, N)
        pass


    def policy(self, state: tuple) -> torch.Tensor:
        """
        :param x: torch.Tensor
        :return: torch.Tensor
        """
        return self.forward(state)

    # returns the final p&l
    def compute_portfolio(self, hedge_paths, logging = True) -> torch.Tensor:
        # number of time steps
        P, T, N = hedge_paths.shape

        cash_account = torch.zeros(P, T, device=self.device)
        portfolio_value = torch.zeros(P, T, device=self.device)
        positions = torch.zeros(P, T, N, device=self.device)
        q_batch = torch.zeros(P, T, device=self.device)

        #if q != 0:
            #q_batch = q * torch.ones(P, T, device=self.device)
        #else:
            #q_batch = torch.zeros(P, T, device=self.device)
        
        state = hedge_paths[:,:1], cash_account[:,:1], positions[:,:1], T, q_batch
        action = self.policy(state)
        positions[:, 0] = action
        cost_of_action = self.cost_function(action, state)
        purchase = (action * hedge_paths[:, 0]).sum(dim=-1)
        spent = purchase + cost_of_action # (P, 1)
        # update cash account
        cash_account[:,0] = - spent # (P, 1)
        # update portfolio value
        portfolio_value[:,0] = purchase # (P, 1)

        for t in range(1, T):
            # define state
            state = hedge_paths[:,:t+1], cash_account[:,:t], positions[:,:t], T, q_batch
            # compute action
            action = self.policy(state)
            # update positions
            positions[:, t] = positions[:, t-1] + action # (P, N)
            # compute cost of action
            cost_of_action = self.cost_function(action, state) # (P, 1)
            # TODO: check if other operations are possible
            spent = (action * hedge_paths[:, t]).sum(dim=-1) + cost_of_action # (P, 1)
            # update cash account
            cash_account[:,t] = cash_account[:, t-1] * (1+self.interest_rate) - spent # (P, 1)
            # update portfolio value
            portfolio_value[:,t] = (positions[:,t] * hedge_paths[:,t]).sum(dim=-1) # (P, 1)


        if logging:
            self.portfolio_logs = {
                "portfolio_value": portfolio_value.detach().cpu(),
                "cash_account": cash_account.detach().cpu(),
                "positions": positions.detach().cpu(),
                "hedge_paths": hedge_paths.detach().cpu(),
            }


        return portfolio_value[:,-1] + cash_account[:,-1]

    def compute_portfolio_if_CRRA(self, hedge_paths, logging=True, initial_wealth: float = 1.0) -> torch.Tensor:
        if hedge_paths.dim() == 2:
            hedge_paths = hedge_paths.unsqueeze(-1)
        P, T, N = hedge_paths.shape #reminder: N is number of hedging instrument. If it's just underlying stock, N=1.
        device = self.device
        eps = 1e-12
    
        cash_account = torch.zeros(P, T, device=device) 
        portfolio_value = torch.zeros(P, T, device=device)
        positions = torch.zeros(P, T, N, device=device)
        q_batch = torch.zeros(P, T, device=self.device)
        
        # ---------- t = 0 ----------
        S0 = hedge_paths[:, 0]  # (P, N)
        cash0 = torch.full((P,), float(initial_wealth + self.q), device=device)
        cash_account[:, 0] = cash0  # put initial wealth into the history tensor

        #if q != 0:
            #q_batch = q * torch.ones(P, T, device=self.device)
        #else:
            #q_batch = torch.zeros(P, T, device=self.device)
        
        state0 = (hedge_paths[:, :1], cash_account[:, :1], positions[:, :1], T, q_batch)  # same convention
        dtheta0 = self.policy(state0)
    
        spend0 = (dtheta0 * S0).sum(dim=-1)                 # (P,)  positive = pay cash
        cost0 = self.cost_function(dtheta0, state0)
        cost0 = cost0.squeeze(-1) if cost0.dim() > 1 else cost0
    
        need0 = spend0 + cost0                               # cash needed net of sells
        scale0 = torch.ones_like(need0)
        mask0 = need0 > 0
        scale0[mask0] = torch.clamp(cash0[mask0] / (need0[mask0] + eps), max=1.0)
    
        dtheta0 = dtheta0 * scale0.unsqueeze(-1)
        spend0 = spend0 * scale0
        cost0 = cost0 * scale0  # exact if proportional / 1-homogeneous costs; otherwise recompute cost_fn here
    
        positions[:, 0] = dtheta0
        cash_account[:, 0] = cash0 - spend0 - cost0
        portfolio_value[:, 0] = cash_account[:, 0] + (positions[:, 0] * S0).sum(dim=-1)
    
        # ---------- t = 1..T-1 ----------
        for t in range(1, T):
            St = hedge_paths[:, t]  # (P, N)
    
            # accrue interest on cash
            #cash_prev = cash_account[:, t-1] * (1.0 + self.interest_rate) outdated
            cash_account[:, t] = cash_account[:, t-1] * (1.0 + self.interest_rate)
        
            state = (hedge_paths[:, :t+1], cash_account[:, :t+1], positions[:, :t], T, q_batch)  # same as compute_portfolio; outdated: cash_account was cash_account[:, :t]
            dtheta = self.policy(state)
            cash_avail = cash_account[:, t-1] * (1.0 + self.interest_rate)       # for scaling only
    
            spend = (dtheta * St).sum(dim=-1)               # (P,)
            cost = self.cost_function(dtheta, state)
            cost = cost.squeeze(-1) if cost.dim() > 1 else cost
    
            need = spend + cost                              # if >0, we must have enough cash
            scale = torch.ones_like(need) #create tensor of same shape but all 1s
            mask = need > 0 #mask is a tensor of Booleans, showing if each component is >0 or not
            scale[mask] = torch.clamp(cash_avail[mask] / (need[mask] + eps), max=1.0)
    
            dtheta = dtheta * scale.unsqueeze(-1)
            spend = spend * scale
            cost = cost * scale  # exact if proportional / 1-homogeneous costs; otherwise recompute cost_fn here
    
            positions[:, t] = positions[:, t-1] + dtheta
            cash_account[:, t] = cash_account[:, t] - spend - cost
            portfolio_value[:, t] = cash_account[:, t] + (positions[:, t] * St).sum(dim=-1)
    
        if logging:
            self.portfolio_logs = {
                "portfolio_value": portfolio_value.detach().cpu(),
                "cash_account": cash_account.detach().cpu(),
                "positions": positions.detach().cpu(),
                "hedge_paths": hedge_paths.detach().cpu(),
            }

        all_wealth_paths = portfolio_value + cash_account  # (P,T)
        terminal_wealth = all_wealth_paths[:, -1]  # (P,)
        
        return terminal_wealth, all_wealth_paths


    def generate_paths(self, P, T, contingent_claim):
        # 1. check how many primaries are invloved
        primaries: set = set([hedge.primary() for hedge in self.hedging_instruments])
        primaries.add(contingent_claim.primary())

        # 2. generate paths for all the primaries
        primary_paths = {primary: primary.simulate(P, T) for primary in primaries}

        # 3. generate paths for all derivatives based on the primary paths
        hedge_paths = [instrument.value(primary_paths[instrument.primary()]).to(self.device) for instrument in self.hedging_instruments] # N x tensor(P x T)
        # convert to P x T x N tensor
        hedge_paths = torch.stack(hedge_paths, dim=-1) # P x T x N

        return hedge_paths, primary_paths[contingent_claim.primary()]

    def pl(self, contingent_claim: Claim, P, T, logging = True, initial_wealth=1.0): #calculates pl if criterion is not CRRA/is entropy; calculates terminal wealth if criterion is CRRA
        """
        :param contingent_claim: Instrument
        :param paths: int
        :return: None
        """
        # number of time steps: T
        # number of hedging instruments: N
        # number of paths: P

        hedge_paths, claim_path = self.generate_paths(P, T, contingent_claim) # P x T x N, P x 1
        claim_payoff = contingent_claim.payoff(claim_path).to(self.device) # P x 1

        if self.criterion.__class__.__name__ == "CRRA":
            portfolio_value, wealth_path = self.compute_portfolio_if_CRRA(hedge_paths, logging=True, initial_wealth=1.0)
        else:
            portfolio_value = self.compute_portfolio(hedge_paths, logging)

        profit = portfolio_value - claim_payoff.squeeze(-1) # P
        
        if logging:
            self.portfolio_logs["claim_payoff"] = claim_payoff.detach().cpu()
            delta = contingent_claim.delta(claim_path)
            self.portfolio_logs["claim_delta"] = delta
            

        return profit, wealth_path #changed from: profit, claim_payoff

    def crra_ruin_penalized_loss(
        self,
        terminal_wealth: torch.Tensor,      # shape: (P,) or (P,1)
        wealth_path: torch.Tensor,          # shape: (P,T)
        lambda_ruin: float = 1e10,
        tau: float = 1e-2,
        p: int = 2,
        eps: float = 1e-2,
    ) -> torch.Tensor:
        """
        Loss = -E[ CRRA(W_T) ] + lambda * E[ softplus(-min_t W_t / tau)*tau ]^p
    
        Keeps CRRA definition untouched for W>0; eps is only to avoid log/NaNs at W<=0.
        """
        import torch.nn.functional as F
    
        # --- terminal CRRA term (unchanged for W>0) ---
        W_T = terminal_wealth.squeeze(-1) if terminal_wealth.ndim > 1 else terminal_wealth
        W_T_pos = torch.clamp(W_T, min=eps)
    
        util = self.criterion(W_T_pos)  # could be (P,) or scalar depending on your CRRA implementation
        if util.ndim > 0:
            util = util.mean()
        base_loss = -util
    
        # --- pathwise ruin penalty via minimum wealth ---
        W_min = wealth_path.min(dim=1).values  # (P,)
    
        # softplus(x) ~= max(0,x) but smooth. Multiplying by tau makes it scale like (-W_min)^+
        shortfall = F.softplus((-W_min) / tau) * tau  # approx max(0, -W_min)
        penalty = (shortfall ** p).mean()
    
        return base_loss + lambda_ruin * penalty

    def fit(self, contingent_claim: Claim, batch_paths: int, epochs = 50, paths = 100, verbose = False, T = 365, logging = True):
        # non-CRRA case
        """
        :param contingent_claim: Instrument
        :param epochs: int
        :param batch_size: int
        :param verbose: bool
        :return: None
        """
        losses = []
        if logging:
            self.training_logs["training_PL"] = torch.zeros(epochs, paths)

        for epoch in tqdm(range(epochs), desc="Training", total=epochs, leave=False, unit="epoch"):
            self.train()
            epoch_loss_sum = 0.0
            epoch_paths = 0
            batch_iter = [(start, min(batch_paths, paths - start)) for start in range(0, paths, batch_paths)]
            for start, current_batch_size in batch_iter:
                pl, _ = self.pl(contingent_claim, current_batch_size, T, False, self.q)
                loss = - self.criterion(pl)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss_sum += loss.item() * current_batch_size
                epoch_paths += current_batch_size
                
                if logging:
                    self.training_logs["training_PL"][epoch, start:start + current_batch_size] = pl.detach().cpu()
                    
            epoch_loss = epoch_loss_sum / max(epoch_paths, 1)
            losses.append(epoch_loss)
            if verbose:
                print(f"Epoch: {epoch}, Loss: {epoch_loss: .2f}")

        if logging:
            self.training_logs["training_losses"] = torch.Tensor(losses).cpu()

        return losses
        
    def fit_CRRA(self, contingent_claim: Claim, batch_paths: int, epochs = 50, paths = 100, verbose = False, T = 365, logging = True):
        """
        :param contingent_claim: Instrument
        :param epochs: int
        :param batch_size: int
        :param verbose: bool
        :return: None
        """
        losses = []
        if logging:
            self.training_logs["training_PL"] = torch.zeros(epochs, paths)

        for epoch in tqdm(range(epochs), desc="Training", total=epochs, leave=False, unit="epoch"):
            self.train()
            epoch_loss_sum = 0.0
            epoch_paths = 0
            batch_iter = [(start, min(batch_paths, paths - start)) for start in range(0, paths, batch_paths)]
            for start, current_batch_size in batch_iter:
                profit, wealth_path = self.pl(contingent_claim, current_batch_size, T, False, self.q) 
                loss = self.crra_ruin_penalized_loss(
                    terminal_wealth=profit,        # or terminal_wealth if you want ruin relative to gross wealth
                    wealth_path=wealth_path,       # penalize min wealth along the hedge
                    lambda_ruin=1e10,
                    tau=1e-2,
                    p=2
                )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss_sum += loss.item() * current_batch_size
                epoch_paths += current_batch_size
                
                if logging:
                    self.training_logs["training_PL"][epoch, start:start + current_batch_size] = profit.detach().cpu()
                    
            epoch_loss = epoch_loss_sum / max(epoch_paths, 1)
            losses.append(epoch_loss)
            if verbose:
                print(f"Epoch: {epoch}, Loss: {epoch_loss: .2f}")

        if logging:
            self.training_logs["training_losses"] = torch.Tensor(losses).cpu()

        return losses
        
    def q_range_penalty(q: torch.Tensor, q_min: float, q_max: float, tau: float = 1e-3, power: int = 2):
        """
        Smooth penalty = softplus((q_min - q)/tau)^power + softplus((q - q_max)/tau)^power
        Zero-ish inside [q_min, q_max], grows smoothly outside.
        """
        low  = F.softplus((q_min - q) / tau)
        high = F.softplus((q - q_max) / tau)
        return (low**power + high**power)

    def fit_CRRA_option_price(self, contingent_claim: Claim, batch_paths: int, epochs = 50, paths = 100, verbose = False, T = 365, logging = True, alpha=None, beta=None, baseline_EU_mean = None, p_norm = None, q_min = 0.0, q_max = 1.0):
        losses = []
        q_history = []
        
        for epoch in range(epochs):
            self.train()
            epoch_loss_sum = 0.0
            epoch_paths = 0
            batch_iter = [(start, min(batch_paths, paths - start)) for start in range(0, paths, batch_paths)]
            for start, current_batch_size in batch_iter:
                profit, wealth_path = self.pl(contingent_claim, current_batch_size, T, False, self.q)
                EU_with_liability = self.criterion(profit)
                loss_before_other_penalties = self.crra_ruin_penalized_loss(
                    terminal_wealth=profit, wealth_path=wealth_path,lambda_ruin=1e10, tau=1e-2, p=2)
                low  = F.softplus((q_min - self.q) / 1e-1)
                high = F.softplus((self.q - q_max) / 1e-1)
                penalty_q = low**2 + high**2
                penalty_match = torch.abs(EU_with_liability - baseline_EU_mean) ** p_norm
                loss = alpha * penalty_q + beta * penalty_match
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss_sum += loss.item() * current_batch_size
                epoch_paths += current_batch_size
                
            epoch_loss = epoch_loss_sum / max(epoch_paths, 1)
            losses.append(epoch_loss)
            q_history.append(self.q.detach().item())

        return losses, q_history


    def validate(self, contingent_claim: Claim, paths = int(1e6), T = 365, logging = True):
        """
        :param contingent_claim: Instrument
        :param epochs: int
        :param batch_size: int
        :return: None
        """
        with torch.no_grad():
            self.eval()
            profit, claim_payoff = self.pl(contingent_claim, paths, T, True)
            loss = -self.criterion(profit)
            if logging:
                self.validation_logs["validation_profit"] = profit.detach().cpu()
                self.validation_logs["validation_claim_payoff"] = claim_payoff.detach().cpu()
                self.validation_logs["validation_loss"] = loss.detach().cpu()

            return loss.item()
