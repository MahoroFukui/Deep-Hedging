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

    def compute_portfolio_if_CRRA_no_borrowing_constraint(self, hedge_paths, logging=True, initial_wealth: float = 1.0) -> torch.Tensor:
        if hedge_paths.dim() == 2:
            hedge_paths = hedge_paths.unsqueeze(-1)
        P, T, N = hedge_paths.shape #reminder: N is number of hedging instrument. If it's just underlying stock, N=1.
        device = self.device
        eps = 1e-12
        initial_wealth = self.initial_wealth
    
        cash_account = torch.zeros(P, T, device=device) 
        portfolio_value = torch.zeros(P, T, device=device)
        positions = torch.zeros(P, T, N, device=device)
        q_batch = torch.zeros(P, T, device=self.device)
        
        # ---------- t = 0 ----------
        S0 = hedge_paths[:, 0]  # (P, N)
        
        cash0 = torch.full((P,), initial_wealth / 2, device=device) + self.q
        cash_account[:, 0] = cash0  # put initial wealth into the history tensor
        positions[:, 0] = initial_wealth / (2 * S0)

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

    def compute_portfolio_if_CRRA(self, hedge_paths, logging: bool = True, initial_wealth: float = 1.0):
        """
        Portfolio simulation with borrowing allowed but capped at (pre-trade) net worth at each time k:
            cash_after >= -max(0, net_worth_before)
    
        Notes:
        - Assumes policy outputs trade increments dtheta in shares.
        - Assumes cost_function(dtheta, state) returns per-path cost (P,) or (P,1).
        """
    
        import torch
    
        if hedge_paths.dim() == 2:
            hedge_paths = hedge_paths.unsqueeze(-1)
    
        P, T, N = hedge_paths.shape
        device = self.device
        eps = 1e-12
    
        # Use agent's stored initial wealth unless you truly want the argument
        initial_wealth = self.initial_wealth
    
        cash_account = torch.zeros(P, T, device=device)
        portfolio_value = torch.zeros(P, T, device=device)
        positions = torch.zeros(P, T, N, device=device)
    
        # Keep q_batch shape consistent with your agents; not used inside SimpleAgent, used in RecurrentAgent
        q_batch = torch.zeros(P, T, device=device)
    
        # ---------- t = 0 ----------
        S0 = hedge_paths[:, 0]  # (P, N)
    
        cash0 = torch.full((P,), initial_wealth / 2, device=device) + self.q
        cash_account[:, 0] = cash0
    
        # Your baseline initial holdings (50/50 cash vs stock). Keep it unless you want otherwise.
        positions[:, 0] = initial_wealth / (2 * S0)
    
        # Policy action: trade increment at t=0
        state0 = (hedge_paths[:, :1], cash_account[:, :1], positions[:, :1], T, q_batch)
        dtheta0 = self.policy(state0)  # (P, N)
    
        spend0 = (dtheta0 * S0).sum(dim=-1)  # (P,)  positive = pay cash (buy)
        cost0 = self.cost_function(dtheta0, state0)
        cost0 = cost0.squeeze(-1) if cost0.dim() > 1 else cost0  # (P,)
    
        need0 = spend0 + cost0  # net cash outflow required
    
        # Borrowing cap at pre-trade net worth: cash_after >= -max(0, NW_before)
        stock_value0 = (positions[:, 0] * S0).sum(dim=-1)  # (P,)
        nw0 = cash0 + stock_value0                         # (P,)
        borrow_cap0 = torch.clamp(nw0, min=0.0)            # cannot borrow if already insolvent
        min_cash0 = -borrow_cap0                           # cash floor
    
        # Ensure cash_after = cash0 - need0 >= min_cash0  => need0 <= cash0 - min_cash0
        limit0 = cash0 - min_cash0
    
        scale0 = torch.ones_like(need0)
        mask0 = need0 > limit0
        scale0[mask0] = torch.clamp(limit0[mask0] / (need0[mask0] + eps), max=1.0)
    
        dtheta0 = dtheta0 * scale0.unsqueeze(-1)
        spend0 = spend0 * scale0
        cost0 = cost0 * scale0  # exact for proportional/1-homogeneous costs; otherwise recompute after scaling
    
        # Update holdings and cash
        positions[:, 0] = positions[:, 0] + dtheta0
        cash_account[:, 0] = cash0 - spend0 - cost0
        portfolio_value[:, 0] = cash_account[:, 0] + (positions[:, 0] * S0).sum(dim=-1)
    
        # ---------- t = 1..T-1 ----------
        for t in range(1, T):
            St = hedge_paths[:, t]  # (P, N)
    
            # accrue interest on cash
            cash_account[:, t] = cash_account[:, t - 1] * (1.0 + self.interest_rate)
            cash_before = cash_account[:, t]  # cash available before trade at time t
    
            # policy action (trade increment)
            state = (hedge_paths[:, :t + 1], cash_account[:, :t + 1], positions[:, :t], T, q_batch)
            dtheta = self.policy(state)  # (P, N)
    
            spend = (dtheta * St).sum(dim=-1)  # (P,)
            cost = self.cost_function(dtheta, state)
            cost = cost.squeeze(-1) if cost.dim() > 1 else cost  # (P,)
    
            need = spend + cost
    
            # Borrowing cap based on pre-trade net worth at time t (using positions at t-1 valued at St)
            stock_value_before = (positions[:, t - 1] * St).sum(dim=-1)  # (P,)
            nw_before = cash_before + stock_value_before                 # (P,)
            borrow_cap = torch.clamp(nw_before, min=0.0)
            min_cash = -borrow_cap
    
            # Enforce cash_after >= min_cash => need <= cash_before - min_cash
            limit = cash_before - min_cash
    
            scale = torch.ones_like(need)
            mask = need > limit
            scale[mask] = torch.clamp(limit[mask] / (need[mask] + eps), max=1.0)
    
            dtheta = dtheta * scale.unsqueeze(-1)
            spend = spend * scale
            cost = cost * scale  # exact for proportional costs; otherwise recompute after scaling
    
            positions[:, t] = positions[:, t - 1] + dtheta
            cash_account[:, t] = cash_before - spend - cost
            portfolio_value[:, t] = cash_account[:, t] + (positions[:, t] * St).sum(dim=-1)
    
        if logging:
            self.portfolio_logs = {
                "portfolio_value": portfolio_value.detach().cpu(),
                "cash_account": cash_account.detach().cpu(),
                "positions": positions.detach().cpu(),
                "hedge_paths": hedge_paths.detach().cpu(),
            }
    
        # Total wealth is already portfolio_value (cash + stock value); do NOT add cash again.
        all_wealth_paths = portfolio_value  # (P, T)
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

    def pl(self, contingent_claim: Claim, P, T, logging = True, initial_wealth=1.0,): #calculates pl if criterion is not CRRA/is entropy; calculates terminal wealth if criterion is CRRA
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
        lambda_ruin: float = 500,
        tau: float = 1e-2,
        p: int = 1,
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
                print(f"Epoch: {epoch}, Loss: {epoch_loss: .5f}")

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
        
        best_loss = float("inf")
        bad_epochs = 0
        best_state = None
        
        if logging:
            self.training_logs["training_PL"] = torch.zeros(epochs, paths)

        import copy
        from tqdm import tqdm

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
                    lambda_ruin=1,
                    tau=1e-2,
                    p=1
                )

                self.opt_policy.zero_grad()
                loss.backward()
                self.opt_policy.step()
                epoch_loss_sum += loss.item() * current_batch_size
                epoch_paths += current_batch_size
                
                if logging:
                    self.training_logs["training_PL"][epoch, start:start + current_batch_size] = profit.detach().cpu()
                    
            epoch_loss = epoch_loss_sum / max(epoch_paths, 1)
            losses.append(epoch_loss)
            if verbose:
                print(f"Epoch: {epoch}, Loss: {epoch_loss: .5f}")

            # ---- EARLY STOPPING LOGIC ----
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                bad_epochs = 0
    
                # deep copy model + optimizer-independent state
                best_state = {
                    "network": copy.deepcopy(self.network.state_dict()),
                    "q": self.q.detach().clone()
                }
    
            else:
                bad_epochs += 1
                if verbose:
                    print(f"  no improvement ({bad_epochs}/{3})")
    
                if bad_epochs >= 3:
                    if verbose:
                        print(
                            f"Early stopping at epoch {epoch}. "
                            f"Best loss: {best_loss:.6f}"
                        )
                    break
    
        # ---- RESTORE BEST MODEL ----
        if best_state is not None:
            self.network.load_state_dict(best_state["network"])
            with torch.no_grad():
                self.q.copy_(best_state["q"])

        if logging:
            self.training_logs["training_losses"] = torch.Tensor(losses).cpu()

        return losses
        
    def fit_CRRA_option_price_no_earlystopping(self, contingent_claim: Claim, batch_paths: int, epochs = 50, paths = 100, verbose = True, T = 365, logging = True, alpha=None, beta1=None, beta2=None, baseline_EU_mean = None, p_norm = None, q_min = 0.0, q_max = 1.0):
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
                    terminal_wealth=profit, wealth_path=wealth_path,lambda_ruin=0.01, tau=1e-2, p=1)
                low  = F.softplus((q_min - self.q) / 1e-1)
                high = F.softplus((self.q - q_max) / 1e-1)
                penalty_q = low**2 + high**2
                penalty_match = torch.abs(EU_with_liability - baseline_EU_mean) ** p_norm
                loss = alpha * loss_before_other_penalties + beta1 * penalty_q + beta2 * penalty_match
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss_sum += loss.item() * current_batch_size
                epoch_paths += current_batch_size
                
            epoch_loss = epoch_loss_sum / max(epoch_paths, 1)
            losses.append(epoch_loss)
            q_history.append(self.q.detach().item())
            q_val = self.q.detach().item() if torch.is_tensor(self.q) else float(self.q)
            if verbose:
                print(f"Epoch: {epoch}, Loss: {epoch_loss: .5f}, EU diff: {EU_with_liability - baseline_EU_mean: .5f}, option price: {q_val: .5f}")

        return losses, q_history
        
    def fit_CRRA_option_price_earlystopping_outdated(
    self,
    contingent_claim: Claim,
    batch_paths: int,
    epochs: int = 50,
    paths: int = 100,
    verbose: bool = True,
    T: int = 365,
    logging: bool = True,
    alpha=None,
    beta1=None,
    beta2=None,
    baseline_EU_mean=None,
    p_norm=None,
    q_min: float = 0.0,
    q_max: float = 1.0,
    patience: int = 3):
        """
        Early stopping if epoch loss fails to improve for `patience` consecutive epochs.
        Saves/restores the best model (policy network + q).
    
        Returns:
            losses (list[float]): epoch losses up to stopping point
            q_history (list[float]): q values per epoch up to stopping point
        """
        import copy
        import torch
        import torch.nn.functional as F
    
        losses = []
        q_history = []
    
        # ---- early stopping bookkeeping ----
        best_loss = float("inf")
        bad_epochs = 0
        best_state = None
    
        for epoch in range(epochs):
            self.train()
            epoch_loss_sum = 0.0
            epoch_paths = 0
    
            # For logging print (use last batch's EU; optionally change to epoch-average)
            EU_with_liability_last = None
    
            batch_iter = [(start, min(batch_paths, paths - start)) for start in range(0, paths, batch_paths)]
            for start, current_batch_size in batch_iter:
                profit, wealth_path = self.pl(contingent_claim, current_batch_size, T, False, self.q)
    
                EU_with_liability = self.criterion(profit)
                EU_with_liability_last = EU_with_liability.detach()
    
                loss_before_other_penalties = self.crra_ruin_penalized_loss(
                    terminal_wealth=profit,
                    wealth_path=wealth_path,
                    lambda_ruin=0.01,
                    tau=1e-2,
                    p=1
                )
    
                low = F.softplus((q_min - self.q) / 1e-1)
                high = F.softplus((self.q - q_max) / 1e-1)
                penalty_q = low**2 + high**2
    
                penalty_match = torch.abs(EU_with_liability - baseline_EU_mean) ** p_norm
    
                loss = alpha * loss_before_other_penalties + beta1 * penalty_q + beta2 * penalty_match
                    
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
    
                epoch_loss_sum += loss.item() * current_batch_size
                epoch_paths += current_batch_size
    
            epoch_loss = epoch_loss_sum / max(epoch_paths, 1)
            losses.append(epoch_loss)
    
            q_val = self.q.detach().item() if torch.is_tensor(self.q) else float(self.q)
            q_history.append(q_val)
    
            if verbose:
                eu_diff = float("nan")
                if EU_with_liability_last is not None:
                    # EU_with_liability_last may be tensor scalar; make it float
                    eu_diff = (EU_with_liability_last - baseline_EU_mean).item() if torch.is_tensor(EU_with_liability_last) else float(EU_with_liability_last - baseline_EU_mean)
                print(f"Epoch: {epoch}, Loss: {epoch_loss: .5f}, EU diff: {eu_diff: .5f}, option price: {q_val: .5f}")
    
            # ---- EARLY STOPPING LOGIC ----
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                bad_epochs = 0
    
                best_state = {
                    "network": copy.deepcopy(self.network.state_dict()),
                    "q": self.q.detach().clone()
                }
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}. Best loss: {best_loss:.6f}")
                    break
    
        # ---- RESTORE BEST MODEL ----
        if best_state is not None:
            self.network.load_state_dict(best_state["network"])
            with torch.no_grad():
                self.q.copy_(best_state["q"])
    
        return losses, q_history


    def fit_CRRA_option_price(self,
    contingent_claim: Claim,
    batch_paths: int,
    epochs: int = 50,
    paths: int = 100,
    verbose: bool = True,
    T: int = 365,
    logging: bool = True,
    alpha=None,
    beta1=None,
    beta2=None,
    baseline_EU_mean=None,
    p_norm=None,
    q_min: float = 0.0,
    q_max: float = 1.0,
    patience: int = 5):
        losses, q_history = [], []
    
        best_total_loss = float("inf")
        best_policy_loss = float("inf")
        best_q_loss = float("inf")
        bad_epochs = 0
        best_state = None
    
        for epoch in range(epochs):
            self.train()
            epoch_loss_sum = 0.0
            epoch_policy_loss_sum = 0.0
            epoch_q_loss_sum = 0.0
            epoch_paths = 0
            EU_with_liability_last = None
    
            batch_iter = [(start, min(batch_paths, paths - start)) for start in range(0, paths, batch_paths)]
            for start, current_batch_size in batch_iter:
    
                # =========================
                # (A) POLICY UPDATE ONLY
                # =========================
                profit_p, wealth_path_p = self.pl(
                    contingent_claim, current_batch_size, T, False,
                    self.q.detach()          # <-- blocks grads to q
                )
    
                policy_loss = alpha * self.crra_ruin_penalized_loss(
                    terminal_wealth=profit_p,
                    wealth_path=wealth_path_p,
                    lambda_ruin=0.01,
                    tau=1e-2,
                    p=1
                )
    
                self.opt_policy.zero_grad(set_to_none=True)
                policy_loss.backward()
                self.opt_policy.step()
    
                # =========================
                # (B) q UPDATE ONLY
                # =========================
                # Freeze policy network so q-loss does NOT update policy_params
                for p in self.network.parameters():
                    p.requires_grad_(False)
    
                profit_q, wealth_path_q = self.pl(
                    contingent_claim, current_batch_size, T, False,
                    self.q                  # <-- allow grads to q
                )
    
                EU_with_liability = self.criterion(profit_q)
                EU_with_liability_last = EU_with_liability.detach()
    
                low  = F.softplus((q_min - self.q) / 1e-1)
                high = F.softplus((self.q - q_max) / 1e-1)
                penalty_q = low**2 + high**2
    
                penalty_match = torch.abs(EU_with_liability - baseline_EU_mean) ** p_norm
    
                q_loss = beta1 * penalty_q + beta2 * penalty_match
    
                self.opt_q.zero_grad(set_to_none=True)
                q_loss.backward()
                self.opt_q.step()
    
                # Unfreeze policy network
                for p in self.network.parameters():
                    p.requires_grad_(True)
    
                # bookkeeping (for early stopping you can decide what "epoch_loss" means)
                batch_loss_for_logging = (policy_loss.detach() + q_loss.detach()).item()
                epoch_loss_sum += batch_loss_for_logging * current_batch_size
                detached_policy_loss = policy_loss.detach().item()
                epoch_policy_loss_sum += detached_policy_loss * current_batch_size
                detached_q_loss = q_loss.detach().item()
                epoch_q_loss_sum += detached_q_loss * current_batch_size
                
                epoch_paths += current_batch_size
    
            epoch_total_loss = epoch_loss_sum / max(epoch_paths, 1)
            epoch_policy_loss = epoch_policy_loss_sum / max(epoch_paths, 1)
            epoch_q_loss = epoch_q_loss_sum / max(epoch_paths, 1)

            #----------
            losses.append(epoch_total_loss)
            q_val = self.q.detach().item()
            q_history.append(q_val)
            #----------^not really relevant though
    
            if verbose:
                eu_diff = (EU_with_liability_last - baseline_EU_mean).item() if EU_with_liability_last is not None else float("nan")
                print(f"Epoch: {epoch}, Total Loss: {epoch_total_loss: .5f}, Policy Loss: {epoch_policy_loss: .5f},  Q (option pr) Loss: {epoch_q_loss: .5f}")
                print(f"Epoch: {epoch}, EU diff: {eu_diff: .5f}, option price: {q_val: .5f}")
    
            # Early stopping uses epoch_loss; you may prefer to stop on policy_loss only
            if epoch_total_loss < best_total_loss:
                best_total_loss = epoch_total_loss
                bad_epochs = 0
                best_state = {
                    "network": copy.deepcopy(self.network.state_dict()),
                    "q": self.q.detach().clone()
                }
            elif epoch_policy_loss < best_policy_loss:
                best_policy_loss = epoch_policy_loss
                bad_epochs = 0
                best_state = {
                    "network": copy.deepcopy(self.network.state_dict()),
                }
            elif epoch_q_loss < best_q_loss:
                best_q_loss = epoch_q_loss
                bad_epochs = 0
                best_state = {
                    "q": self.q.detach().clone()
                }
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}. Best Total Loss: {epoch_total_loss: .5f}")
                    break
    
        if best_state is not None:
            self.network.load_state_dict(best_state["network"])
            with torch.no_grad():
                self.q.copy_(best_state["q"])
    
        return losses, q_history




    def validate(self, contingent_claim: Claim, paths = int(1e4), T = 365, logging = True):
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
