from collections import OrderedDict
from typing import List
import torch
from Costs import CostFunction
from agents.Agent import Agent
from instruments.Instruments import Instrument


class SimpleAgent(Agent):

    def __init__(self,
                 criterion: torch.nn.Module,
                 cost_function: CostFunction,
                 hedging_instruments: List[Instrument],
                 interest_rate,
                 pref_gpu=True,
                 q=0.5,
                 h_dim=15,
                 liability: bool = True):

        self.N = len(hedging_instruments)
        network_input_dim = self.input_dim()
        h_dim=self.h_dim

        super().__init__(criterion, cost_function, hedging_instruments, interest_rate, pref_gpu, liability)
        self.q = torch.nn.Parameter(torch.tensor(q, dtype=torch.float32))
        self.network = torch.nn.Sequential(
        OrderedDict([
            ('fc1', torch.nn.Linear(network_input_dim, h_dim)),
            ('relu1', torch.nn.ReLU()),
            ('fc2', torch.nn.Linear(h_dim, h_dim)),
            ('relu2', torch.nn.ReLU()),
            ('fc3', torch.nn.Linear(h_dim, self.N))
        ])
        ).to(self.device)
        policy_params = list(self.network.parameters())
        q_params = [self.q]     

        lr_policy=1e-4
        lr_q=5e-3
        
        if optimizer.lower() == "sgd":
            self.optimizer = torch.optim.SGD([
                {"params": policy_params, "lr": lr_policy},
                {"params": q_params,      "lr": lr_q},], betas=(0.9, 0.999))
        else:
            self.optimizer = torch.optim.Adam([
                {"params": policy_params, "lr": lr_policy},
                {"params": q_params,      "lr": lr_q},], betas=(0.9, 0.999))
            
        #if optimizer.lower() == "sgd":
            #self.optimizer = torch.optim.SGD(list(self.network.parameters()) + [self.q], lr=lr)

        #else:
            #self.optimizer = torch.optim.Adam(list(self.network.parameters()) + [self.q], lr=lr)

    def input_dim(self) -> int:
        return self.N + 1

    def feature_transform(self, state: tuple) -> torch.Tensor:
        """
        :param state: tuple
        :return: torch.Tensor
        """
        paths, cash_account, positions, T, q_batch = state

        P, t, N = paths.shape

        last_prices = state[0][:, -1, :] # (P, N)
        # log prices
        log_prices = torch.log(last_prices) # (P, N)

        times = torch.ones(P, 1, device=self.device) * (T-t) # (P, 1)
        # features is log_prices and t

        features = torch.cat([log_prices, times], dim=1) # (P, N+1)

        return features.to(self.device)

    def forward(self, state: tuple) -> torch.Tensor:
        features = self.feature_transform(state) # D x input_dim
        return self.network(features)
