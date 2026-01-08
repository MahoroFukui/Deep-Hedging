

import torch
from agents.SimpleAgent import SimpleAgent


class RecurrentAgent(SimpleAgent):

    def input_dim(self) -> int:
        return super().input_dim() + 2 * self.N + 1

    def feature_transform(self, state: tuple) -> torch.Tensor:
        """
        :param state: tuple
        :return: torch.Tensor
        """

        simple_features = super().feature_transform(state) # (P, N+1)

        if
        current_cash_account = state[1][:, -1]
        q_batch = state[4][:, -1]
        
        current_positions = state[2][:, -1] # (P, N)
        

        features = torch.cat([simple_features, current_cash_account, current_positions, current_positions], dim=1) # (P, 3N+2)
        
        #outdated: features = torch.cat([simple_features, current_positions], dim=1)
        
        
        return features.to(self.device)
