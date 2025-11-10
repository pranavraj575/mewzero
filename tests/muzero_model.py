import torch, numpy as np

from muzero_parts.dynamics import Dynamics
from muzero_parts.representation import Representation
from muzero_parts.action_enc_dec import MuzeroActionEncDec
from muzero_parts.prediction import Prediction


class MuZeroModel:
    def __init__(self,
                 representation: Representation,
                 action_enc_dec: MuzeroActionEncDec,
                 dynamics: Dynamics,
                 prediction: Prediction,
                 ):
        self.representation = representation
        self.action_enc_dec = action_enc_dec
        self.dynamics = dynamics
        self.prediction = prediction

    def sample_action(self, true_state, legal_actions=None, search=False):
        abs_state = self.representation.encode(state=true_state)
        if search:
            raise NotImplementedError
        else:
            if self.prediction.finite_action_space:
                policy = self.prediction.policy_only(states=abs_state).flatten()
                if legal_actions is None:
                    action = np.random.choice(np.arange(len(policy)), p=policy.detach().cpu().numpy())
                else:
                    policy = policy[legal_actions]
                    policy = policy/torch.sum(policy)
                    action = np.random.choice(legal_actions, p=policy.detach().cpu().numpy())
            else:
                raise NotImplementedError
                actions = self.prediction.sample_actions(state=abs_state, n=1)
        return self.action_enc_dec.decode(state=true_state,
                                          action=action,
                                          )
