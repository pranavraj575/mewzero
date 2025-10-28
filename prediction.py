"""
muzero policy/value prediction functions
    s -> Delta(A), v
    where v is either a real number (single/two player) or vector

value function is easy, and can usually be a learned FFN
when there are a finite number of total actions possible, policy function can also be a FFN or similar
If the action space is continuous or discrete and infinite, this is a bit annoying
    continuous:
        must produce a continuous distribution pi(A | S)
        Can learn a static distribution from training data (i.e. through a VAE) and not update this
        otherwise must use tricks to update pi given improvements on sampled actions
            i.e. sampled a1,...,ak, found through MCTS search that a1 performed better, so we must push distribution towards a1
                arXiv:2104.06303
    discrete and countably infinite:
         annoying in a different way, as we cannot take advantage of euclidean space topology
         if finite size at every state (i.e. jenga, 'increasing size board game'), we may produce a policy via a map
            (s,a) -> R, which is then passed throuch softmax to produce a distribution over valid actions
"""
from torch import nn
from networks.nn_from_config import CustomNN


class Prediction(nn.Module):
    def __init__(self):
        super().__init__()

    def value_only(self, states):
        raise NotImplementedError

    def policy_only(self, states):
        raise NotImplementedError

    def policy_value(self, states):
        """
        ONLY VALID FOR DISCRETE FINITE ACTION SPACES
        :param states: batch of states
        :return: (policy, value), both tensors, policy is in Delta(A), value is a real number or a vector
        """
        raise NotImplementedError

    def sample_actions(self, state, n=1):
        """
        ONLY USEFUL IN INFINITE ACTION SPACES
        samples actions from probability distribution
        :param state:
        :return: batch of n actions
        """
        raise NotImplementedError


class MuZeroPrediction(Prediction):
    """
    standard muzero prediction
    fixed state representation (i.e. shaped tensor)
    finite action space
    """

    def __init__(self, network_config, num_actions=None, device=None):
        """
        :param network_config:
            produces a network (see nn_from_config) that returns (policy, value)
            this parameter will imply the num actions, as this will be the size of the policy vector
        :param num_actions:
        """
        super().__init__()
        self.network = CustomNN(structure=network_config, device=device)
