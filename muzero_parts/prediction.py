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
    finite_action_space=False
    def __init__(self):
        super().__init__()

    def policy_only(self, state):
        return self.policy_value(state)[0]

    def value_only(self, state):
        return self.policy_value(state)[1]

    def policy_value(self, state):
        """
        ONLY VALID FOR FINITE ACTION SPACES
        :param state: state or batch of states
        :return: (policy, value), both tensors, policy is in Delta(A), value is a real number or a vector
        """
        raise NotImplementedError

    def sample_actions(self, state, n=1):
        """
        samples actions from probability distribution
        :param state:
        :param n: number of actions to sample
        :return: batch of n actions
        """
        raise NotImplementedError


class MuZeroPrediction(Prediction):
    """
    standard muzero prediction
    fixed state representation (i.e. shaped tensor)
    finite action space
    """
    finite_action_space = True
    def __init__(self, network_config):
        """
        :param network_config:
            produces a network (see nn_from_config) that returns (policy, value)
            this parameter will imply the num actions, as this will be the size of the policy vector
        """
        super().__init__()
        self.network = CustomNN(structure=network_config)
        self.num_actions = self.network.output_shape[0]

    def policy_value(self, state):
        return self.network(state)

    def sample_actions(self, state, n=1):
        self.policy_only(state=state)
        if n==1:
            return

if __name__ == "__main__":
    import torch, pyspiel, numpy as np

    g = pyspiel.load_game("tic_tac_toe")
    network_config = {
        'input_shape': tuple(g.observation_tensor_shape()),
        'layers': [
            {'type': 'flatten'},
            {
                "type": "linear",
                "out_features": 64,
            },
            {'type': 'relu'},
            {
                "type": 'split',
                'branches': [
                    [  # policy head
                        {
                            "type": "linear",
                            "out_features": g.num_distinct_actions(),
                        },
                        {
                            "type": "softmax",
                        },
                    ],
                    [  # value head
                        {
                            "type": "linear",
                            "out_features": g.num_players(),
                        },
                    ],
                ]
            },
        ]
    }
    prediction_net = MuZeroPrediction(network_config=network_config)
    s = g.new_initial_state()
    while not s.is_terminal():
        if s.is_chance_node():
            s.apply_action(np.random.choice(s.legal_actions()))
        else:
            obs = torch.tensor(s.observation_tensor()).reshape(g.observation_tensor_shape())
            obs = obs.unsqueeze(0)  # batch of 1
            policy, value = prediction_net.policy_value(obs)
            policy = torch.flatten(policy).detach().cpu().numpy()
            print('full policy', policy)
            policy = policy[s.legal_actions()]
            policy = policy/np.sum(policy)
            print('restricted policy', policy)
            action = np.random.choice(s.legal_actions(), p=policy)
            print('action choice', action)
            s.apply_action(action)

        print(s)
        print()
