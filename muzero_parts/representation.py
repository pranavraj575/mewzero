"""
representation function to go from state to encoded abstract state
if we are training in non-abstract spaces, this can be either the identity or an bijection
    i.e. we are using the full state as its own representation
"""
import torch, os
from torch import nn
from networks.nn_from_config import CustomNN


class Representation(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, state):
        """
        encodes a batch of states
        :param state: state or batch of states
        :return: encoded state or encoded batch of states
        """
        return state

    def save(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self.state_dict(), os.path.join(save_dir, 'net.pkl'))

    def load(self, save_dir):
        self.load_state_dict(torch.load(os.path.join(save_dir, 'net.pkl'), weights_only=True))


class PyspielObservationRepreseentation(Representation):
    """
    for alphazero algorithm (no abstraction) applied to pyspiel games
        uses the observation tensors as the encoded state
    this is a test for very simple games (tic tac toe) where the observation tensor at a state
        encodes all previous observations as well.
    """

    def __init__(self, game=None):
        super().__init__()
        if game is None:
            self.obs_shape = None
        else:
            self.obs_shape = game.observation_tensor_shape()

    def encode(self, state):
        """
        states will be a list of pyspiel state objects
        """
        if type(state) == list:
            return torch.stack([self.encode_single(s) for s in state], dim=0)
        else:
            return self.encode_single(state).unsqueeze(dim=0)

    def encode_single(self, state):
        if self.obs_shape is None:
            return torch.tensor(state.observation_tensor()).reshape(state.get_game().observation_tensor_shape())
        else:
            return torch.tensor(state.observation_tensor()).reshape(self.obs_shape)


class LearnedRepreseentation(Representation):
    """
    learns a representation of a tensor
        assumes the input is always a batched tensor
    """

    def __init__(self, network_config):
        super().__init__()
        self.network = CustomNN(structure=network_config)

    def encode(self, state):
        return self.network(state)


class ChainRepresentation(Representation):
    def __init__(self, representation_list):
        super().__init__()
        self.reps = nn.ModuleList(representation_list)

    def encode(self, state):
        for rep in self.reps:
            state = rep.encode(state)
        return state


if __name__ == '__main__':
    import ast, pyspiel

    g = pyspiel.load_game('tic_tac_toe')
    f = open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'networks', 'net_configs', 'ttt_rep.txt'), 'r')
    network_config = ast.literal_eval(f.read())
    f.close()

    rep = ChainRepresentation([PyspielObservationRepreseentation(game=g), LearnedRepreseentation(network_config=network_config)])
    s = g.new_initial_state()
    print(rep.encode(s))
    print(rep)
    print(list(rep.parameters()))
