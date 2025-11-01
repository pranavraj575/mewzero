"""
representation function to go from state to encoded abstract state
if we are training in non-abstract spaces, this can be either the identity or an bijection
    i.e. we are using the full state as its own representation
"""
import torch
from torch import nn


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
