"""
action encoder and decoder
in standard muzero, this is not a thing (i.e. it is the identitty)
enc(state, action) produces an abstract action vector given a true state and action
dec(state, abs_action) generates a true action given a true state and abstract action
    we may cheat a little here, and condition on the TRUE state instead of abstract state.
    The motivation is that whenever we decode an action for use in the true game, we will have access to the true state.
ideally, at a given state, we should have action = dec(state, enc(state,action))
"""
from torch import nn


class MuzeroActionEncDec(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, state, action):
        """
        encodes batch of real actions
        :param state: state or batch of real states
        :param action: action or batch of real actions
        :return: abstract action or batch of encoded abstract actions
        """
        raise NotImplementedError

    def decode(self, state, action):
        """
        decodes batch of abstract actions
        :param state: state or batch of real states
        :param action: abstract action or batch of abstract actions
        :return: action or batch of real actions
        """
        raise NotImplementedError


class IdentityActionEncDec(MuzeroActionEncDec):
    """
    for when no encoding is necessary, we are using true actions in the abstract game
    """

    def encode(self, state, action):
        return action

    def decode(self, state, action):
        return action
