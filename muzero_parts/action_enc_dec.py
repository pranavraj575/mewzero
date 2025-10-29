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

    def encode(self, states, actions):
        """
        encodes batch of real actions
        :param states: batch of real states
        :param actions: batch of real actions
        :return: batch of encoded abstract actions
        """
        raise NotImplementedError

    def decode(self, states, actions):
        """
        decodes batch of abstract actions
        :param states: batch of real states
        :param actions: batch of abstract actions
        :return: batch of real actions
        """
        raise NotImplementedError


class IdentityActionEncDec(MuzeroActionEncDec):
    """
    for when no encoding is necessary, we are using true actions in the abstract game
    """

    def encode(self, states, actions):
        return actions

    def decode(self, states, actions):
        return actions
