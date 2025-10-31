import torch, numpy as np

from muzero_parts.dynamics import Dynamics
from muzero_parts.representation import MuzeroRepresentation
from muzero_parts.action_enc_dec import MuzeroActionEncDec
from muzero_parts.prediction import Prediction
from muzero_parts.MCTS import AbsMCTS


def train(initial_state,representation:MuzeroRepresentation,dynamics:Dynamics,mcts:AbsMCTS,player=0):
    """
    until a certian depth
    :param initial_state:
    :param representation:
    :param dynamics:
    :return:
    """
    s0=representation.encode(initial_state)

    mcts.get_mcts_policy_value(state=s0,
                               num_sims=1,
                               dynamics=dynamics,
                               player=player,
                               temp=1,
                               root=None,
                               depth=float('inf'),
                               )
    pass
