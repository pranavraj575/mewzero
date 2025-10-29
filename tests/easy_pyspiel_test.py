import pyspiel, torch, numpy as np

from muzero_model import MuZeroModel
from muzero_parts.dynamics import PyspielDynamics
from muzero_parts.representation import PyspielObservationRepreseentation
from muzero_parts.action_enc_dec import IdentityActionEncDec
from muzero_parts.prediction import MuZeroPrediction

game = pyspiel.load_game("tic_tac_toe")

network_config = {
    'input_shape': tuple(game.observation_tensor_shape()),
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
                        "out_features": game.num_distinct_actions(),
                    },
                    {
                        "type": "softmax",
                    },
                ],
                [  # value head
                    {
                        "type": "linear",
                        "out_features": game.num_players(),
                    },
                ],
            ]
        },
    ]
}

prediction = MuZeroPrediction(network_config=network_config)
dynamics = PyspielDynamics()
representation = PyspielObservationRepreseentation(game=game)
action_enc_dec = IdentityActionEncDec()
mzm = MuZeroModel(
    representation=representation,
    action_enc_dec=action_enc_dec,
    dynamics=dynamics,
    prediction=prediction,
)

s = game.new_initial_state()
terminal = False
sum_returns = np.zeros(game.num_players())
while not terminal:
    action = mzm.sample_action(true_state=s, legal_actions=s.legal_actions())
    s.apply_action(action)
    returns=s.returns()
    terminal=s.is_terminal()
    print(s)
    print()
