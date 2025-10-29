import pyspiel

from muzero_parts.dynamics import PyspielDynamics
from muzero_parts.representation import PyspielObservationRepreseentation
from muzero_parts.action_enc_dec import IdentityActionEncDec
from muzero_parts.prediction import MuZeroPrediction

game = pyspiel.load_game("tic_tac_toe")

dynamics = PyspielDynamics()
representation = PyspielObservationRepreseentation(game=game)
action_enc_dec = IdentityActionEncDec()
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
s = game.new_initial_state()
print(dynamics.predict(state=s, action=0, mutate=False))
