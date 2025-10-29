import pyspiel, torch, numpy as np

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
terminal = False
sum_returns = np.zeros(game.num_players())
while not terminal:
    enc_state = representation.encode(s)
    policy, value = prediction.policy_value(state=enc_state.unsqueeze(0))
    policy = policy.flatten()[s.legal_actions()]
    policy = policy / torch.sum(policy)
    abs_action = np.random.choice(s.legal_actions(), p=policy.detach().cpu().numpy())
    action = action_enc_dec.decode(state=s, action=abs_action)

    s, returns, terminal = dynamics.predict(state=s, action=action, mutate=False)
    print(s)
    print()
