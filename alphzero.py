import torch, numpy as np

from muzero_parts.dynamics import Dynamics
from muzero_parts.representation import Representation
from muzero_parts.action_enc_dec import MuzeroActionEncDec
from muzero_parts.prediction import Prediction
from muzero_parts.MCTS import AbsMCTS


def get_trajectory(initial_state, representation: Representation, dynamics: Dynamics, mcts: AbsMCTS, player=0):
    """
    until a certian depth
    :param initial_state:
    :param representation:
    :param dynamics:
    :return:
    """
    state = representation.encode(initial_state)
    terminal = False
    traj_states = [state]  # len n+1 array of encoded states along path (including the terminal state, which has no associated policy/action)
    traj_policies = []  # len n array of MCTS policies from each state
    traj_actions = []  # len n array of actions taken
    traj_rewards = []  # len n array of rewards obtained at each action
    # in games like chess, this is usually all zeros except the last step

    traj_values = []  # len n array of MCTS root node values estimated at each state
    # in AZ paper, this is ignored, instead it learns based on the reward of the episode
    while not terminal:
        root, policy, value, actions = mcts.get_mcts_policy_value(state=state,
                                                                  num_sims=1,
                                                                  dynamics=dynamics,
                                                                  player=player,
                                                                  temp=1,
                                                                  root=None,
                                                                  depth=float('inf'),
                                                                  )
        action_idx = np.random.choice(np.arange(len(policy)), p=policy)
        action = mcts.get_action(node=root, state=state, action_idx=action_idx)
        next_state, reward, next_player, terminal = dynamics.predict(state=state, action=action, mutate=False)

        traj_states.append(next_state)  # terminal state is added here on last iteration
        traj_policies.append(policy)
        traj_actions.append(action)
        traj_rewards.append(reward)
        traj_values.append(value)

        state = next_state  # update state

    return initial_state, traj_states, traj_policies, traj_actions, sum(traj_rewards), traj_rewards, traj_values


def get_training_data_from_trajectory(trajectory, state_conversion=None):
    """
    returns training data from a trajectory
    :param trajectory:
    :param state_conversion: map of (state used in trajectory -> state to store for training)
        if None, just uses the identity
        this is useful if we want to for example convert pyspiel states to observation tensors
    :return:
    """
    initial_state, traj_states, traj_policies, traj_actions, rewards, traj_rewards, traj_values = trajectory
    data = []
    for i in range(len(traj_actions)):
        if state_conversion is None:
            data.append((traj_states[i], traj_policies[i], rewards))
        else:
            data.append((state_conversion(traj_states[i]), traj_policies[i], rewards))
    return data


if __name__ == '__main__':
    import pyspiel
    import ast, os
    from muzero_parts.dynamics import PyspielDynamics
    from muzero_parts.representation import PyspielObservationRepreseentation
    from muzero_parts.prediction import MuZeroPrediction
    from muzero_parts.MCTS import AlphaZeroMCTS

    game = pyspiel.load_game('tic_tac_toe')
    state = game.new_initial_state()
    f = open(os.path.join(os.path.dirname(__file__), 'networks', 'net_configs', 'ttt_pred.txt'), 'r')
    network_config = ast.literal_eval(f.read())
    f.close()
    prediction = MuZeroPrediction(network_config=network_config, representation=PyspielObservationRepreseentation(game=game))

    mcts = AlphaZeroMCTS(num_players=game.num_players(),
                         prediction=prediction,
                         is_pyspiel=True,
                         )
    get_trajectory(initial_state=state,
                   representation=Representation(),
                   dynamics=PyspielDynamics(),
                   mcts=mcts,
                   player=state.current_player(),
                   )
