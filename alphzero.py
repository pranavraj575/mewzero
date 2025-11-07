import torch, numpy as np

from muzero_parts.dynamics import Dynamics
from muzero_parts.representation import Representation
from muzero_parts.action_enc_dec import MuzeroActionEncDec
from muzero_parts.prediction import Prediction
from muzero_parts.MCTS import AbsMCTS


def get_trajectory(initial_state,
                   representation: Representation,
                   true_dynamics: Dynamics,
                   dynamics: Dynamics,
                   mcts: AbsMCTS,
                   player,
                   ):
    """
    until a certian depth
    :param initial_state:
    :param true_dynamics: used to enact the action in the true environment
    :param representation:
    :param dynamics:
    :return:
    """

    traj_true_states = []  # len n array of states along path (not including the terminal state)
    traj_states = []  # len n array of encoded states along path (not including the terminal state)
    traj_players = []  # len n array of players whose turn it is
    traj_policies = []  # len n array of MCTS policies from each state
    traj_actions = []  # len n array of actions taken
    traj_avail_actions = []  # len n array of actions possible at each step
    traj_rewards = []  # len n array of rewards obtained at each action
    # in games like chess, this is usually all zeros except the last step

    traj_values = []  # len n array of MCTS root node values estimated at each state
    # in AZ paper, this is ignored, instead it learns based on the reward of the episode

    true_state = initial_state
    state = representation.encode(true_state)
    terminal = False
    while not terminal:
        root, policy, value, actions = mcts.get_mcts_policy_value(state=state,
                                                                  num_sims=2000,
                                                                  dynamics=dynamics,
                                                                  player=player,
                                                                  temp=1,
                                                                  root=None,
                                                                  depth=float('inf'),
                                                                  )
        action_idx = np.random.choice(np.arange(len(policy)), p=policy)
        action = mcts.get_action(node=root, state=state, action_idx=action_idx)
        next_true_state, reward, next_player, terminal = true_dynamics.predict(state=true_state, player=player, action=action, mutate=False)
        next_state = representation.encode(next_true_state)

        traj_true_states.append(true_state)
        traj_states.append(state)
        traj_players.append(player)
        traj_policies.append(policy)
        traj_actions.append(action)
        traj_avail_actions.append(actions)
        traj_rewards.append(reward)
        traj_values.append(value)

        # update state and player
        true_state = next_true_state
        state = next_state
        player = next_player
    return {
        'traj_true_states': traj_true_states,
        'traj_states': traj_states,
        'traj_players': traj_players,
        'traj_policies': traj_policies,
        'traj_actions': traj_actions,
        'traj_avail_actions': traj_avail_actions,
        'rewards': sum(traj_rewards),
        'traj_rewards': traj_rewards,
        'traj_values': traj_values,
    }


def get_training_data_from_trajectory(trajectory, state_conversion=None):
    """
    returns training data from a trajectory
    :param trajectory:
    :param state_conversion: map of (state used in trajectory -> state to store for training)
        if None, just uses the identity
        this is useful if we want to for example convert pyspiel states to observation tensors
    :return:
    """
    data = []
    traj_states = trajectory['traj_states']
    rewards = trajectory['rewards']
    traj_mcts_values = trajectory['traj_values']
    traj_avail_actions = trajectory['traj_avail_actions']
    for i in range(len(trajectory['traj_actions'])):
        policy = trajectory['traj_policies'][i]
        if state_conversion is None:
            data.append((traj_states[i], policy, traj_avail_actions[i], rewards))
            # data.append((traj_states[i], policy, traj_avail_actions[i], traj_mcts_values[i]))
        else:
            data.append((state_conversion(traj_states[i]), policy, traj_avail_actions[i], rewards))
            # data.append((state_conversion(traj_states[i]), policy, traj_avail_actions[i], traj_mcts_values[i]))
    return data


def train(prediction: Prediction, data, optim):
    optim.zero_grad()
    mean_loss = 0.
    t = 0
    for state, targ_pol, avail_actions, rewards in data:
        policy, value = prediction.policy_value(states=state)
        policy = policy[:, avail_actions]
        policy = policy/torch.sum(policy)
        loss = torch.mean(torch.square(value - torch.tensor(rewards))) - torch.sum(torch.tensor(targ_pol)*torch.log(policy))
        mean_loss += loss
        t += 1
    mean_loss = mean_loss/t
    mean_loss.backward()
    optim.step()
    return mean_loss.item()


if __name__ == '__main__':
    import pyspiel
    import ast, os
    from muzero_parts.dynamics import PyspielDynamics
    from muzero_parts.representation import PyspielObservationRepreseentation
    from muzero_parts.prediction import MuZeroPrediction
    from muzero_parts.MCTS import AlphaZeroMCTS, MCTS
    from storage.replay_buffer import ReplayBufferList

    game = pyspiel.load_game('tic_tac_toe')
    state = game.new_initial_state()
    # effects appear less when searching from closer to root, potentially because the state is never reached
    # can def do something smarter, but this is fine for tests
    state.apply_action(0)
    state.apply_action(1)
    # state.apply_action(4)
    # state.apply_action(8)
    f = open(os.path.join(os.path.dirname(__file__), 'networks', 'net_configs', 'ttt_pred.txt'), 'r')
    network_config = ast.literal_eval(f.read())
    f.close()
    prediction = MuZeroPrediction(network_structure=network_config, representation=PyspielObservationRepreseentation(game=game))

    mcts = AlphaZeroMCTS(num_players=game.num_players(),
                         prediction=prediction,
                         is_pyspiel=True,
                         )
    cmp_mcts = MCTS(num_players=game.num_players(),
                    is_pyspiel=True,
                    )
    buff = ReplayBufferList(config={'tensor_tuple': False})
    optim = torch.optim.Adam(prediction.parameters())
    test_state = game.new_initial_state()
    test_state.apply_action(0)
    test_state.apply_action(1)
    test_state.apply_action(4)
    test_state.apply_action(8)
    root, corr_policy, corr_val, _ = cmp_mcts.get_mcts_policy_value(state=test_state, num_sims=2000, dynamics=PyspielDynamics(), player=test_state.current_player())

    for i in range(100):
        trajs = get_trajectory(initial_state=state,
                               representation=Representation(),
                               dynamics=PyspielDynamics(),
                               true_dynamics=PyspielDynamics(),
                               mcts=mcts,
                               player=state.current_player(),
                               )
        data = get_training_data_from_trajectory(trajs)
        buff.extend(data)
        loss = train(prediction=prediction, data=buff.sample(), optim=optim)
        print(i, 'mean loss:', loss)
        print(test_state)
        pol, val = prediction.policy_value(test_state)
        pol = pol.flatten().detach()[test_state.legal_actions()]
        val = val.flatten().detach().numpy()
        print('pred policy:', (pol/torch.sum(pol)).numpy())
        print('true policy:', corr_policy)
        print('pred value:', val)
        print('true value:', corr_val)
