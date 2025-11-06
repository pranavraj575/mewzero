import torch, numpy as np

from muzero_parts.dynamics import Dynamics, LearnedDynamics
from muzero_parts.representation import Representation, LearnedRepreseentation
from muzero_parts.action_enc_dec import MuzeroActionEncDec
from muzero_parts.prediction import Prediction
from muzero_parts.MCTS import MuZeroMCTS


def tensor_idx(s):
    if torch.is_tensor(s):
        return s
    return torch.tensor([s])


def get_trajectory(initial_state,
                   representation: Representation,
                   true_dynamics: Dynamics,
                   dynamics: Dynamics,
                   mcts: MuZeroMCTS,
                   player,
                   is_pyspiel=False,
                   ):
    """
    until a certian depth
    :param initial_state:
    :param true_dynamics: used to enact the action in the true environment
    :param representation:
    :param dynamics:
    :param is_pyspiel: whether the true states are pyspiel states, so we can query legal_actions
    :return:
    """

    traj_true_states = []  # len n array of states along path (not including the terminal state)
    traj_states = []  # len n+1 array of encoded states along path (including the terminal state)
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
                                                                  legal_action_indices=true_state.legal_actions() if is_pyspiel else None,
                                                                  )

        action_idx = np.random.choice(np.arange(len(policy)), p=policy)
        action = mcts.get_action(node=root, state=state, action_idx=action_idx)
        next_true_state, reward, next_player, terminal = true_dynamics.predict(state=true_state,
                                                                               player=player,
                                                                               action=action,
                                                                               mutate=False)

        traj_true_states.append(true_state)
        traj_states.append(state.detach())
        traj_players.append(player)
        traj_policies.append(policy)

        traj_actions.append(action)
        # TODO: THIS
        traj_avail_actions.append(root.data['legal_action_mask'])
        traj_rewards.append(reward)
        traj_values.append(value)

        if terminal:
            # add terminal state to states
            terminal_state, _, _, _ = dynamics.predict(state=state,
                                                       player=tensor_idx(player),
                                                       action=tensor_idx(action),
                                                       mutate=False
                                                       )
            traj_states.append(terminal_state.detach())
        else:
            # update state and player
            true_state = next_true_state
            next_state = representation.encode(next_true_state)
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


def get_prediciton_training_data(trajectory, state_conversion=None):
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
    terminal_state = traj_states[-1]
    if state_conversion is None:
        data.append((terminal_state, None, None, rewards))
    else:
        data.append((state_conversion(terminal_state), None, None, rewards))

    return data


def get_dynamics_training_data(trajectory, state_conversion=None):
    """
    returns training data from a trajectory
    :param trajectory:
    :param state_conversion: map of (state used in trajectory -> state to store for training)
        if None, just uses the identity
        this is useful if we want to for example convert pyspiel states to observation tensors
    :return:
    """
    data = []
    traj_true_states = trajectory['traj_true_states']
    traj_rewards = trajectory['traj_rewards']
    traj_actions = trajectory['traj_actions']
    traj_players = trajectory['traj_players']
    for i in range(len(traj_true_states)):
        true_state = traj_true_states[i]
        true_next_state = traj_true_states[i + 1] if i + 1 < len(traj_true_states) else None
        if state_conversion is not None:
            true_state = state_conversion(true_state)
            if true_next_state is not None:
                true_next_state = state_conversion(true_next_state)
        data.append((true_state, true_next_state, traj_players[i:], traj_actions[i:], traj_rewards[i:]))
    return data


def train_representation_dynamics(representation: Representation, dynamics: LearnedDynamics, data, optim, sample_action):
    optim.zero_grad()
    mean_returns_loss = torch.tensor(0.)  # want returns to be close to the true returns
    mean_consistency_loss = torch.tensor(0.)  # want dynamics(encode(state),action) to be similar to encode(next_state)
    mean_terminal_state_loss = torch.tensor(0.)  # if state is an encoding of a terminal state, we want dynamics(state,action) to have zero reward
    t_ret = 0
    t_cons = 0
    t_ts = 0
    for true_state, true_next_state, traj_players, traj_actions, traj_rewards in data:
        state = representation.encode(true_state)
        # pred_traj_players=[]
        pred_traj_returns = []
        for player, action in zip(traj_players, traj_actions):
            new_state, returns, next_player, _ = dynamics.predict(state=state,
                                                                  player=tensor_idx(player),
                                                                  action=tensor_idx(action),
                                                                  mutate=False)
            state = new_state
            pred_traj_returns.append(returns)
        # mse over the returns
        returns_loss = sum(torch.mean(torch.square(torch.tensor(r) - rp))
                           for r, rp in zip(traj_rewards, pred_traj_returns))/len(pred_traj_returns)
        mean_returns_loss += returns_loss

        t_ret += 1
        if true_next_state is not None:
            pred_next_state, _, _, _ = dynamics.predict(
                state=representation.encode(true_state),
                player=tensor_idx(traj_players[0]),
                action=tensor_idx(traj_actions[0]),
                mutate=False)

            consistency_loss = torch.mean(torch.square(pred_next_state - representation.encode(true_next_state)))
            mean_consistency_loss += consistency_loss
            t_cons += 1
        else:
            # (true_state,traj_actions[0]) leads to a terminal state
            # taking any action from a terminal state should result in zero returns
            # we can also enforce that taking any action from a terminal state does not change the state
            state = representation.encode(true_state)
            terminal_state, _, next_player, _ = dynamics.predict(
                state=state,
                player=tensor_idx(traj_players[0]),
                action=tensor_idx(traj_actions[0]),
                mutate=False)
            bonus_state, returns, _, _ = dynamics.predict(
                state=terminal_state,
                player=tensor_idx(next_player),
                action=tensor_idx(sample_action(representation.encode(true_state))),
                mutate=False)
            terminal_state_loss = torch.mean(torch.square(returns)) + torch.mean(torch.square(bonus_state - terminal_state))
            mean_terminal_state_loss += terminal_state_loss
            t_ts += 1

    mean_returns_loss = mean_returns_loss/t_ret
    if t_cons > 0:
        mean_consistency_loss = mean_consistency_loss/t_cons
    if t_ts > 0:
        mean_terminal_state_loss = mean_terminal_state_loss/t_ts
    loss = mean_returns_loss + mean_consistency_loss + mean_terminal_state_loss
    loss.backward()
    optim.step()
    return mean_returns_loss.item(), mean_consistency_loss.item(), mean_terminal_state_loss.item()


def train_prediction(prediction: Prediction, data, optim):
    optim.zero_grad()
    mean_loss = 0.
    t = 0
    for state, targ_pol, avail_actions, rewards in data:
        if targ_pol is None:
            # terminal state, predict value and ignore policy
            policy, value = prediction.policy_value(states=state)
            loss = torch.mean(torch.square(value - torch.tensor(rewards)))
        else:
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
    from muzero_parts.dynamics import PyspielDynamics, LearnedDynamics
    from muzero_parts.representation import PyspielObservationRepreseentation, ChainRepresentation, LearnedRepreseentation
    from muzero_parts.prediction import MuZeroPrediction
    from muzero_parts.MCTS import AlphaZeroMCTS, MCTS
    from storage.replay_buffer import ReplayBufferList

    game = pyspiel.load_game('tic_tac_toe')
    state = game.new_initial_state()

    true_dynamics = PyspielDynamics()

    # learned dynamics uses abstract state
    f = open(os.path.join(os.path.dirname(__file__), 'networks', 'net_configs', 'ttt_dyn_with_plyr.txt'), 'r')
    network_config = ast.literal_eval(f.read())
    f.close()
    learned_dynamics = LearnedDynamics(network_structure=network_config)

    # prediction uses abstract state
    f = open(os.path.join(os.path.dirname(__file__), 'networks', 'net_configs', 'ttt_abs_pred.txt'), 'r')
    network_config = ast.literal_eval(f.read())
    f.close()
    prediction = MuZeroPrediction(network_structure=network_config)

    # representation goes from pyspiel state -> tensor
    f = open(os.path.join(os.path.dirname(__file__), 'networks', 'net_configs', 'ttt_rep.txt'), 'r')
    network_config = ast.literal_eval(f.read())
    f.close()
    representation = ChainRepresentation(
        [
            PyspielObservationRepreseentation(game=game),
            LearnedRepreseentation(network_config=network_config),
        ]
    )

    mcts = MuZeroMCTS(num_players=game.num_players(),
                      prediction=prediction,
                      num_distinct_actions=game.num_distinct_actions()
                      )
    cmp_mcts = MCTS(num_players=game.num_players(),
                    is_pyspiel=True,
                    )
    buff_pred = ReplayBufferList(config={'tensor_tuple': False})
    buff2 = ReplayBufferList(config={'tensor_tuple': False})
    optim_pred = torch.optim.Adam(prediction.network.parameters())
    optim2 = torch.optim.Adam(list(representation.parameters()) + list(learned_dynamics.network.parameters()))
    test_state = game.new_initial_state()
    # test_state.apply_action(1)
    # test_state.apply_action(0)
    # test_state.apply_action(2)
    # for i in range(4):
    #    test_state.apply_action(np.random.choice(test_state.legal_actions()))
    root, corr_policy, corr_val, _ = cmp_mcts.get_mcts_policy_value(state=test_state, num_sims=2000, dynamics=true_dynamics,
                                                                    player=test_state.current_player())


    def save(represenation: Representation,
             dynamics: LearnedDynamics, prediciton: MuZeroPrediction, folder):
        represenation.save(os.path.join(folder, 'rep'))
        dynamics.save(os.path.join(folder, 'dyn'))
        prediciton.save(os.path.join(folder, 'pred'))


    def load(represenation: Representation,
             dynamics: LearnedDynamics, prediciton: MuZeroPrediction, folder):
        represenation.load(os.path.join(folder, 'rep'))
        dynamics.load(os.path.join(folder, 'dyn'))
        prediciton.load(os.path.join(folder, 'pred'))


    save_dir = os.path.join(os.path.dirname(__file__), 'output', 'mz_test')
    if True:
        i = 0
        for i in range(3000):
            if not i%10:
                save(representation, dynamics=learned_dynamics, prediciton=prediction,
                     folder=os.path.join(save_dir, str(i)))
            trajs = get_trajectory(initial_state=state,
                                   representation=representation,
                                   true_dynamics=true_dynamics,
                                   dynamics=learned_dynamics,
                                   mcts=mcts,
                                   player=state.current_player(),
                                   is_pyspiel=True,
                                   )
            data_pred = get_prediciton_training_data(trajs)
            data2 = get_dynamics_training_data(trajs)
            buff_pred.extend(data_pred)
            buff2.extend(data2)
            loss_pred = train_prediction(prediction=prediction, data=buff_pred.sample(), optim=optim_pred)
            loss2 = train_representation_dynamics(representation=representation,
                                                  dynamics=learned_dynamics,
                                                  data=buff2.sample(),
                                                  optim=optim2,
                                                  sample_action=lambda s: torch.randint(0, 9, (1,)),
                                                  )

            print(i, 'loss prediciton:', loss_pred)
            print(i, 'loss rep/dynamics:', loss2)
            print(test_state)
            pol, val = prediction.policy_value(representation.encode(test_state))
            pol = pol.flatten().detach()[test_state.legal_actions()]
            val = val.flatten().detach().numpy()
            print('pred policy:', (pol/torch.sum(pol)).numpy())
            print('true policy:', corr_policy)
            print('pred value:', val)
            print('true value:', corr_val)
        save(representation, dynamics=learned_dynamics, prediciton=prediction,
             folder=os.path.join(save_dir, str(i + 1)))
    import matplotlib.pyplot as plt

    print('true value:', corr_val)
    d = []
    for i in range(4, 1000, 5):
        load(represenation=representation, dynamics=learned_dynamics, prediciton=prediction,
             folder=os.path.join(save_dir, str(i)))
        pol, val = prediction.policy_value(representation.encode(test_state))
        d.append(val.flatten()[0].detach().cpu().item())
        print(val)
    print('true value:', corr_val)
    print(test_state)
    plt.plot(range(4, 1000, 5), d)
    plt.plot([4, 999], (corr_val[0], corr_val[0]), linestyle='--')
    for i in range(9):
        action = np.random.choice(test_state.legal_actions())
        state, rewards, next_player, _ = learned_dynamics.predict(state=representation.encode(test_state),
                                                                  player=torch.tensor([test_state.current_player()]),
                                                                  action=torch.tensor([action]))
        pol, val = prediction.policy_value(state)
        print('value estimate', val.detach().cpu().flatten().numpy())
        print('dynamics returns', rewards.detach().cpu().flatten().numpy())
        print('state difference', torch.mean(torch.square(representation.encode(test_state) - state)).detach().cpu().numpy())
        test_state.apply_action(action)
        print(test_state)
        if test_state.is_terminal():
            bonus_state, rewards, _, _ = learned_dynamics.predict(state=state,
                                                                  player=torch.tensor([next_player]),
                                                                  action=torch.tensor([action]))
            pol, val = prediction.policy_value(bonus_state)
            print('post terminal')
            print('state difference', torch.mean(torch.square(bonus_state - state)).detach().cpu().numpy())
            print('dynamic returns', rewards.detach().cpu().flatten().numpy())
            print('value estimate', val.detach().cpu().flatten().numpy())

        print()
        if test_state.is_terminal():
            break
    plt.show()
