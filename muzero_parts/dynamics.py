"""
dynamics maps (assumed to be torch modules)
 s,a -> s',r
in non-abstract spaces, this is the directly coded game dynamics, and does not need to be learned
in muzero, this is abstract, s and s' are abstract actions, a is a game action, and r is the reward obtained at the step
    the dynamics in this case is learned to predict r (and potentially s') accurately
in continuous muzero, a is also an abstract action

training using rollout, no enforced consistency:
    rollout produces (s'0,a'1,u1),(s'1,a'2,u2),...
        ui are the immediate rewards at each timestep (scalar for single/double player, can be vector for multiplayer games)
        s'i,a'i are the true states/actions
    define s0=enc(s'0) as the encoded s'0, ai as encoded a'i (ai=a'i for nomral muzero),
        and use the dynamics function T to produce s{i+1},r{i+1} = T(si,a{i+1})
    then take loss l(ri,ui) enforcing ri \approx ui

training with enforced consistency:
    same scenario, except we also enforce that for s,_ = T(enc(s'{i-i}),ai), s \approx enc(s'i)

NOTE: muzero has no idea in general whether an abstract state is terminal, and will keep searching past this
    in the MCTS search (https://www.julian.ac/blog/2020/12/22/muzero-intuition/)
        during training time "treat terminal states as absorbing" is what online posts say
    not an issue for fixed depth games
"""
import numpy as np

class Dynamics:
    def __init__(self):
        super().__init__()

    def predict(self, state, action, mutate=False):
        """
        :param state: current (abstract or true) state of game
        :param action: (abstract or true) game action
        :param mutate: whether state is allowed to be mutated
        :return: (next state, reward, next_player, terminal)
            next state is the predicted next state of the game. If state is abstract, next state is abstract,
            reward is a real number (or vector), the reward obtained (for each player) at this transition
            next_player is the index of next player
                for most games this alternates between player 0 and player 1
            terminal is a boolean for if the game terminates
                for learned dynamics, we will always return terminal=False,
                    unless we are in a special scenario like fixed-depth games
        """
        raise NotImplementedError

    def consistency_loss(self, state_encoder, true_state, action, true_next_state, ):
        pass


class PyspielDynamics(Dynamics):
    """
    assumes state is a PySpiel state
    uses state.apply_action to produce next state
        rewards only occur at terminal states, otherwise returns 0 for the reward
    """

    def __init__(self):
        super().__init__()

    def predict(self, state, action, mutate=False):
        if mutate:
            new_state = state
            new_state.apply_action(action)
        else:
            new_state = state.child(action)
        return new_state, np.array(new_state.returns()), new_state.current_player(), new_state.is_terminal()


if __name__ == '__main__':
    import pyspiel
    import numpy as np

    g = pyspiel.load_game('universal_poker', {k: pyspiel.GameParameter(v) for k, v in (
        ('numPlayers', 3),
        ('stack', "12 12 12"),
        ('blind', "1 1 1"),
    )})
    state = g.new_initial_state()
    terminal = False
    dynamics = PyspielDynamics()
    sum_returns = np.zeros(3)
    while not terminal:
        state, returns, player, terminal = dynamics.predict(state=state, action=np.random.choice(state.legal_actions()), mutate=True)
        sum_returns += returns
        print(state)
    print(sum_returns)
