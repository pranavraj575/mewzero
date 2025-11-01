import numpy as np
import torch

from muzero_parts.dynamics import Dynamics
from muzero_parts.representation import Representation
from muzero_parts.prediction import Prediction


class Node:
    # list of children, assumes actions are ordered as well
    children: list

    def __init__(self, parent, data, children=None):
        self.parent = parent
        self.data = data
        self.children = children  # children can be set/updated later


class AbsMCTS:
    """
    This class handles the MCTS tree.
    nodes value is stored in the q value of parent: node.parent.data[node.data['parent_action_idx']]

    Assumes node has these keys (along with a pointer to parent):
        'terminal' -> whether the node is terminal
        'Nsa' -> array of at least size (num expanded actions) that tracks number of times each action was visited
        'Qsa' -> matrix of size (num expanded actions, num players), node.data['Qsa'][action] is the returns expected for each player
        'Ns' -> total number of visits, equal to sum(node.data['Nsa'])
        'parent_action_idx' -> action index to go from parent to child
    Optional keys:
        'root' -> whether node is root, if unspecified, the node is not root
        'dummy' -> whether node is dummy (parent of root)
        'state' -> if specified, stores the state of each node instead of recalculating it from action sequences
            trades computation for RAN space
        'legal_action_mask' -> vector of size (num actions), ones where actions are legal and zero otherwise
            This is used in Muzero, since only the root node needs to worry about illegal actions
        'returns' (required only for terminal nodes) -> immediate returns at a node
            if available for nonzero nodes, the value of moving to a node is calculated as
                normal node's value + returns at the node
        'actions' -> list of actions taken that correspond with the list of node children
            this is not required in MCTS/alphazero/muzero, as actions are the same as indices
            however, in continuous space games, we will need to sample actions and store them, so we will use that for this
    """

    def __init__(self, num_players):
        self.num_players = num_players

    def make_root_node(self, state, **kwargs):
        dummy_node = Node(
            parent=None,
            data={
                'terminal': False,
                'dummy': True,
                'Nsa': np.zeros(1),
                'Qsa': np.zeros((1, self.num_players)),
                'player': -1,
                'Ns': 0,
            }
        )
        dummy_node.children = [None]  # will be set to [root] in make_leaf_node
        root = self.make_leaf_node(state=state,
                                   parent=dummy_node,
                                   parent_action_idx=0,
                                   terminal=False,
                                   root=True,
                                   **kwargs,
                                   )
        return root

    def make_leaf_node(self, state, parent, parent_action_idx, terminal, **kwargs):
        """
        creates leaf node with state 'state' from taking parent_action_idx at parent
        updates parent's children to include this node
        """
        raise NotImplementedError

    def get_node_value(self, node):
        return node.parent.data['Qsa'][node.data['parent_action_idx']]

    ### SELECTION
    def is_terminal(self, node):
        return node.data['terminal']

    def select_action_idx(self, node, state):
        """
        selection of the action to take after node
        we know that self.not_expanded(node) is False
        if 'legal_action_mask' is in node.data, must restrict to only where this vector is 1
        :param node:
        :return:
        """
        raise NotImplementedError

    ### EXPANSION/simulation

    def not_expanded(self, node):
        raise NotImplementedError

    def expand_node_and_sim_value(self, node, state, dynamics: Dynamics):
        """
        called when we arrive at a node that is a leaf
        we know that self.not_expanded(node) is True.
        :param node:
        :return: the child node created, value obtained from simulation
        """
        raise NotImplementedError

    def get_action(self, node, state, action_idx):
        raise NotImplementedError

    ### BACKPROP
    def backprop_childs_value(self, node, action_idx, value):
        Nsa = node.data['Nsa'][action_idx]
        Qsa = node.data['Qsa'][action_idx]
        node.data['Qsa'][action_idx] = (Nsa*Qsa + value)/(Nsa + 1)
        node.data['Nsa'][action_idx] = Nsa + 1

        node.data['Ns'] += 1

    def search(self, node: Node, state, dynamics: Dynamics, depth=float('inf')):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            value estimate to backprop to parent
        """

        if self.is_terminal(node):  # do not need to update node.data['Qsa'], so send the returns value to parent
            return node.data['returns']
        if depth <= 0:
            # TODO: here, we stop and call an evaluation function
            #  also we should split expand_node_and_sim_value into expand node, sim value
            pass
        if self.not_expanded(node):
            child_node, value = self.expand_node_and_sim_value(node, state=state, dynamics=dynamics)
            self.backprop_childs_value(node=node,
                                       action_idx=child_node.data['parent_action_idx'],
                                       value=value)
            return value + node.data.get('returns', 0)
        # otherwise, we continue to one of the children
        action_idx = self.select_action_idx(node, state=state)
        child = node.children[action_idx]
        new_state = child.data.get('state', None)
        if new_state is None:
            new_state, returns, next_player, _ = dynamics.predict(state=state,
                                                                  action=self.get_action(node, state=state, action_idx=action_idx),
                                                                  mutate=False)
        v = self.search(node=child, state=new_state, dynamics=dynamics, depth=depth - 1)
        self.backprop_childs_value(node, action_idx, v)
        if node.data.get('root', False):  # also store the value of the root node, if we ever want to use this
            self.backprop_childs_value(node.parent, node.data['parent_action_idx'], v)
        return v + node.data.get('returns', 0)

    def get_mcts_policy_value(self, state, num_sims, dynamics: Dynamics, player=0, temp=1, root=None, depth=float('inf')):
        """
        runs num_sims of MCTS search and returns the action probabilities (based on visits of root children)
        Args:
            depth: if not infinity, only grows MCTS tree to a certian depth, and relies on the value function for leaves
                this is useful in Muzero if we know the game has a certian max depth
        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
            value: value of root node for each player
            actions: actions correspoinding to policy, used for continuous space search
                None usually, as this is implied in MCTS, Alphazero, Muzero
        """
        if root is None:
            root = self.make_root_node(state=state, player=player)
        for _ in range(num_sims):
            self.search(root, state=state, dynamics=dynamics, depth=depth)
        value = self.get_node_value(root)
        counts = np.array([root.data['Nsa'][i] for i in range(len(root.children))])
        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = np.zeros_like(counts)
            probs[bestA] = 1
            return probs, value, root.data.get('actions', None)
        counts = np.power(counts, 1./temp)
        return counts/np.sum(counts), value, root.data.get('actions', None)


class MCTS(AbsMCTS):
    """
    assumes we can list all legal actions at a state
    simple version that is not guided by an expansion policy
    """

    def __init__(self, num_players, is_pyspiel=False, exploration_constant=np.sqrt(2)):
        """
        :param is_pyspiel: whether every state is a pyspiel state
        """
        super().__init__(num_players=num_players)
        self.is_pyspiel = is_pyspiel
        self.exploration_constant = exploration_constant

    def get_legal_actions(self, state):
        if self.is_pyspiel:
            return state.legal_actions()
        raise NotImplementedError

    def get_action(self, node, state, action_idx):
        return self.get_legal_actions(state)[action_idx]

    def make_leaf_node(self, state, parent, parent_action_idx, terminal, **kwargs):
        legal_actions = self.get_legal_actions(state=state)
        default_data = {
            'terminal': terminal,
            'Nsa': np.zeros(len(legal_actions)),
            'Qsa': np.zeros((len(legal_actions), self.num_players)),
            'player': 0,
            'Ns': 0,
            'parent_action_idx': parent_action_idx,
        }
        default_data.update(kwargs)
        leaf = Node(parent=parent, data=default_data)
        leaf.children = [None for _ in legal_actions]
        parent.children[parent_action_idx] = leaf
        return leaf

    def not_expanded(self, node):
        if 'legal_action_mask' in node.data:
            return np.any(np.logical_and(node.data['Nsa'] == 0, node.data['legal_action_mask'] == 1))
        else:
            return np.any(node.data['Nsa'] == 0)

    def simulate_rollout(self, state, dynamics: Dynamics):
        terminal = False
        value = 0  # accumulate returns along path, though this is usually just one return at terminal
        # value can end up being a np array, depending on the type of returns
        while not terminal:
            action = np.random.choice(self.get_legal_actions(state))
            state, returns, next_player, terminal = dynamics.predict(state=state, action=action, mutate=False)
            value += returns
        return value

    def expand_node_and_sim_value(self, node, state, dynamics: Dynamics):
        # dont need to mess with storing priors here, since this is simple MCTS
        # however, we do need a value estimate
        if 'legal_action_mask' in node.data:
            valid = np.logical_and(node.data['Nsa'] == 0, node.data['legal_action_mask'] == 1)
            zeros = np.array(valid).flatten()
        else:
            zeros = np.array(np.argwhere(node.data['Nsa'] == 0)).flatten()

        action_idx = np.random.choice(zeros)
        action = self.get_legal_actions(state=state)[action_idx]
        new_state, returns, next_player, terminal = dynamics.predict(state=state, action=action, mutate=False)
        leaf = self.make_leaf_node(state=new_state,
                                   parent=node,
                                   terminal=terminal,
                                   player=next_player,
                                   returns=returns,
                                   parent_action_idx=action_idx,
                                   )
        if terminal:
            value = returns
        else:
            value = self.simulate_rollout(state=new_state, dynamics=dynamics)
        return leaf, value

    def select_action_idx(self, node, state):
        if self.is_pyspiel and state.is_chance_node():
            outcomes = state.chance_outcomes()
            return np.random.choice([a for a, p in outcomes], p=[p for a, p in outcomes])
        # since we have that self.not_expanded(node) is False, node.data['Nsa'] are all at least 1, except in
        #   places where the legal_action_mask is 0, which we ignore anyway. Thus, clipping is safe here
        u = self.exploration_constant*np.sqrt(np.log(node.data['Ns'])/np.clip(node.data['Nsa'], 1., np.inf))
        ucb = node.data['Qsa'][:, node.data['player']] + u
        if 'legal_action_mask' in node.data:
            illegal = np.argwhere(node.data['legal_action_mask'] == 0)
            ucb[illegal] = -np.inf
        return np.argmax(ucb)


class AlphaZeroMCTS(MCTS):
    """
    Alphazero version of MCTS guided by a policy/value
    nodes now also store a prior policy (returned by policy_value_net)
        'prior_policy': probability distribution over valid action indexes
    """

    def __init__(self, num_players, prediction: Prediction, is_pyspiel=False, exploration_constant=np.sqrt(2)):
        """
        :param num_players:
        :param prediction:
            map from state -> (policy tensor, value tensor)
                policy is over all possible actions, this is restricted to only legal actions and renormalized by self.create_leaf_node
        :param is_pyspiel:
        """
        super().__init__(num_players, is_pyspiel=is_pyspiel, exploration_constant=exploration_constant)
        self.prediciton = prediction

    def restrict_policy(self, policy, legal_action_indices):
        """
        restricts torch policy vector to only indices with legal actions
        """
        policy = policy[legal_action_indices]
        return policy/torch.sum(policy)

    def add_direchlet_noise(self, policy):
        eps = .25
        direchlet = torch.distributions.dirichlet.Dirichlet(
            torch.ones(len(policy))*0.3
        ).sample()
        return policy*(1 - eps) + eps*direchlet

    def make_leaf_node(self, state, parent, parent_action_idx, terminal, **kwargs):
        if terminal:
            # do not produce a prior policy and value for terminal nodes
            return super().make_leaf_node(state=state,
                                          parent=parent,
                                          parent_action_idx=parent_action_idx,
                                          terminal=terminal,
                                          **kwargs)
        policy, value = self.prediciton.policy_value(state)
        legal_action_indices = self.get_legal_actions(state=state)
        policy = self.restrict_policy(policy=policy.flatten(),
                                      legal_action_indices=legal_action_indices,
                                      )
        if kwargs.get('root', False):
            # add direchlet noise when creating the root
            policy = self.add_direchlet_noise(policy=policy)
        value = value.flatten()
        return super().make_leaf_node(state=state,
                                      parent=parent,
                                      parent_action_idx=parent_action_idx,
                                      terminal=terminal,
                                      prior_policy=policy.detach().cpu().numpy(),
                                      prior_value=value.detach().cpu().numpy(),
                                      actions=legal_action_indices,
                                      **kwargs)

    def expand_node_and_sim_value(self, node, state, dynamics: Dynamics):
        prior = node.data['prior_policy']
        # pick the maximizer of the prior, over the actions that have not been visited
        # use prior + 1 in case the prior is zero in some entries, though this should not happen
        if 'legal_action_mask' in node.data:
            valid = np.logical_and(node.data['Nsa'] == 0, node.data['legal_action_mask'] == 1)
            action_idx = np.argmax((prior + 1)*valid)
        else:
            action_idx = np.argmax((prior + 1)*(node.data['Nsa'] == 0))
        action = self.get_legal_actions(state=state)[action_idx]
        new_state, returns, next_player, terminal = dynamics.predict(state=state, action=action, mutate=False)
        leaf = self.make_leaf_node(state=new_state,
                                   parent=node,
                                   terminal=terminal,
                                   player=next_player,
                                   returns=returns,
                                   parent_action_idx=action_idx,
                                   )
        if terminal:
            value = returns
        else:
            value = leaf.data['prior_value']  # use evaluation instead of a full rollout
        return leaf, value

    def select_action_idx(self, node, state):
        if self.is_pyspiel and state.is_chance_node():
            return super().select_action_idx(node=node, state=state)
        # this changes because of experimental results for AlphaZero, we no longer need to worry about div 0 either
        u = self.exploration_constant*node.data['prior_policy']*np.sqrt(node.data['Ns'])/(1 + node.data['Nsa'])
        ucb = node.data['Qsa'][:, node.data['player']] + u
        if 'legal_action_mask' in node.data:
            illegal = np.argwhere(node.data['legal_action_mask'] == 0)
            ucb[illegal] = -np.inf
        return np.argmax(ucb)


class MuZeroMCTS(AlphaZeroMCTS):
    """
    very similar to Alphazero, except
        We search over abstract states, so the input to getActionProb must be an abstract state
        All moves are assumed to be legal, except for at the root of the search, where we have access to true state
            Thus, get_legal_actions always returns range(self.num_distinct_actions)
        There are generally no terminal nodes, as we rely on a learned dynamics model
    """

    def __init__(self, num_players, prediction, num_distinct_actions, exploration_constant=np.sqrt(2)):
        super().__init__(
            num_players=num_players,
            prediction=prediction,
            is_pyspiel=False,
            exploration_constant=exploration_constant,
        )
        self.num_distinct_actions = num_distinct_actions

    def get_legal_actions(self, state):
        return list(range(self.num_distinct_actions))

    def make_leaf_node(self, state, parent, parent_action_idx, terminal, **kwargs):
        """
        maybe should store the state of the leaf
        """
        leaf = super().make_leaf_node(state=state,
                                      parent=parent,
                                      parent_action_idx=parent_action_idx,
                                      terminal=terminal,
                                      **kwargs)
        leaf.data['state'] = state
        return leaf

    def get_mcts_policy_value(self, state, num_sims, dynamics: Dynamics, legal_action_indices=None, player=0, temp=1, root=None, depth=float('inf')):
        """
        :param state: ABSTRACT state!!!
        :param legal_action_indices: list of indices of legal actions
            if none, assumes all actions are legal from root
        same as in super, except adds a legal action mask to the root node
        """
        if legal_action_indices is not None and root is None:
            root = self.make_root_node(state=state, player=player)
            mask = np.zeros(self.num_distinct_actions)
            mask[legal_action_indices] = 1.
            root.data['legal_action_mask'] = mask
        return super().get_mcts_policy_value(state=state,
                                             num_sims=num_sims,
                                             dynamics=dynamics,
                                             player=player,
                                             temp=temp,
                                             root=root,
                                             depth=depth,
                                             )


if __name__ == '__main__':
    import pyspiel
    import torch
    from muzero_parts.dynamics import PyspielDynamics

    game = pyspiel.load_game('tic_tac_toe')
    state = game.new_initial_state()
    state.apply_action(4)
    state.apply_action(1)
    state.apply_action(8)
    state.apply_action(0)
    state.apply_action(2)
    print(state)
    dynamics = PyspielDynamics()
    num_actions = game.num_distinct_actions()
    # alphazero version where the policy is always uniform over A, value is alwyas [0,0]
    mcts = AlphaZeroMCTS(num_players=game.num_players(), is_pyspiel=True,
                         prediction=lambda state: (torch.ones(num_actions)/num_actions, torch.zeros(game.num_players()))
                         )
    print(mcts.get_mcts_policy_value(state=state, num_sims=10000, dynamics=dynamics, player=state.current_player(), temp=1))
