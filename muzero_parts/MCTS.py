import math

import numpy as np, torch
from muzero_parts.dynamics import Dynamics


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

        'returns' (required only for terminal nodes) -> immediate returns at a node
            if available for nonzero nodes, the value of moving to a node is calculated as
                normal node's value + returns at the node
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
                'Qsa': np.zeros(1, self.num_players),
                'player': -1,
                'Ns': 0,
            }
        )

        root = self.make_leaf_node(state=state, parent=dummy_node, parent_action_idx=0, root=True, **kwargs)
        dummy_node.children = [root]
        return root

    def make_leaf_node(self, state, parent, parent_action_idx, **kwargs):
        raise NotImplementedError

    def get_node_value(self, node):
        return node.parent.data['Qsa'][node.data['parent_action_idx']]

    ### SELECTION
    def is_terminal(self, node):
        return node.data['terminal']

    def select_action_idx(self, node):
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

    def search(self, node: Node, state, dynamics: Dynamics):
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
        if self.not_expanded(node):
            child_node, value = self.expand_node_and_sim_value(node, state=state, dynamics=dynamics)
            self.backprop_childs_value(node=node,
                                       action_idx=child_node.data['parent_action_idx'],
                                       value=value)
            return value + node.data.get('returns', 0)
        # otherwise, we continue to one of the children
        action_idx = self.select_action_idx(node)
        child = node.children[action_idx]
        new_state = child.data.get('state', None)
        if new_state is None:
            new_state, returns, next_player, _ = dynamics.predict(state=state,
                                                                  action=self.get_action(node, state=state, action_idx=action_idx),
                                                                  mutate=False)
        v = self.search(node=child, state=new_state, dynamics=dynamics)
        self.backprop_childs_value(node, action_idx, v)
        return v + node.data.get('returns', 0)

    def getActionProb(self, state, num_sims, dynamics: Dynamics, player=0, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        root = self.make_root_node(state=state, player=player)
        for i in range(num_sims):
            self.search(root, state=state, dynamics=dynamics)
        counts = np.array([root.data['Nsa'][i] for i in range(len(root.children))])
        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = np.zeros_like(counts)
            probs[bestA] = 1
            return probs
        counts = np.power(counts, 1./temp)
        return counts/np.sum(counts)


class MCTS(AbsMCTS):
    """
    assumes we can list all legal actions at a state
    simple version that is not guided by an expansion policy
    """

    def __init__(self, num_players):
        super().__init__(num_players=num_players)

    def get_legal_actions(self, state):
        raise NotImplementedError

    def get_action(self, node, state, action_idx):
        return self.get_legal_actions(state)[action_idx]

    def make_leaf_node(self, state, parent, **kwargs):
        children = self.get_legal_actions(state=state)
        default_data = {
            'terminal': False,
            'Nsa': np.zeros(len(children)),
            'Qsa': np.zeros(len(children), self.num_players),
            'player': 0,
            'Ns': 0,
        }
        default_data.update(kwargs)
        root = Node(parent=parent, data=default_data)
        root.children = children
        return root

    def not_expanded(self, node):
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

        zeros = np.array(np.argwhere(node.data['Nsa'] == 0)).flatten()
        action_idx = np.random.choice(zeros)
        action = self.get_legal_actions(state=state)[action_idx]
        new_state, returns, next_player, terminal = dynamics.predict(state=state, action=action, mutate=False)
        leaf = self.make_leaf_node(state=new_state,
                                   parent=node,
                                   terminal=terminal,
                                   player=next_player,
                                   returns=returns,
                                   )
        if terminal:
            value = returns
        else:
            value = self.simulate_rollout(state=new_state, dynamics=dynamics)
        return leaf, value

    def select_action_idx(self, node):
        C = 1
        u = node.data['Qsa'][:, node.data['player']] + C*np.sqrt(node.data['Ns'])/(1 + node.data['Nsa'])
        return np.argmax(u)


class PyspielMCTS(MCTS):
    def get_legal_actions(self, state):
        return state.legal_actions()


if __name__ == '__main__':
    import pyspiel

    game = pyspiel.load_game('tic_tac_toe')
    mcts = PyspielMCTS(num_players=2)
