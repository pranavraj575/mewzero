"""
muzero policy/value prediction functions
    s -> Delta(A), v
    where v is either a real number (single/two player) or vector

value function is easy, and can usually be a learned FFN
when there are a finite number of total actions possible, policy function can also be a FFN or similar
If the action space is continuous or discrete and infinite, this is a bit annoying
    continuous:
        must produce a continuous distribution pi(A | S)
        Can learn a static distribution from training data (i.e. through a VAE) and not update this
        otherwise must use tricks to update pi given improvements on sampled actions
            i.e. sampled a1,...,ak, found through MCTS search that a1 performed better, so we must push distribution towards a1
                arXiv:2104.06303
    discrete and countably infinite:
         annoying in a different way, as we cannot take advantage of euclidean space topology
         if finite size at every state (i.e. jenga, 'increasing size board game'), we may produce a policy via a map
            (s,a) -> R, which is then passed throuch softmax to produce a distribution over valid actions
"""
import torch, os
from torch import nn

from networks.nn_from_config import CustomNN
from muzero_parts.representation import Representation


class Prediction(nn.Module):
    finite_action_space = False  # whether there is a fixed size action space
    num_actions = -1  # if finite_action_space, stores the size

    def __init__(self, representation: Representation = None):
        super().__init__()
        self.representation = representation

    def policy_only(self, states):
        return self.policy_value(states)[0]

    def value_only(self, states):
        return self.policy_value(states)[1]

    def sample_actions(self, states):
        raise NotImplementedError

    def policy_value(self, states):
        """
        ONLY VALID FOR FINITE ACTION SPACES
        :param states: batch of states
        :return: (policy, value), both tensors, policy is in Delta(A), value is a real number or a vector
        """
        raise NotImplementedError

    def save(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def load(self, save_dir):
        pass


class MuZeroPrediction(Prediction):
    """
    standard muzero prediction
    fixed state representation (i.e. shaped tensor)
    finite action space
    """
    finite_action_space = True

    def __init__(self, network_structure, representation: Representation = None):
        """
        :param network_structure:
            produces a network (see nn_from_config) that returns (policy, value)
            this parameter will imply the num actions, as this will be the size of the policy vector
        """
        super().__init__(representation=representation)
        self.network = CustomNN(structure=network_structure)
        self.num_actions = self.network.output_shape[0]
        self.unbatched_input_shape = network_structure['input_shape']

    def policy_value(self, states):
        if self.representation is None:
            return self.network(states)
        else:
            return self.network(self.representation.encode(states))

    def sample_actions(self, states):
        # (m,num_actions)
        dist = self.policy_only(states=states).detach().cpu()
        # (m,1).flatten() sample of actions
        return torch.multinomial(dist, 1).flatten().numpy()

    def save(self, save_dir):
        super().save(save_dir)
        self.network.state_dict()
        torch.save(self.network.state_dict(), os.path.join(save_dir, 'net.pkl'))

    def load(self, save_dir):
        super().load(save_dir)
        self.network.load_state_dict(torch.load(os.path.join(save_dir, 'net.pkl'), weights_only=True))


class BadPrediction(Prediction):
    """
    predicts the uniform policy, and a value of zero always
    """
    finite_action_space = True

    def __init__(self, num_actions, num_players, representation: Representation = None):
        """
        :param num_actions: number of possible unique actions
        :param num_players: number of players
        """
        super().__init__(representation=representation)
        self.num_actions = num_actions
        self.num_players = num_players

    def sample_actions(self, states):
        return np.random.choice(self.num_actions, len(states))

    def policy_value(self, states):
        return (torch.ones(self.num_actions)/self.num_actions,
                torch.zeros(self.num_players))


class CVAEPrediction(Prediction):
    """
    samples actions using a cvae (samples from (action | state))
    can train the CVAE from training data that is a list of (action|state)
    produces value network normally
    look at CVAE test for how the VAE parts should work
    """

    def __init__(self,
                 value_nn_config,
                 encoder_nn_config,
                 decoder_nn_config,
                 representation: Representation = None):
        """
        Args:
            value_nn_config: network that returns value (shape num_players) given a state
            encoder_nn_config: network that encodes a (action, state) pair into latent space (shaped latent_dim)
                produces a mean and log variance, so the output is a (mu, log var) tuple, each of shape (latent_dim,)
            decoder_nn_config:
                reconstructs an action given (z, state) where z is a latent variable
                    output is shaped as an action (same as first input to encoder net)

        """
        super().__init__(representation=representation)

        self.value_network = CustomNN(value_nn_config)
        self.action_encoder = CustomNN(encoder_nn_config)
        self.latent_dim = self.action_encoder.output_shape[0][0]
        self.action_decoder = CustomNN(decoder_nn_config)

    def encode_action(self, input_and_state):
        """
        encodes an (action,state)
        """
        # Result is mu and log var components
        # of the latent Gaussian distribution
        return self.action_encoder(input_and_state)

    def decode(self, z_and_state):
        """
        produces an action given (z,state)
        """
        return self.action_decoder(z_and_state)

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps*std + mu

    def forward(self, input, **kwargs):
        #TODO: where to put representation?
        _, state = input
        mu, log_var = self.encode_action(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode((z, state)), input, mu, log_var]

    def cvae_loss_function(self,
                           *args,
                           **kwargs):
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input, state = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs.get('M_N', .5)  # Account for the minibatch samples from the dataset
        recons_loss = nn.functional.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5*torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0)

        loss = recons_loss + kld_weight*kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss.detach(), 'KLD': -kld_loss.detach()}

    def sample_actions(self,
                       states,
                       current_device=None, **kwargs):
        """
        Samples from the latent space and return the corresponding data
        :param states: (n,*'), states to condition on
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        num_samples = len(states)
        if self.representation is not None:
            states=self.representation.encode(states)
        z = torch.randn(num_samples,
                        self.latent_dim)
        if current_device is not None:
            z = z.to(current_device)

        samples = self.decode((z, states))
        return samples

    def generate(self, x, **kwargs):
        """
        Given input (action, state), returns the reconstructed image
        :param x: ([B x *],[B x *'])
        :return: (Tensor) [B x *]
        """

        return self.forward(x)[0]


if __name__ == "__main__":
    import pyspiel, numpy as np

    g = pyspiel.load_game("tic_tac_toe")
    network_config = {
        'input_shape': tuple(g.observation_tensor_shape()),
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
                            "out_features": g.num_distinct_actions(),
                        },
                        {
                            "type": "softmax",
                        },
                    ],
                    [  # value head
                        {
                            "type": "linear",
                            "out_features": g.num_players(),
                        },
                    ],
                ]
            },
        ]
    }
    prediction_net = MuZeroPrediction(network_structure=network_config)
    s = g.new_initial_state()
    while not s.is_terminal():
        if s.is_chance_node():
            s.apply_action(np.random.choice(s.legal_actions()))
        else:
            obs = torch.tensor(s.observation_tensor()).reshape(g.observation_tensor_shape())
            obs = obs.unsqueeze(0)  # batch of 1
            policy, value = prediction_net.policy_value(obs)
            policy = torch.flatten(policy).detach().cpu().numpy()
            print('full policy', policy)
            policy = policy[s.legal_actions()]
            policy = policy/np.sum(policy)
            print('restricted policy', policy)
            action = np.random.choice(s.legal_actions(), p=policy)
            print('action choice', action)
            s.apply_action(action)

        print(s)
        print()
