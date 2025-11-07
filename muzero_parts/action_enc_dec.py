"""
https://github.com/AntixK/PyTorch-VAE

action encoder and decoder
in standard muzero, this is not a thing (i.e. it is the identitty)
enc(state, action) produces an abstract action vector given a true state and action
dec(state, abs_action) generates a true action given a true state and abstract action
    we may cheat a little here, and condition on the TRUE state instead of abstract state.
    The motivation is that whenever we decode an action for use in the true game, we will have access to the true state.
ideally, at a given state, we should have action = dec(state, enc(state,action))
"""
import torch
from torch import nn
from networks.nn_from_config import CustomNN


class MuzeroActionEncDec(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, state, action):
        """
        encodes batch of real actions
        :param state: state or batch of real states
        :param action: action or batch of real actions
        :return: abstract action or batch of encoded abstract actions
        """
        raise NotImplementedError

    def decode(self, state, action):
        """
        decodes batch of abstract actions
        :param state: state or batch of real states
        :param action: abstract action or batch of abstract actions
        :return: action or batch of real actions
        """
        raise NotImplementedError


class IdentityActionEncDec(MuzeroActionEncDec):
    """
    for when no encoding is necessary, we are using true actions in the abstract game
    """

    def encode(self, state, action):
        return action

    def decode(self, state, action):
        return action


class VanillaVAE(nn.Module):

    def __init__(self,
                 encoder_nn_config,
                 decoder_nn_config,
                 **kwargs):
        """
        data is of shape (*), batched shape (B,*)
        encoder returns mu, log var given data shaped (B,*), both of the same shape (B,LATENT_DIM)
        decoder takes input of shape (B,LATENT_DIM), and produces an output of the original dimension (B,*)
        for unbatched, just ignore the B everywhere
        """
        super(VanillaVAE, self).__init__()

        self.encoder = CustomNN(encoder_nn_config)
        self.latent_dim = self.encoder.output_shape[0][0]
        self.decoder = CustomNN(decoder_nn_config)

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        # Result is mu and log var components
        # of the latent Gaussian distribution
        return self.encoder(input)

    def decode(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x *]
        """
        return self.decoder(z)

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
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs.get('M_N', .5)  # Account for the minibatch samples from the dataset
        recons_loss = nn.functional.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5*torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0)

        loss = recons_loss + kld_weight*kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss.detach(), 'KLD': -kld_loss.detach()}

    def sample(self,
               num_samples: int,
               current_device=None, **kwargs):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)
        if current_device is not None:
            z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x, **kwargs):
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]


if __name__ == '__main__':
    import os, ast, time
    import matplotlib.pyplot as plt

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    f = open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'networks', 'net_configs', 'test_vae_enc.txt'))
    enc_config = ast.literal_eval(f.read())
    f.close()
    f = open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'networks', 'net_configs', 'test_vae_dec.txt'))
    dec_config = ast.literal_eval(f.read())
    f.close()
    vae = VanillaVAE(encoder_nn_config=enc_config, decoder_nn_config=dec_config)
    vae.to(device)
    print(vae)
    print(vae.sample(4, current_device=device))

    a = time.time()


    def sample_distribution(n, dim=2, modes=((-2, -2), (-1, 2), (1, 1), (1, -1))):
        possible = torch.stack([torch.randn(n, dim)/10 + torch.tensor(mode).reshape(1, dim) for mode in modes])

        return possible[torch.randint(0, len(modes), (n,)), torch.arange(n), :]


    optim = torch.optim.Adam(vae.parameters(), weight_decay=.0001)
    all_losses = []

    for _ in range(1000):
        optim.zero_grad()
        batch = vae(input=sample_distribution(n=128).to(device))
        losses = vae.loss_function(*batch)
        loss = losses['loss']
        loss.backward()
        optim.step()
        all_losses.append({k: losses[k].detach().cpu().item() for k in losses})
    print(round(time.time() - a), 'seconds on device', device)
    sample = vae.sample(128, current_device=device).detach().cpu().numpy()
    plt.scatter(sample[:, 0], sample[:, 1])
    sample = sample_distribution(n=128)
    plt.scatter(sample[:, 0], sample[:, 1])
    plt.show()
    for key in all_losses[0]:
        plt.plot([losses[key] for losses in all_losses], label=key)
    plt.legend()
    plt.show()
