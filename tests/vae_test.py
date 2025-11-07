"""
https://github.com/AntixK/PyTorch-VAE
"""
import torch
from torch import nn

from networks.nn_from_config import CustomNN


class VAE(nn.Module):

    def __init__(self,
                 encoder_nn_config,
                 decoder_nn_config, ):
        """
        data is of shape (*), batched shape (B,*)
        encoder returns mu, log var given data shaped (B,*), both of the same shape (B,LATENT_DIM)
        decoder takes input of shape (B,LATENT_DIM), and produces an output of the original dimension (B,*)
        for unbatched, just ignore the B everywhere
        """
        super(VAE, self).__init__()

        self.encoder = CustomNN(encoder_nn_config)
        self.latent_dim = self.encoder.output_shape[0][0]
        self.decoder = CustomNN(decoder_nn_config)

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [B x *]
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
        :param x: (Tensor) [B x *]
        :return: (Tensor) [B x *]
        """

        return self.forward(x)[0]


class CVAE(nn.Module):

    def __init__(self,
                 encoder_nn_config,
                 decoder_nn_config,
                 ):
        """
        Same, except everyhting is conditioned on a continuous state s of shape *'
        data is of shape ((*),(*')), batched shape ((B,*),(B,*'))
        encoder returns mu, log var given data shaped ((B,*),(B,*')), both of the same shape (B,LATENT_DIM)
        decoder takes input of shape ((B,LATENT_DIM),(B,*')), and produces an output of the original dimension (B,*)
        for unbatched, just ignore the B everywhere

        ideally, for state s, data a, we have that for mu,var ~ encode(a,s), decode((z~(mu,var)),s) is close to a
        """
        super(CVAE, self).__init__()

        self.encoder = CustomNN(encoder_nn_config)
        self.latent_dim = self.encoder.output_shape[0][0]
        self.decoder = CustomNN(decoder_nn_config)

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: ([B x *], [B x *'])
        :return: (Tensor) List of latent codes
        """
        # Result is mu and log var components
        # of the latent Gaussian distribution
        return self.encoder(input)

    def decode(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: ([B x D], [B x *'])
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
        _, state = input
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode((z, state)), input, mu, log_var]

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
        input, state = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs.get('M_N', .5)  # Account for the minibatch samples from the dataset
        recons_loss = nn.functional.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5*torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0)

        loss = recons_loss + kld_weight*kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss.detach(), 'KLD': -kld_loss.detach()}

    def sample(self,
               num_samples: int,
               states,
               current_device=None, **kwargs):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param states: (n,*'), states to condition on
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)
        if current_device is not None:
            z = z.to(current_device)

        samples = self.decode((z, states))
        return samples

    def generate(self, x, **kwargs):
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x *]
        :return: (Tensor) [B x *]
        """

        return self.forward(x)[0]


if __name__ == '__main__':
    import os, ast, time
    import matplotlib.pyplot as plt

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    f = open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'networks', 'net_configs', 'test_cvae_enc.txt'))
    enc_config = ast.literal_eval(f.read())
    f.close()
    f = open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'networks', 'net_configs', 'test_cvae_dec.txt'))
    dec_config = ast.literal_eval(f.read())
    f.close()
    cvae = CVAE(encoder_nn_config=enc_config, decoder_nn_config=dec_config)
    cvae.to(device)
    print(cvae)
    print(cvae.sample(5, current_device=device, states=torch.rand(5, 4)))

    a = time.time()


    def sample_distribution(n, dim=2, modes=((-2, -2), (-1, 2), (1, 1), (1, -1)), device=None):
        possible = torch.stack([torch.randn(n, dim)/10 + torch.tensor(mode).reshape(1, dim) for mode in modes])
        dists = torch.randint(0, len(modes), (n,))
        hot = torch.zeros(n, len(modes))
        hot[torch.arange(n), dists] = 1
        hot = hot + torch.randn(hot.shape)/5
        return possible[dists, torch.arange(n), :].to(device), hot.to(device)


    optim = torch.optim.Adam(cvae.parameters(), weight_decay=.0001)
    all_losses = []

    for _ in range(1000):
        optim.zero_grad()
        batch = cvae(input=sample_distribution(n=128, device=device))
        losses = cvae.loss_function(*batch)
        loss = losses['loss']
        loss.backward()
        optim.step()
        all_losses.append({k: losses[k].detach().cpu().item() for k in losses})
    print(round(time.time() - a), 'seconds on device', device)
    true_sample, sampled_states = sample_distribution(n=128)
    sample = cvae.sample(128, current_device=device, states=sampled_states).detach().cpu().numpy()
    plt.scatter(sample[:, 0], sample[:, 1], label='sampled from distribution')
    plt.scatter(true_sample[:, 0], true_sample[:, 1], label='true distribution')
    reconstructed = cvae.forward((true_sample, sampled_states))[0].detach().cpu().numpy()
    plt.scatter(reconstructed[:, 0], reconstructed[:, 1], label='reconstructed')
    plt.legend()
    plt.show()

    resids = true_sample - reconstructed
    plt.scatter(resids[:, 0], resids[:, 1])
    plt.title('residuals')
    plt.show()
    for key in all_losses[0]:
        plt.plot([losses[key] for losses in all_losses], label=key)
    plt.legend()
    plt.show()

    f = open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'networks', 'net_configs', 'test_vae_enc.txt'))
    enc_config = ast.literal_eval(f.read())
    f.close()
    f = open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'networks', 'net_configs', 'test_vae_dec.txt'))
    dec_config = ast.literal_eval(f.read())
    f.close()
    vae = VAE(encoder_nn_config=enc_config, decoder_nn_config=dec_config)
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
