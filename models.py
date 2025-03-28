import torch
import torch.nn as nn
import numpy as np
from utils import relative_euclidean_distance, cosine_similarity, normalize_tuple

# ------------------------------------------------------------------------------
# Core Model: Combines a pre-trained SOM with a DAGMM for anomaly detection.
# ------------------------------------------------------------------------------
class SOM_DAGMM(nn.Module):
    def __init__(self, dagmm, pretrained_som=None):
        """
        Args:
            dagmm: An initialized DAGMM model.
            pretrained_som: A pre-trained Self-Organizing Map.
        """
        super(SOM_DAGMM, self).__init__()
        self.dagmm = dagmm
        self.pretrained_som = pretrained_som

    def forward(self, input):
        # Retrieve the winning node for each input sample from the SOM.
        if self.pretrained_som is None:
            raise ValueError("Pretrained SOM is not provided.")
        winners = [self.pretrained_som.winner(i) for i in input]
        winners = torch.tensor([normalize_tuple(winners[i], 10) for i in range(len(winners))],
                                 dtype=torch.float32)
        # Combine SOM spatial information with DAGMM's density estimation.
        return self.dagmm(input, winners)

# ------------------------------------------------------------------------------
# DAGMM: Deep Autoencoding Gaussian Mixture Model.
# Components:
#   1. Compression: Autoencoder for dimensionality reduction.
#   2. Estimation: Network to estimate mixture component affiliations.
#   3. GMM: Gaussian Mixture Model layer.
# ------------------------------------------------------------------------------
class DAGMM(nn.Module):
    def __init__(self, compression_module, estimation_module, gmm_module):
        super(DAGMM, self).__init__()
        self.compressor = compression_module
        self.estimator = estimation_module
        self.gmm = gmm_module

    def forward(self, input, winners):
        # Encode input and reconstruct it to obtain reconstruction errors.
        encoded = self.compressor.encode(input)
        decoded = self.compressor.decode(encoded)
        rel_euclid = relative_euclidean_distance(input, decoded).view(-1, 1)
        cos_sim = cosine_similarity(input, decoded).view(-1, 1)
        # Concatenate latent representation with error features and SOM winners.
        latent_vectors = torch.cat([encoded, rel_euclid, cos_sim, winners], dim=1)
        # Update GMM parameters with an EM-style step during training.
        if self.training:
            mixtures_affiliations = self.estimator(latent_vectors)
            self.gmm._update_mixtures_parameters(latent_vectors, mixtures_affiliations)
        return self.gmm(latent_vectors)

# ------------------------------------------------------------------------------
# Mixture: A single Gaussian mixture component with EM-style parameter updates.
# ------------------------------------------------------------------------------
class Mixture(nn.Module):
    def __init__(self, dimension_embedding):
        super(Mixture, self).__init__()
        self.dimension_embedding = dimension_embedding
        self.Phi = nn.Parameter(torch.from_numpy(np.random.random([1])).float(), requires_grad=False)
        self.mu = nn.Parameter(torch.from_numpy(2. * np.random.random([dimension_embedding]) - 0.5).float(), requires_grad=False)
        self.Sigma = nn.Parameter(torch.from_numpy(np.eye(dimension_embedding)).float(), requires_grad=False)
        self.eps_Sigma = torch.FloatTensor(np.diag([1.e-8 for _ in range(dimension_embedding)]))

    def forward(self, samples, with_log=True):
        # Compute probability densities for samples under this mixture component.
        batch_size, _ = samples.shape
        out_values = []
        inv_sigma = torch.pinverse(self.Sigma)
        det_sigma = np.linalg.det(self.Sigma.data.cpu().numpy())
        det_sigma = torch.from_numpy(det_sigma.reshape([1])).float()
        det_sigma = torch.autograd.Variable(det_sigma)
        for sample in samples:
            diff = (sample - self.mu).view(-1, 1)
            out = -0.5 * torch.mm(torch.mm(diff.view(1, -1), inv_sigma), diff)
            out = (self.Phi * torch.exp(out)) / (torch.sqrt(2. * np.pi * det_sigma))
            if with_log:
                out = -torch.log(out + 1e-8)
            out_values.append(float(out.data.cpu().numpy()))
        out = torch.autograd.Variable(torch.FloatTensor(out_values))
        return torch.FloatTensor(out)

    def _update_parameters(self, samples, affiliations):
        # Update mixture parameters using an exponential moving average (EMA).
        if not self.training:
            return
        
        with torch.no_grad():
            phi = torch.mean(affiliations)
            num = torch.sum(affiliations.view(-1, 1) * samples, dim=0)
            denom = torch.sum(affiliations)
            
            new_mu = num / denom
            diff = samples - new_mu
            new_Sigma = (diff.T @ (diff * affiliations.view(-1, 1))) / denom
            
            self.Phi.data = 0.9 * self.Phi.data + 0.1 * phi
            self.mu.data = 0.9 * self.mu.data + 0.1 * new_mu
            self.Sigma.data = 0.9 * self.Sigma.data + 0.1 * new_Sigma
            
        # Regularize covariance to ensure numerical stability.
        self.Sigma.data += torch.eye(self.dimension_embedding).to(self.Sigma.device) * 1e-6

    def gmm_loss(self, out, L1, L2):
        # Compute regularization loss on the GMM parameters.
        term1 = L1 * torch.mean(out)
        epsilon = 1e-8
        diag_sigma = torch.diag(self.Sigma) + epsilon
        cov_diag = torch.sum(1 / diag_sigma)
        term2 = L2 * cov_diag
        return term1 + term2

# ------------------------------------------------------------------------------
# GMM: Aggregates multiple mixture components.
# ------------------------------------------------------------------------------
class GMM(nn.Module):
    def __init__(self, num_mixtures, dimension_embedding):
        super(GMM, self).__init__()
        self.num_mixtures = num_mixtures
        self.dimension_embedding = dimension_embedding
        mixtures = [Mixture(dimension_embedding) for _ in range(num_mixtures)]
        self.mixtures = nn.ModuleList(mixtures)

    def forward(self, inputs):
        # Sum the contributions from all mixture components.
        out = None
        for mixture in self.mixtures:
            to_add = mixture(inputs, with_log=False)
            if out is None:
                out = to_add
            else:
                out += to_add
        return -torch.log(out)

    def _update_mixtures_parameters(self, samples, mixtures_affiliations):
        # Update parameters for each mixture component.
        if not self.training:
            return
        for i, mixture in enumerate(self.mixtures):
            affiliations = mixtures_affiliations[:, i]
            mixture._update_parameters(samples, affiliations)

# ------------------------------------------------------------------------------
# Estimation Network: Estimates mixture component memberships.
# ------------------------------------------------------------------------------
class EstimationNetwork(nn.Module):
    def __init__(self):
        super(EstimationNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 10),
            nn.Tanh(),
            nn.Dropout(p=0.5),
            nn.Linear(10, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, input):
        return self.net(input)

# ------------------------------------------------------------------------------
# Compression Network: Autoencoder for reconstruction and latent space formation.
# ------------------------------------------------------------------------------
class CompressionNetwork(nn.Module):
    def __init__(self, size):
        """
        Args:
            size: Number of input features.
        """
        super().__init__()
        self.size = size
        self.encoder = nn.Sequential(
            nn.Linear(self.size, 10),
            nn.Tanh(),
            nn.Linear(10, 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 10),
            nn.Tanh(),
            nn.Linear(10, self.size)
        )
        self._reconstruction_loss = nn.MSELoss()

    def forward(self, input):
        # Reconstruct input through autoencoder.
        out = self.encoder(input)
        out = self.decoder(out)
        return out

    def encode(self, input):
        return self.encoder(input)

    def decode(self, input):
        return self.decoder(input)

    def reconstruction_loss(self, input):
        # Compute the mean squared error between input and its reconstruction.
        target_hat = self(input)
        return self._reconstruction_loss(target_hat, input)
