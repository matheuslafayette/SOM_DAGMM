import torch
import torch.nn as nn
import numpy as np
from utils import relative_euclidean_distance, cosine_similarity, normalize_tuple

class SOM_DAGMM(nn.Module):
    def __init__(self, dagmm, pretrained_som=None):
        super(SOM_DAGMM, self).__init__()
        self.dagmm = dagmm
        self.pretrained_som = pretrained_som

    def forward(self, input):
        if self.pretrained_som is None:
            raise ValueError("Pretrained SOM is not provided.")
        winners = [self.pretrained_som.winner(i) for i in input]
        winners = torch.tensor([normalize_tuple(winners[i], 10) for i in range(len(winners))], dtype=torch.float32)
        return self.dagmm(input, winners)

class DAGMM(nn.Module):
    def __init__(self, compression_module, estimation_module, gmm_module):
        super(DAGMM, self).__init__()
        self.compressor = compression_module
        self.estimator = estimation_module
        self.gmm = gmm_module

    def forward(self, input, winners):
        encoded = self.compressor.encode(input)
        decoded = self.compressor.decode(encoded)
        rel_euclid = relative_euclidean_distance(input, decoded)
        cos_sim = cosine_similarity(input, decoded)
        rel_euclid = rel_euclid.view(-1, 1)
        cos_sim = cos_sim.view(-1, 1)
        #cos_sim = rel_euclid.view(-1, 1)
        latent_vectors = torch.cat([encoded, rel_euclid, cos_sim, winners], dim=1)
        if self.training:
            mixtures_affiliations = self.estimator(latent_vectors)
            self.gmm._update_mixtures_parameters(latent_vectors, mixtures_affiliations)
        return self.gmm(latent_vectors)

class Mixture(nn.Module):
    def __init__(self, dimension_embedding):
        super(Mixture, self).__init__()
        self.dimension_embedding = dimension_embedding
        self.Phi = nn.Parameter(torch.from_numpy(np.random.random([1])).float(), requires_grad=False)
        self.mu = nn.Parameter(torch.from_numpy(2.*np.random.random([dimension_embedding]) - 0.5).float(), requires_grad=False)
        self.Sigma = nn.Parameter(torch.from_numpy(np.eye(dimension_embedding)).float(), requires_grad=False)
        self.eps_Sigma = torch.FloatTensor(np.diag([1.e-8 for _ in range(dimension_embedding)]))

    def forward(self, samples, with_log=True):
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

    # models.py - Mixture class
    def _update_parameters(self, samples, affiliations):
        if not self.training:
            return
        
        # Atualizar com gradientes habilitados
        with torch.no_grad():
            # Calcular novos valores
            phi = torch.mean(affiliations)
            num = torch.sum(affiliations.view(-1, 1) * samples, dim=0)
            denom = torch.sum(affiliations)
            
            new_mu = num / denom
            diff = samples - new_mu
            new_Sigma = (diff.T @ (diff * affiliations.view(-1, 1))) / denom
            
            # Atualizar com momentum
            self.Phi.data = 0.9 * self.Phi.data + 0.1 * phi
            self.mu.data = 0.9 * self.mu.data + 0.1 * new_mu
            self.Sigma.data = 0.9 * self.Sigma.data + 0.1 * new_Sigma
            
        # Adicionar ruído para estabilidade
        self.Sigma.data += torch.eye(self.dimension_embedding).to(self.Sigma.device) * 1e-6

    def gmm_loss(self, out, L1, L2):
        term1 = (L1 * torch.mean(out)) 
        epsilon = 1e-8
        diag_sigma = torch.diag(self.Sigma) + epsilon
        cov_diag = torch.sum(1 / diag_sigma)
        term2 = L2 * cov_diag

        return term1 + term2

class GMM(nn.Module):
    def __init__(self, num_mixtures, dimension_embedding):
        super(GMM, self).__init__()
        self.num_mixtures = num_mixtures
        self.dimension_embedding = dimension_embedding
        mixtures = [Mixture(dimension_embedding) for _ in range(num_mixtures)]
        self.mixtures = nn.ModuleList(mixtures)

    def forward(self, inputs):
        out = None
        for mixture in self.mixtures:
            to_add = mixture(inputs, with_log=False)
            if out is None:
                out = to_add
            else:
                out += to_add
        return -torch.log(out)

    def _update_mixtures_parameters(self, samples, mixtures_affiliations):
        if not self.training:
            return
        for i, mixture in enumerate(self.mixtures):
            affiliations = mixtures_affiliations[:, i]
            mixture._update_parameters(samples, affiliations)

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

class CompressionNetwork(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.encoder = nn.Sequential(nn.Linear(self.size, 10),
                                     nn.Tanh(),
                                     nn.Linear(10, 2))
        self.decoder = nn.Sequential(nn.Linear(2, 10),
                                     nn.Tanh(),
                                     nn.Linear(10, self.size))

        self._reconstruction_loss = nn.MSELoss()


    def forward(self, input):
        out = self.encoder(input)
        out = self.decoder(out)
        return out

    def encode(self, input):
        return self.encoder(input)

    def decode(self, input):
        return self.decoder(input)

    def reconstruction_loss(self, input):
        target_hat = self(input)
        return self._reconstruction_loss(target_hat, input)
