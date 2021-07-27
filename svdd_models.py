import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import math
import numpy as np
import copy

import autoencoder_models


class ReccurentSVDD(nn.Module):
    def __init__(
        self,
        base_encoder: autoencoder_models.RNNEncoder,
        R: float = 0.0,
        nu: float = 0.1,
    ):
        super(ReccurentSVDD, self).__init__()

        self.base_encoder = base_encoder
        self.svdd_hidden_size = (
            self.base_encoder.hidden_size * 2
            if self.base_encoder.bi
            else self.base_encoder.hidden_size
        )
        self.bi = self.base_encoder.bi
        self.R = torch.tensor(R, device=self.device())
        self.nu = nu

    def encoded_embedding(self, inputs, input_lengths):
        _, h_n, _ = self.base_encoder.encode(inputs, input_lengths)

        return h_n.view(self.base_encoder.batch_size, -1)

    def init_center_c(self, anchor_batch, batch_lens, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        self.center = torch.zeros(self.svdd_hidden_size, device=self.device())

        with torch.no_grad():
            center = self.encoded_embedding(anchor_batch, batch_lens)
            print(center.shape)
            self.center = torch.mean(center, dim=0, keepdim=True).squeeze(0)

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        self.center[(abs(self.center) < eps) & (self.center < 0)] = -eps
        self.center[(abs(self.center) < eps) & (self.center > 0)] = eps

    def get_radius(self, dist: torch.Tensor):
        """Optimally solve for radius R via the (1-nu)-quantile of distances."""
        return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - self.nu)

    def device(self) -> torch.device:
        """Heuristic to determine which device this module is on."""
        first_param = next(self.parameters())
        return first_param.device