import torch
import torch.nn as nn

class Poly2Vec(nn.Module):
    """
    Implementation of Poly2Vec for point encoding based on https://github.com/USC-InfoLab/poly2vec.
    """

    def __init__(self, embedding_dim, f_min = 0.1, f_max = 10, n_freqs = 10, device='cuda'):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.device = device

        # create meshgrid for sampling frequencies
        self.U, self.V, self.fourier_dim = self.create_gmf_meshgrid(n_freqs, f_min, f_max)
        self.U = self.U[None, :, :].to(device)
        self.V = self.V[None, :, :].to(device)
        
        self.location_embedding = nn.Sequential(
            nn.Linear(2 * self.fourier_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim),
            nn.Dropout(0.3),
        )
        
        self.nn = nn.Sequential(
            nn.Linear(self.fourier_dim, 2*self.fourier_dim),
            nn.ReLU(),
            nn.Linear(self.fourier_dim*2, self.fourier_dim),
            nn.Dropout(0.3)
        )
        
        self.param_phase = nn.Sequential(
            nn.Linear(self.fourier_dim, self.fourier_dim),
            self.nn
        )
        self.param_magnitude = nn.Sequential(
            nn.Linear(self.fourier_dim, self.fourier_dim),
            self.nn
        )
    
    def create_gmf_meshgrid(self, n_freqs, f_min, f_max):
        """Create a geometric meshgrid for frequency sampling."""
        g = (f_max / f_min)**(1/(n_freqs - 1))
        positive_wu = torch.tensor([f_min * g**u for u in range(n_freqs)], dtype=torch.float32)

        if (2 * n_freqs + 1) % 2 == 1:
            Wx = torch.cat((-torch.flip(positive_wu, dims=[0]), torch.tensor([0]), positive_wu))
        else:
            Wx = torch.cat((-torch.flip(positive_wu[:-1], dims=[0]), torch.tensor([0]), positive_wu))

        if n_freqs % 2 == 1:
            Wy = torch.cat((torch.tensor([0]), positive_wu))
        else:
            Wy = positive_wu

        U, V = torch.meshgrid(Wx, Wy, indexing='ij')
        return U, V, Wx.shape[0] * Wy.shape[0]

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        """
        Args:
            p: (B, T, 2) - input coordinates, batched sequences
        Returns:
            output: (B, T, embedding_dim)
        """
        B, T, _ = p.shape
        p_x = p[:, :, 0].unsqueeze(-1).unsqueeze(-1) # (B, T, 1, 1)
        p_y = p[:, :, 1].unsqueeze(-1).unsqueeze(-1)

        # Compute complex Fourier encoding
        loc_enc = torch.exp(-2j * torch.pi * (self.U * p_x + self.V * p_y)).reshape(B, T, -1)

        # Apply magnitude and phase encoders
        mag = self.param_magnitude(torch.abs(loc_enc))
        phase = self.param_phase(torch.angle(loc_enc))

        # Concatenate and flatten
        loc_enc = torch.cat((mag, phase), dim=-1)
        loc_enc = loc_enc.reshape(B, T, -1)

        return self.location_embedding(loc_enc) # (B, T, embedding_dim)