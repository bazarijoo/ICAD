import torch
import torch.nn as nn

from modules.poly2vec import Poly2Vec
from modules.time2vec import Time2Vec
from utils.constants import  PAD


class SourceInput(nn.Module):
    def __init__(self, num_regions, d_embed, lambda_min, lambda_max):
        """
        Args:
            num_regions (int): The number of tokens in the vocabulary.
            d_embed (int): The dimension of the embeddings.
            lambda_min (float): The minimum scale for space2vec if used.
            lambda_max (float): The maximum scale for space2vec if used.
        """
        super().__init__()
        self.poly2vec = Poly2Vec(d_embed)
        self.time2vec = Time2Vec(d_embed)
        self.region_embedding = nn.Embedding(num_regions, d_embed, padding_idx=PAD)

    def forward(self, region_id, latitude, longitude, arrival_time, departure_time):
        """
        Each input variable is a tensor of size (batch_size, seq_len)
        """
        locations = torch.stack([latitude, longitude], dim=-1)
        location_encoding = self.poly2vec(locations)
        arrival_encoding = self.time2vec(arrival_time)
        departure_encoding = self.time2vec(departure_time)
        region_embedding = self.region_embedding(region_id.long())
        visit_embedding = torch.concat([
            location_encoding,
            region_embedding,
            arrival_encoding,
            departure_encoding
        ], dim=-1)  # (batch_size, seq_len, d_embed*4)
        return visit_embedding
