import torch.nn as nn
import torch

from modules.decoder import CausalMemoryDecoder
from modules.encoder import CausalEncoder
from modules.gmm import GMM
from modules.input import SourceInput


class ICAD(nn.Module):
    def __init__(self, num_regions, sequence_len=300, lambda_max=10000, 
                 num_heads=2, num_layers=4, num_gaussians=3, d_feedforward=32, d_embed=32, lambda_min=1e0):
        super().__init__()
        self.num_regions = num_regions
        self.d_model = d_embed * 4  # Four: location, arrival_time, departure_time, region_id

        # Sequence encoder
        self.input = SourceInput(num_regions, d_embed, lambda_min, lambda_max)
        self.encoder = CausalEncoder(self.d_model, num_heads, num_layers, sequence_len)

        # Region prediction
        self.region_id_decoder = CausalEncoder(self.d_model, num_heads, 1, sequence_len)
        self.region_id_head = nn.Linear(self.d_model, num_regions)

        # Arrival (travel) prediction
        self.d_travel = d_embed * 2  # Two: region_id, location
        self.travel_decoder = CausalMemoryDecoder(self.d_travel, num_heads, sequence_len, d_feedforward, kdim=self.d_model, vdim=self.d_model)
        self.travel_head = GMM(self.d_travel, num_gaussians=num_gaussians)

        # Departure prediction
        self.d_depature = d_embed * 3  # Three: region_id, location, arrival_time
        self.departure_decoder = CausalMemoryDecoder(self.d_depature, num_heads, sequence_len, d_feedforward, kdim=self.d_model, vdim=self.d_model)
        self.departure_head = GMM(self.d_depature, num_gaussians=num_gaussians)
        
    def forward(self, kwargs):
        # Encode input sequence
        memory, tgt = self.encode_sequence(kwargs)
        # Predict region id
        region_id_out = self.predict_region(memory)
        # Predict travel time with teacher forcing
        travel_out = self.predict_travel_time(memory, tgt)

        # Predict departure time with teacher forcing
        departure_out = self.predict_departure_time(memory, tgt)

        return {
            'region_id': region_id_out, 
            'travel_time': travel_out, 
            'departure_time_of_day': departure_out,
        }

    def encode_sequence(self, kwargs):
        seq = self.input(**kwargs)
        memory = self.encoder(seq[:, :-1, :])  # (batch_size, src_seq_len == seq_len - 1, d_model)
        tgt = seq[:, 1:, :]  # (batch_size, tgt_seq_len == seq_len - 1, d_model)
        return memory, tgt
            
    def predict_region(self, memory):
        """memory: (batch_size, src_seq_len, d_model)"""
        region_id_dec = self.region_id_decoder(memory)  # (batch_size, src_seq_len, d_model)
        region_id_out = self.region_id_head(region_id_dec)  # (batch_size, src_seq_len, num_regions)
        return region_id_out
    
    def predict_travel_time(self, memory, travel_tgt):
        """travel_tgt: (batch_size, tgt_seq_len, d_travel)"""
        travel_decoder_input = travel_tgt[..., :self.d_travel]
        travel_dec = self.travel_decoder(travel_decoder_input, memory)  # (batch_size, tgt_seq_len, d_model)
        travel_out = self.travel_head(travel_dec)  # (batch_size, tgt_seq_len, num_gaussians=1) for key in ['weight', 'loc','scale']        
        return travel_out
    

    
    def predict_departure_time(self, memory, departure_tgt):
        """departure_tgt: (batch_size, tgt_seq_len, d_departure)"""
        departure_decoder_input = departure_tgt[..., :self.d_depature]
        departure_dec = self.departure_decoder(departure_decoder_input, memory)  # (batch_size, seq_len-1, d_model)
        departure_out = self.departure_head(departure_dec)  # (batch_size, seq_len-1, num_gaussians=3) for key in ['weight', 'loc','scale']        
        return departure_out
