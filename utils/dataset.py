import torch
from torch.utils.data import Dataset
from torch.nn.functional import pad
from utils.constants import PAD, FIELDS


def construct_batched_visit_collate_fn(batch):
    max_visit_len = max([item.shape[0] for item in batch])
    visit_sequences = []
    for visit_sequence in batch:
        L, D = visit_sequence.shape
        pad_len = max_visit_len - L
        if pad_len > 0:
            # Pad the sequence to the maximum length to the left of the sequence
            visit_sequence = pad(visit_sequence, (0, 0, pad_len, 0), mode='constant', value=PAD)  # (L+pad_len, D)
        visit_sequences.append(visit_sequence)

    visit_sequences = torch.stack(visit_sequences, dim=0)  # (B, L, D)
    return visit_sequences
    
class ICADDataset(Dataset):
    def __init__(self, df, indices, include_anomalies=False):
        self.df = df[[col for col in FIELDS if col in df.columns]]
        self.indices = indices.astype(int)
        self.include_anomalies = include_anomalies

    def __len__(self):
        return len(self.indices)
        
    def __getitem__(self, idx):
        first_index, last_index = self.indices.loc[idx].values
        fields = [field for field in FIELDS if field in self.df.columns]
        instance = self.df.loc[first_index:last_index][fields]

        instance = instance.astype({col: 'int' for col in instance.select_dtypes(include=['bool']).columns}) # for boolean column "anomaly"
        instance = torch.from_numpy(instance.values).float()

        return instance
