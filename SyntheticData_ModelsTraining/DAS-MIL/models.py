import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from nystrom_attention import NystromAttention
import numpy as np
import torch_geometric

class ftEx(nn.Module):
    def __init__(self, M, E, L):
        super(ftEx, self).__init__()

        self.M = M
        self.E = E

        self.L = L

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(1, self.M, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(self.M, self.E, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(self.E * 4 * 4, self.L),
            nn.ReLU(),
        )
        

    def forward(self, x):
        x = x.squeeze(0)

        H2 = self.feature_extractor_part1(x)
        H2 = H2.view(-1, self.E * 4 * 4)
        H2 = self.feature_extractor_part2(H2)  # NxL
        
        return H2
    
class EmbeddingTable(nn.Module):
    """Embedding table with trainable embeddings."""

    def __init__(self, dim, num_embeddings=2, trainable: bool = True):
        super().__init__()
        self.embeddings = torch.empty(num_embeddings, dim)
        if trainable:
            self.embeddings = nn.Parameter(self.embeddings)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        nn.init.normal_(self.embeddings, mean=0, std=1)

    def forward(self, weights):
        weights  # Nxnum_embeddings
        embeddings = self.embeddings  # num_embeddings x dim
        # embeddings = embeddings * torch.tensor([0, 1]).unsqueeze(-1)
        X = weights @ embeddings  # Nxdim
        return X
class ContinuousEmbeddingIndex(nn.Module):
    """Provides an embedding index for continuous values using interpolation via a sigmoid."""

    def __init__(self, num_embeddings=2):
        super().__init__()
        assert num_embeddings == 2
        self.bias = nn.Parameter(torch.empty(1))
        self.multiplier = nn.Parameter(torch.empty(1))
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        self.bias.data.fill_(.5)
        self.multiplier.data.fill_(10)  # TODO: test different initializations

    def forward(self, x):
        x  # num_edges x 1
        x = torch.sigmoid((x - self.bias) * self.multiplier)
        x = torch.cat([x, 1 - x], dim=-1)
        return x


class DiscreteEmbeddingIndex(nn.Module):
    """Provides an embedding index for discrete values using one-hot encoding."""

    def __init__(self, num_embeddings=2):
        super().__init__()
        self.num_embeddings = num_embeddings

    def forward(self, x):
        x  # num_edges x 1
        x = x * (self.num_embeddings - 1)  # assume 0 <= x <= 1
        x = torch.round(x)
        x = torch.clamp(x, 0, self.num_embeddings - 1)
        return F.one_hot(x.long().squeeze(-1), self.num_embeddings).float()

class DASMIL_end(nn.Module):
    def __init__(self, feature_size: int, hidden_dim: int, output_size: int = None, num_embeddings: int = 2, continuous: bool = True,
                 embed_keys: bool = True,
                 embed_queries: bool = True,
                 embed_values: bool = True,
                 do_term3: bool = True,
                 trainable_embeddings: bool = True):
        super().__init__()
        if output_size is None:
            output_size = feature_size
        self.keys = nn.Linear(feature_size, hidden_dim, bias=False)
        self.queries = nn.Linear(feature_size, hidden_dim, bias=False)
        self.values = nn.Linear(feature_size, output_size, bias=False)
        self.embed_keys = embed_keys
        self.embed_queries = embed_queries
        self.embed_values = embed_values
        self.do_term3 = do_term3

        if embed_keys:
            self.embed_k = EmbeddingTable(
                hidden_dim, num_embeddings=num_embeddings, trainable=trainable_embeddings)
        if embed_queries:
            self.embed_q = EmbeddingTable(
                hidden_dim, num_embeddings=num_embeddings, trainable=trainable_embeddings)
        if embed_values:
            self.embed_v = EmbeddingTable(
                output_size, num_embeddings=num_embeddings, trainable=trainable_embeddings)

        EmbeddingIndex = ContinuousEmbeddingIndex if continuous else DiscreteEmbeddingIndex
        self.index_k = EmbeddingIndex(num_embeddings=num_embeddings)
        # self.index_q = EmbeddingIndex(num_embeddings=num_embeddings)
        # self.index_v = EmbeddingIndex(num_embeddings=num_embeddings)

        self.dropout = nn.Dropout(.1)

    def forward(self, features, edge_index, edge_attr):
        edge_index = edge_index.to(torch.int64)
        N = features.shape[0]
        H = features  # NxL
        L = features.shape[-1]

        # Compute key, query, value vectors
        k = self.keys(H)  # NxD
        q = self.queries(H)  # NxD
        v = self.values(H)  # NxO

        # Compute attention scores (dot product) from classic self-attention
        A = q @ k.transpose(-2, -1)  # NxN

        # Compute additional distance-aware terms for keys/queries

        # Term 1
        if self.embed_keys:
            rk = self.embed_k(self.index_k(edge_attr)).to(torch.int64)  # num_edges x D
            Rk = torch_geometric.utils.to_dense_adj(
                edge_index, edge_attr=rk, max_num_nodes=N).squeeze(0)  # NxNxD
            q_repeat = q.unsqueeze(1).repeat(1, N, 1)  # NxNxD
            A = A + (q_repeat * Rk).sum(axis=-1)  # NxN

        # Term 2
        if self.embed_queries:
            rq = self.embed_q(self.index_k(edge_attr)).to(torch.int64)  # num_edges x D
            Rq = torch_geometric.utils.to_dense_adj(
                edge_index, edge_attr=rq, max_num_nodes=N).squeeze(0)  # NxNxD
            k_repeat = k.unsqueeze(0).repeat(N, 1, 1)  # NxNxD
            A = A + (k_repeat * Rq).sum(axis=-1)  # NxN

        # Term 3
        if self.do_term3 and self.embed_keys and self.embed_queries:
            A = A + (q_repeat * k_repeat).sum(axis=-1)  # NxN

        # Scale by sqrt(L)
        A = A / L**.5


        # Softmax over N
        A = F.softmax(A, dim=-1)  # NxN
        # Apply dropout
        A = self.dropout(A)

        # Apply attention weights to values
        M = A @ v  # NxO

        # Infuse distance-aware term in the value computation
        if self.embed_values:
            embeddings = self.index_k(edge_attr)  # num_edges x 2
            rv = self.embed_v(embeddings)  # num_edgesxO
            Rv = torch_geometric.utils.to_dense_adj(
                edge_index, edge_attr=rv, max_num_nodes=N).squeeze(0)  # NxNxO
            M = M + (A.unsqueeze(-1) * Rv).sum(axis=-2)  # NxO

        return M


class DASMIL(nn.Module):
    def __init__(self, currfeatureEx, M, E, L, D):
        super(DASMIL, self).__init__()

        self.M = M
        self.E = E
        self.L = L
        self.D = D
        self.K = 1   
                
        self.ftEx = currfeatureEx(self.M, self.E, self.L)


        self.feature_extractor_part3 = nn.Sequential(
            nn.Linear(self.L, self.L),
            nn.ReLU(),
            nn.Linear(self.L, self.L),
            nn.ReLU(),
        )

        self.dasmil = DASMIL_end(self.L, self.D, output_size = 1)
        self._fc2 = nn.Linear(10, 2)

    def forward(self, x):
        # print(H3.dtype, H2.dtype)
        H = self.ftEx(x)
        
        H = self.feature_extractor_part3(H)

        h = self.dasmil(H, torch.linspace(0, 10, steps = 10).to(H.device), torch.zeros((1, 1)).to(H.device))
        
        Y_prob = self._fc2(h.squeeze(1).unsqueeze(0)) #[B, n_classes]
        return Y_prob
    
    

class CIFAR10Dataset(Dataset):
    def __init__(self, name, data):
        """Init GenomeDataset."""
        super().__init__()
        self.name = name
        self.features = data['features']
        self.numinstances =  data['features'].shape[1]
        self.labels = data['labels']
        self.labels2 = data['labels2']
        self.length = len(self.features)

    def __len__(self):
        """Return number of sequences."""
        return self.length

    def __getitem__(self, index):
        """Return sequence and label at index."""
        return index, self.features[index].float(), self.labels[index].float(), self.labels2[index*self.numinstances:index*self.numinstances+self.features.shape[1]]