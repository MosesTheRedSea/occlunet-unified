import torch
from torch import nn
from src.temporal import Temporal
from src.spectral import Spectral
from src.cross_attention import CrossBranchAttention

class OcculNetV2(nn.Module):
    def __init__(self, num_det_classes, num_mat_classes):
        super().__init__()

        self.temporal_branch = Temporal()
        self.spectral_branch = Spectral()

        self.cross_attn = CrossBranchAttention(dim=128, num_heads=8)
 
        # Project each branch to its role
        self.semantic_proj = nn.Sequential(   # spectral  -> detection + material
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.geometric_proj = nn.Sequential(  # temporal  -> distance
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # Shared semantic features
        self.detection_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_det_classes),
            nn.LogSoftmax(dim=1)
        )

        self.material_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_mat_classes),
            nn.LogSoftmax(dim=1)
        )

        # Private geometric features
        self.distance_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, ir_1d, spec_2d):

        t_feat = self.temporal_branch(ir_1d)    # (B, 128, L)
        s_feat = self.spectral_branch(spec_2d)  # (B, 128, H, W)
 
        t_attn, s_attn = self.cross_attn(t_feat, s_feat)
        # t_attn: (B, L,  128)
        # s_attn: (B, HW, 128)
 
        t_embed = t_attn.mean(dim=1)   # (B, 128)  — geometric
        s_embed = s_attn.mean(dim=1)   # (B, 128)  — semantic
 
        # Route by physics
        semantic  = self.semantic_proj(s_embed)   # what / material
        geometric = self.geometric_proj(t_embed)  # how far
 
        # Store for optional orthogonality loss in trainer
        self._semantic  = semantic
        self._geometric = geometric
 
        det_out  = self.detection_head(semantic)
        mat_out  = self.material_head(semantic)
        dist_out = self.distance_head(geometric)
 
        return det_out, dist_out, mat_out

        # t_feat = self.temporal_branch(ir_1d)
        # s_feat = self.spectral_branch(spec_2d)

        # # t_feat : (B,128,L)
        # # s_feat : (B,128,H,W)

        # t_attn, s_attn = self.cross_attn(t_feat, s_feat)

        # # t_attn -> (B,L,128)
        # # s_attn -> (B,HW,128)

        # t_embed = t_attn.mean(dim=1)
        # s_embed = s_attn.mean(dim=1)

        # # Fusion
        # fused = torch.cat([t_embed, s_embed], dim=1)
        # latent = self.fusion(fused)

        # # Disentanglement
        # shared_feat = latent[:, :128]
        # private_feat = latent[:, 128:]

        # # Task heads
        # det_out = self.detection_head(
        #     shared_feat
        # )

        # mat_out = self.material_head(
        #     shared_feat
        # )

        # dist_out = self.distance_head(
        #     private_feat
        # )

        # return (
        #     det_out,
        #     dist_out,
        #     mat_out
        # )

    @property
    def orthogonality_loss(self):
        """
        Soft penalty encouraging semantic and geometric embeddings
        to be decorrelated.  Add lambda * model.orthogonality_loss
        to your total loss during training.
        """
        if not hasattr(self, '_semantic'):
            return torch.tensor(0.0)
        # Normalise, compute cosine similarity matrix, penalise off-diagonal
        s = nn.functional.normalize(self._semantic,  dim=1)
        g = nn.functional.normalize(self._geometric, dim=1)
        # dot product between each sample's semantic and geometric vector
        cos_sim = (s * g).sum(dim=1)   # (B,)
        return cos_sim.pow(2).mean()
 