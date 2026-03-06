# model.py
import torch
import torch.nn as nn

from kan import KAN


class SEFusion(nn.Module):
    def __init__(self, num_channels: int, reduction_ratio: int = 2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(num_channels, num_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(num_channels // reduction_ratio, num_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C(=3), D]
        squeeze_tensor = x.mean(dim=2)               # [B, C]
        excitation_tensor = self.fc(squeeze_tensor)  # [B, C]
        excitation_tensor = excitation_tensor.unsqueeze(2)  # [B, C, 1]
        return x * excitation_tensor.expand_as(x)


class MultiBranchFusionModel(nn.Module):
    def __init__(
        self,
        t5_dim: int = 1024,
        pssm_dim: int = 40,
        nmbac_dim: int = 200,
        hidden_dim: int = 256,
        dropout_rate: float = 0.3,
        kan_grid_size: int = 6,
        kan_spline_order: int = 4,
    ):
        super().__init__()

        # PSSM 分支
        self.pssm_branch = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, hidden_dim),
        )

        # ProtT5 分支
        self.t5_branch = nn.Sequential(
            nn.Linear(t5_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, hidden_dim),
        )

        # NMBAC 分支
        self.nmbac_branch = nn.Sequential(
            nn.Linear(nmbac_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, hidden_dim),
        )

        # SE 融合（把三路表示做 gating 再求和）
        self.fusion = SEFusion(num_channels=3)

        # KAN 分类头（保持你原始设置）
        self.classifier = KAN(
            layers_hidden=[hidden_dim, 128, 1],
            grid_size=kan_grid_size,
            spline_order=kan_spline_order,
        )

    def forward(self, t5: torch.Tensor, pssm: torch.Tensor, nmbac: torch.Tensor) -> torch.Tensor:
        pssm_out = self.pssm_branch(pssm.unsqueeze(1))  # [B, hidden]
        t5_out = self.t5_branch(t5)                     # [B, hidden]
        nmbac_out = self.nmbac_branch(nmbac)            # [B, hidden]

        fused = self.fusion(torch.stack([pssm_out, t5_out, nmbac_out], dim=1))  # [B, 3, hidden]
        return self.classifier(fused.sum(dim=1))  # [B, 1]
