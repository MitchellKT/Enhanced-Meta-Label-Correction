import torch
import torch.nn as nn
from torchvision.models.feature_extraction import create_feature_extractor

class ResNetFeatures(nn.Module):

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.extractor = create_feature_extractor(self.backbone, return_nodes={'avgpool': 'features',  'fc': 'logits'})
        self.flatten = nn.Flatten()

    def forward(self, x, return_h=False):
        outs = self.extractor(x)
        logits, features = outs['logits'], outs['features']
        features = self.flatten(features)
        if return_h:
            return logits, features
        return logits

class TeacherEnhancer(nn.Module):
    def __init__(self, num_classes, embedding_dim, label_embedding_dim, hidden_dim):
        super().__init__()

        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.label_embedder = nn.Embedding(self.num_classes, label_embedding_dim)

        in_dim = embedding_dim + label_embedding_dim

        self.retain_label = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, features, y_n):
        y_emb = self.label_embedder(y_n)
        hin = torch.cat([features, y_emb], dim=-1)
        retain_logit = self.retain_label(hin)
        retain_conf = torch.sigmoid(retain_logit)
        return retain_conf