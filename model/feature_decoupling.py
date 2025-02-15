import torch
import torch.nn as nn

class FeatureDecoupler(nn.Module):
    def __init__(self, input_dim, domain_dim, invariant_dim):
        super(FeatureDecoupler, self).__init__()
        self.domain_fc = nn.Linear(input_dim, domain_dim)
        self.invariant_fc = nn.Linear(input_dim, invariant_dim)

    def forward(self, x):
        domain_features = self.domain_fc(x)
        invariant_features = self.invariant_fc(x)
        return domain_features, invariant_features

def feature_decoupling(features, input_dim=128, domain_dim=64, invariant_dim=64):
    decoupler = FeatureDecoupler(input_dim, domain_dim, invariant_dim)
    domain_features, invariant_features = decoupler(features)
    return domain_features, invariant_features
