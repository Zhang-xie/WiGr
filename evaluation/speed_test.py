import torch
from thop import profile
from models import CNNEncoder,RelationNetwork
from config import (
    Net_factor,
    FEATURE_W,
    FEATURE_H,
    FEATURE_DIM,
)
feature_encoder = CNNEncoder()
relation_network = RelationNetwork(Net_factor)
with torch.no_grad():
    x = torch.randn(1, 3, 200, 30)
    y = feature_encoder(x)
    macs1, params1 = profile(feature_encoder, inputs=(x,))
    x = torch.randn(1, FEATURE_DIM * 2, FEATURE_H, FEATURE_W)
    y = relation_network(x)
    macs2, params2 = profile(relation_network, inputs=(x,))
    print('total size:', (macs1 + macs2) / 1000000, (params1 + params2) / 1000000)

