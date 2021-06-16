import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile, clever_format

from utils import get_nn_running_time

class_num = 6
# T = 12
T = 80
batch_size = 8


class C(nn.Module):
    def __init__(self):
        super(C, self).__init__()
        self.l_ex = nn.Sequential(
            nn.Conv2d(1, 16, 3),  #
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(7056, 64),  #
            nn.ReLU(True),
            nn.Dropout(0.1),  #
            nn.Linear(64, 64),
            nn.ReLU(True),
        )
        self.t_model = nn.GRU(input_size=64, hidden_size=128, batch_first=True)
        self.l_c = nn.Sequential(nn.Dropout(0.1), nn.Linear(128, class_num))

    def forward(self, x):
        b_dim, t_dim = x.shape[:2]
        x = self.l_ex(x.view(-1, 1, *x.shape[2:]))
        x = x.view(b_dim, t_dim, -1)
        x, _ = self.t_model(x)
        x = x[:, -1, :]
        x = self.l_c(x)
        return x


# x = torch.rand(batch_size, T, 20, 20)
# [(30, 180), (36, 125), (45, 80), (60, 45), (90, 20)]
x = torch.rand(batch_size, T, 45, 45)
y = torch.randint(0, class_num, (batch_size,))
c = C()
optim = torch.optim.SGD(c.parameters(), lr=1e-3)


def train_func():
    y_predict = c(x)
    L = F.cross_entropy(y_predict, y)
    optim.zero_grad()
    L.backward()
    optim.step()


def test_func():
    with torch.no_grad():
        y_predict = c(x)


t1 = get_nn_running_time(train_func) / batch_size
print(t1)
t2 = get_nn_running_time(test_func) / batch_size
print(t2)


macs, params = profile(c, inputs=(x,))
macs, params = clever_format([macs, params], "%.3f")
print(f"macs:{macs} params:{params}")
