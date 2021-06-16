import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain

from thop import profile, clever_format

from utils import get_nn_running_time

batch_size = 8
class_num = 6
# _in_c = 1
_in_c = 18


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(_in_c, 32, 11),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.f(x)


class Classifer(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(3008, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 6),
        )

    def forward(self, x):
        return self.f(x)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(3008, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1),
        )

    def forward(self, x):
        # x = torch.cat((x1, x2), dim=1)
        return torch.sigmoid(self.f(x))


def loss2_d(
    y_source,
    y_target,
):
    return (-torch.log(y_source) - torch.log(1 - y_target)).mean()


def loss2_e_s(
    y_source,
    y_target,
):
    return (-torch.log(y_source)).mean()


def loss2_e_t(
    y_source,
    y_target,
):
    return (-torch.log(y_target)).mean()


if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    e_source = Encoder()
    c_source = Classifer()
    e_target = Encoder()
    c_target = Classifer()
    d = Discriminator()
    c_shared = Classifer()

    optim1 = torch.optim.SGD(
        chain(e_source.parameters(), c_source.parameters()), lr=1e-3
    )
    optim2_e = torch.optim.SGD(
        chain(e_source.parameters(), e_target.parameters()), lr=1e-3
    )
    optim2_d = torch.optim.SGD(d.parameters(), lr=1e-3)
    optim3 = torch.optim.SGD(c_shared.parameters(), lr=1e-3)

    # x_source = torch.randn(batch_size, _in_c, 400, 114)
    x_source = torch.randn(batch_size, _in_c, 300, 30)
    y_source = torch.randint(0, class_num, (batch_size,))

    x_target = torch.randn_like(x_source)
    y_target = torch.randint(0, class_num, (batch_size,))
    y_target_domain = torch.ones_like(y_source)
    y_source_domain = torch.zeros_like(y_source)

    # STEP1
    def step1():
        y_source_predict = c_source(e_source(x_source))
        L = F.cross_entropy(y_source_predict, y_source)
        optim1.zero_grad()
        L.backward()
        optim1.step()

    # STEP2
    def step2():
        y_source_predict = d(e_source(x_source))
        y_target_predict = d(e_target(x_target))
        L2_d = loss2_d(y_source_predict, y_target_predict)
        optim2_d.zero_grad()
        L2_d.backward()
        optim2_d.step()

        y_source_predict = d(e_source(x_source))
        y_target_predict = d(e_target(x_target))
        L2_e = loss2_e_s(y_source_predict, y_target_predict) + loss2_e_t(
            y_source_predict, y_target_predict
        )
        optim2_e.zero_grad()
        L2_e.backward()
        optim2_e.step()

    # STEP3
    def step3():
        with torch.no_grad():
            temp = e_source(x_source)
        y_source_predict = c_shared(temp)
        L = F.cross_entropy(y_source_predict, y_source)
        optim3.zero_grad()
        L.backward()
        optim3.step()

    # STEP4(TEST)
    def step4():
        with torch.no_grad():
            y_source_predict = c_shared(e_target(x_target))
        acc = (y_source_predict.argmax(1) == y_source).float().mean().item()

    t1 = get_nn_running_time(step1) / batch_size
    t2 = get_nn_running_time(step2) / batch_size
    t3 = get_nn_running_time(step3) / batch_size
    t4 = get_nn_running_time(step4) / batch_size
    print(t1, t2, t3, t4)

    z0 = e_source(x_source)
    macs, params = profile(e_source, inputs=(x_source,))
    macs, params = clever_format([macs, params], "%.3f")
    print(f"macs:{macs} params:{params}")
    macs, params = profile(c_source, inputs=(z0,))
    macs, params = clever_format([macs, params], "%.3f")
    print(f"macs:{macs} params:{params}")
    macs, params = profile(d, inputs=(z0,))
    macs, params = clever_format([macs, params], "%.3f")
    print(f"macs:{macs} params:{params}")
