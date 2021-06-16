import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
from utils import get_nn_running_time

from thop import profile, clever_format

fe_in_c = 180
ar_in_c = 512


class_num = 6
domain_num = 5
batch_size = 80
dd_in_c = ar_in_c + class_num


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        in_c = fe_in_c
        self.f = nn.Sequential(
            nn.Conv2d(in_c, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.f(x)


class ActivityRecognizer(nn.Module):
    def __init__(self):
        super().__init__()
        in_c = ar_in_c
        out_c = class_num
        self.f = nn.Sequential(nn.Linear(in_c, 32), nn.Softplus(), nn.Linear(32, out_c))

    def forward(self, x):
        return self.f(x)


class DomainDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        in_c = dd_in_c
        out_c = domain_num
        self.f = nn.Sequential(
            nn.Linear(in_c, 32),
            nn.Softplus(),
            nn.Linear(32, out_c),
            nn.Softplus(),
        )

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        return self.f(x)


def naive_cross_entropy(y_pred, y_true):
    y_pred = F.softmax(y_pred, dim=1)
    y_true = F.softmax(y_true, dim=1)
    y_pred = torch.clamp(y_pred, 1e-9, 1 - 1e-9)
    return -(y_true * torch.log(y_pred)).sum(dim=1).mean()


def js_div(x1, x2):
    x1 = F.softmax(x1, dim=1)
    x2 = F.softmax(x2, dim=1)
    return 0.5 * (
        F.kl_div(((x1 + x2) / 2).log(), x1, reduction="batchmean")
        + F.kl_div(((x1 + x2) / 2).log(), x2, reduction="batchmean")
    )


def lc(y):
    y = F.softmax(y, dim=1)
    return -(torch.log(y) + torch.log(1 - y)).sum(dim=1).mean()


fe = FeatureExtractor()
ar = ActivityRecognizer()
dd = DomainDiscriminator()


optim = torch.optim.SGD(
    chain(fe.parameters(), ar.parameters(), dd.parameters()), lr=1e-3
)

x_labeled = torch.randn(batch_size, fe_in_c, 30, 30)
y_labeled = F.one_hot(torch.randint(0, class_num, (batch_size,)), class_num).float()
y_labeled_domain = F.one_hot(
    torch.randint(0, domain_num, (batch_size,)), domain_num
).float()
x_unlabeled = torch.randn(batch_size, fe_in_c, 30, 30)
y_unlabeled_domain = F.one_hot(
    torch.randint(0, domain_num, (batch_size,)), domain_num
).float()

x_labeled_addnoise = x_labeled + torch.randn_like(x_labeled)
x_unlabeled_addnoise = x_unlabeled + torch.randn_like(x_unlabeled)


def func_train():

    z_labeled = fe(x_labeled)
    y_labeled_predict = ar(z_labeled)
    s_labeled = dd(z_labeled, y_labeled_predict)

    z_unlabeled = fe(x_unlabeled)
    y_unlabeled_predict = ar(z_unlabeled)
    s_unlabeled = dd(z_unlabeled, y_unlabeled_predict)

    y_labeled_predict_pnoise = ar(fe(x_labeled_addnoise))

    y_unlabeled_predict_pnoise = ar(fe(x_unlabeled_addnoise))

    L_a = naive_cross_entropy(y_labeled_predict, y_labeled)
    L_u = naive_cross_entropy(y_unlabeled_predict, y_unlabeled_predict)
    L_d = naive_cross_entropy(
        torch.cat([s_labeled, s_unlabeled], dim=0),
        torch.cat([y_labeled_domain, y_unlabeled_domain], dim=0),
    )
    L_c = 0.5 * (lc(y_labeled_predict) + lc(y_unlabeled_predict))
    L_s = 0.5 * (
        js_div(y_labeled_predict_pnoise, y_labeled_predict)
        + js_div(y_unlabeled_predict_pnoise, y_unlabeled_predict)
    )
    # Pc = F.softmax(torch.rand(class_num), dim=0)

    L = L_a + L_u + L_d + L_c + L_s

    optim.zero_grad()
    L.backward()
    optim.step()


def func_test():
    with torch.no_grad():
        y_labeled_predict = ar(fe(x_labeled))


t1 = get_nn_running_time(func_train) / batch_size
print(t1)
t2 = get_nn_running_time(func_test) / batch_size
print(t2)


z_labeled = fe(x_labeled)
y_labeled_predict = ar(z_labeled)
s_labeled = dd(z_labeled, y_labeled_predict)

macs, params = profile(fe, inputs=(x_labeled,))
macs, params = clever_format([macs, params], "%.3f")
print(f"macs:{macs} params:{params}")
macs, params = profile(ar, inputs=(z_labeled,))
macs, params = clever_format([macs, params], "%.3f")
print(f"macs:{macs} params:{params}")
macs, params = profile(dd, inputs=(z_labeled, y_labeled_predict))
macs, params = clever_format([macs, params], "%.3f")
print(f"macs:{macs} params:{params}")
