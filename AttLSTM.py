import torch
import torch.nn as nn


class AttLSTM(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.D = params[0]  # num of features
        self.E = params[1]
        self.U = params[2]  # LSTM hidden size
        self.V = params[3]

        self.seq1 = nn.Sequential(
            nn.Linear(self.D, self.E),
            nn.Tanh(),
            nn.LSTM(self.E, self.U, batch_first=True),
        )
        self.seq2 = nn.Sequential(
            nn.Linear(self.U, self.V),
            nn.Tanh(),
            nn.Linear(self.V, 1, bias=False),
            nn.Softmax(dim=1),
        )
        self.fc = nn.Linear(self.U * 2, 1)

    def forward(self, x):
        if x.dim() == 3:
            h, _ = self.seq1(x)
            h_T = h[:, -1, :]  # (N, U)
            a_t = self.seq2(h)
            a = torch.sum(h * a_t, dim=1)  # (N, U)
            e = torch.cat([a, h_T], dim=1)
            e.requires_grad_()
            y = self.fc(e)
            return y, e
        if x.dim() == 2:
            return self.fc(x)


def train(train_loader, model, eps, beta, criterion, optimizer, epochs, device):
    for i in range(epochs):
        for inp, tar in train_loader:
            inp, tar = inp.to(device, dtype=torch.float), tar.to(device, dtype=torch.float)
            optimizer.zero_grad()
            y, e = model(inp)
            y = y.squeeze()
            loss_c = criterion(y, tar)
            # .backward accumulates grad, but autograd.grad would not
            with torch.no_grad():
                r_adv = torch.autograd.grad(loss_c, e, retain_graph=True)[0]
                norms = torch.norm(r_adv, p=2, dim=1, keepdim=True)
                tmp = (norms != 0).squeeze()
                norms = norms[tmp]
                r_adv = r_adv[tmp] / norms * eps
            # e_adv need grad otherwise it only optimizes the fc layer
            e_adv = e[tmp] + r_adv
            y_adv = model(e_adv)
            y_adv = y_adv.squeeze()
            loss_adv = criterion(y_adv, tar[tmp])
            loss = loss_c + beta * loss_adv
            loss.backward()
            optimizer.step()


class HingeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        return torch.sum(torch.clamp(1 - y_pred * y_true, min=0))


# grid search
# args_list: a list of tuples
# kwargs_list: a list of dics
def parallel_jobs(n_jobs, func, args_list, kwargs_list):
    if len(args_list) == kwargs_list:
        pass
    if len(args_list) == 1:
        args_list = args_list * len(kwargs_list)
    if len(kwargs_list) == 1:
        kwargs_list = kwargs_list * len(args_list)
    res_list = []
    from joblib import Parallel, delayed
    import psutil
    res_list = Parallel(n_jobs)(delayed(func)(*arg, **kwarg) for arg, kwarg in zip(args_list, kwargs_list))
    return res_list  # ordered


def task(a, b, c, d, e):
    pass


# i,j,k --> a,b,c
parallel_jobs(3, task, args_list=[(i, j, k) for i in range(3) for j in range(4) for k in range(4)],
              kwargs_list=[{'d': 6, 'e': 8}])

