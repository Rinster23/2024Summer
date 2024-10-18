import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


class SDF_Model(nn.Module):
    def __init__(self, mac_feature_dim, ind_feature_dim, fcs, hid):
        super().__init__()
        self.rnn = nn.LSTM(mac_feature_dim, hid, batch_first=True)
        fc_layers = []
        for i in range(len(fcs) - 1):
            if i == 0:
                fc_layers.append(nn.Linear(hid + ind_feature_dim, fcs[0]))
            else:
                fc_layers.append(nn.Linear(fcs[i - 1], fcs[i]))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(0.1))
        fc_layers.append(nn.Linear(fcs[-2], fcs[-1]))
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, I_mac, I_ind, mask):
        N = mask.shape[1]
        I_mac_rnn, _ = self.rnn(I_mac)
        I_mac_rnn = I_mac_rnn.squeeze(0).unsqueeze(1)
        I_mac_tile = I_mac_rnn.repeat(1, N, 1)
        I_mac_masked = I_mac_tile[mask]
        I_ind_masked = I_ind[mask]
        I_concat = torch.cat(tensors=[I_mac_masked, I_ind_masked], dim=1)
        w = self.fc(I_concat)
        return w


class Con_Model(nn.Module):
    def __init__(self, mac_feature_dim, ind_feature_dim, fcs, hid):
        super().__init__()
        self.rnn = nn.LSTM(mac_feature_dim, hid, batch_first=True)
        fc_layers = []
        for i in range(len(fcs) - 1):
            if i == 0:
                fc_layers.append(nn.Linear(hid + ind_feature_dim, fcs[0]))
            else:
                fc_layers.append(nn.Linear(fcs[i - 1], fcs[i]))
            fc_layers.append(nn.ReLU())
        fc_layers.append(nn.Linear(fcs[-2], fcs[-1]))
        fc_layers.append(nn.Tanh())
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, I_mac, I_ind):
        N = I_ind.shape[1]
        I_mac_rnn, _ = self.rnn(I_mac)
        I_mac_rnn = I_mac_rnn.squeeze(0).unsqueeze(1)
        I_mac_tile = I_mac_rnn.repeat(1, N, 1)
        I_concat = torch.cat(tensors=[I_mac_tile, I_ind], dim=2)
        h = self.fc(I_concat)
        h = h.permute(2, 0, 1)
        return h


lr = 0.0002
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

macro = pd.read_pickle('macro.pkl')
individual = pd.read_pickle('individual_features.pkl')
individual.fillna(value=-99.99, inplace=True)
ret = pd.read_pickle('return.pkl')
# ret = ret.sub(ret.mean(axis=1), axis=0)
scaler = StandardScaler()

I_mac_train = macro.loc['19860101':'20101201'].to_numpy()
I_mac_train = scaler.fit_transform(I_mac_train)
I_mac_train = torch.from_numpy(I_mac_train).float().to(device)
I_ind_train = individual.loc['19860101':'20101201']
I_ind_train = torch.from_numpy(np.array([i[1].to_numpy() for i in I_ind_train.groupby(level=0)])).float().to(device)
R_train = torch.from_numpy(ret.loc['19860101':'20101201'].to_numpy()).float().to(device)
mask_train = ~R_train.isnan()

I_mac_val = macro.loc['20110101':'20141201'].to_numpy()
I_mac_val = scaler.fit_transform(I_mac_val)
I_mac_val = torch.from_numpy(I_mac_val).float().to(device)
I_ind_val = individual.loc['20110101':'20141201']
I_ind_val = torch.from_numpy(np.array([i[1].to_numpy() for i in I_ind_val.groupby(level=0)])).float().to(device)
R_val = torch.from_numpy(ret.loc['20110101':'20141201'].to_numpy()).float().to(device)
mask_val = ~R_val.isnan()

I_mac_test = macro.loc['20150101':].to_numpy()
I_mac_test = scaler.fit_transform(I_mac_test)
I_mac_test = torch.from_numpy(I_mac_test).float().to(device)
I_ind_test = individual.loc['20150101':]
I_ind_test = torch.from_numpy(np.array([i[1].to_numpy() for i in I_ind_test.groupby(level=0)])).float().to(device)
R_test = torch.from_numpy(ret.loc['20150101':].to_numpy()).float().to(device)
mask_test = ~R_test.isnan()

mac_feature_num = I_mac_train.shape[1]
ind_feature_num = I_ind_train.shape[2]


def trial(k):
    sdf_model = SDF_Model(mac_feature_num, ind_feature_num, fcs=[64, 64, 1], hid=32)
    sdf_model.to(device)
    con_model = Con_Model(mac_feature_num, ind_feature_num, fcs=[64, 8], hid=32)
    con_model.to(device)
    optimizer_SDF = torch.optim.Adam(sdf_model.parameters(), lr=lr)
    optimizer_con = torch.optim.Adam(con_model.parameters(), lr=lr)

    def train_SDF(I_mac, I_ind, mask, R, h=None):
        N_i = list(torch.sum(mask, axis=1).detach().cpu().numpy())
        T_i = torch.sum(mask, axis=0)
        I_mac = I_mac.unsqueeze(0)
        R_masked = R[mask]
        sdf_model.train()
        for _ in range(4):
            optimizer_SDF.zero_grad()
            w = sdf_model(I_mac, I_ind, mask)
            w = w.reshape(-1)
            # residual loss
            R_masked_list = torch.split(R_masked, split_size_or_sections=N_i)
            w_list = torch.split(w, split_size_or_sections=N_i)
            residual_square = 0
            R_square = 0
            for R_t, w_t in zip(R_masked_list, w_list):
                # R_t_hat = torch.sum(R_t * w_t) / torch.sum(w_t * w_t) * w_t
                R_t_hat = w_t / torch.norm(w_t, p=2)
                residual_square += torch.mean(torch.square(R_t - R_t_hat))
                R_square += torch.mean(torch.square(R_t))
            residual = residual_square / R_square
            # no arbitrage loss
            weighed_R_masked = R_masked * w
            weighed_R_split = torch.split(weighed_R_masked, split_size_or_sections=N_i)
            SDF = 1 - torch.cat([torch.sum(item, dim=0, keepdim=True) for item in weighed_R_split]).unsqueeze(1)
            R_tmp = R * mask
            R_tmp = torch.where(~R_tmp.isnan(), R_tmp, 0)
            if h is None:
                uncon_loss = torch.mean(torch.square(torch.sum(R_tmp * SDF, axis=0) / T_i) * T_i / torch.max(T_i)) * 0.1
                loss = residual + uncon_loss
            else:
                con_loss = torch.mean(
                    torch.square(torch.sum(R_tmp * SDF * h, axis=1) / T_i) * T_i / torch.max(T_i)) * 0.1
                loss = residual + con_loss
            # print(residual.item(), loss.item() - residual.item())
            loss.backward()
            optimizer_SDF.step()
        return SDF.clone().detach()

    def get_sharpe(I_mac, I_ind, mask, R, final=False):
        N_i = list(torch.sum(mask, axis=1).detach().cpu().numpy())
        sdf_model.eval()
        if final:
            sdf_model.load_state_dict(torch.load(f'sdf_model_{k}.pth'))
        w = sdf_model(I_mac, I_ind, mask)
        w = w.reshape(-1)
        w_split = torch.split(w, split_size_or_sections=N_i)
        w = torch.cat([item / torch.sum(torch.abs(item)) for item in w_split])
        R_masked = R[mask]
        weighed_R_masked = R_masked * w
        weighed_R_split = torch.split(weighed_R_masked, split_size_or_sections=N_i)
        portf = torch.cat([torch.sum(item, dim=0, keepdim=True) for item in weighed_R_split], axis=0)
        IR = torch.mean(portf) / torch.std(portf) * torch.sqrt(torch.tensor(12))
        if final:
            return IR.item(), portf.detach().cpu()
        else:
            return IR.item()

    def train_con(I_mac, I_ind, mask, R, SDF):
        T_i = torch.sum(mask, axis=0)
        I_mac = I_mac.unsqueeze(0)
        con_model.train()
        for _ in range(4):
            optimizer_con.zero_grad()
            h = con_model(I_mac, I_ind)
            R_tmp = R * mask
            R_tmp = torch.where(~R_tmp.isnan(), R_tmp, 0)
            con_loss = -torch.mean(torch.square(torch.sum(R_tmp * SDF * h, axis=1) / T_i) * T_i / torch.max(T_i))
            con_loss.backward()
            optimizer_con.step()
        return -con_loss.detach().item(), h.clone().detach()

    best_sharpe = -10
    best_con_loss = 0

    for i in range(5):
        print(f'Round {i}')
        for _ in tqdm(range(512)):
            SDF_uncon = train_SDF(I_mac_train, I_ind_train, mask_train, R_train)
            sharpe_uncon = get_sharpe(I_mac_val, I_ind_val, mask_val, R_val)
            print(f'Train Sharpe: {get_sharpe(I_mac_train, I_ind_train, mask_train, R_train)}')
            if sharpe_uncon > best_sharpe:
                best_sharpe = sharpe_uncon
                print(
                    f'Val Sharpe uncon: {sharpe_uncon}, Test Sharpe uncon: {get_sharpe(I_mac_test, I_ind_test, mask_test, R_test)}')
                SDF = SDF_uncon
                torch.save(sdf_model.state_dict(), f'sdf_model_{k}.pth')
        for _ in tqdm(range(512)):
            con_loss, h_temp = train_con(I_mac_train, I_ind_train, mask_train, R_train, SDF)
            if con_loss > best_con_loss:
                best_con_loss = con_loss
                h = h_temp
        for _ in tqdm(range(512)):
            sdf_model.load_state_dict(torch.load(f'sdf_model_{k}.pth'))
            SDF_con = train_SDF(I_mac_train, I_ind_train, mask_train, R_train, h)
            print(f'Train Sharpe: {get_sharpe(I_mac_train, I_ind_train, mask_train, R_train)}')
            sharpe_con = get_sharpe(I_mac_val, I_ind_val, mask_val, R_val)
            if sharpe_con > best_sharpe:
                print(
                    f'Val Sharpe con: {sharpe_con}, Test Sharpe con: {get_sharpe(I_mac_test, I_ind_test, mask_test, R_test)}')
                best_sharpe = sharpe_con
                torch.save(sdf_model.state_dict(), f'sdf_model_{k}.pth')

    sharpe_train, F_train = get_sharpe(I_mac_train, I_ind_train, mask_train, R_train, final=True)
    sharpe_val, F_val = get_sharpe(I_mac_val, I_ind_val, mask_val, R_val, final=True)
    sharpe_test, F_test = get_sharpe(I_mac_test, I_ind_test, mask_test, R_test, final=True)
    print('#######################################')
    print(f'Train sharpe is {sharpe_train}, val sharpe is {sharpe_val}, Test sharpe is {sharpe_test}')
    return F_train, F_val, F_test


# F_train_ensemble = []
# F_val_ensemble = []
# F_test_ensemble = []
# for i in range(10):
#     F_train, F_val, F_test = trial(i)
#     F_train_ensemble.append(F_train.unsqueeze(0))
#     F_val_ensemble.append(F_val.unsqueeze(0))
#     F_test_ensemble.append(F_test.unsqueeze(0))
# F_train_ensemble = torch.cat(F_train_ensemble, axis=0)
# F_val_ensemble = torch.cat(F_val_ensemble, axis=0)
# F_test_ensemble = torch.cat(F_test_ensemble, axis=0)
# F_train = F_train_ensemble.mean(axis=0).numpy()
# F_val = F_val_ensemble.mean(axis=0).numpy()
# F_test = F_test_ensemble.mean(axis=0).numpy()
# print('Train Sharpe ensemble:', F_train.mean() / F_train.std() * np.sqrt(12))
# print('Val Sharpe ensemble:', F_val.mean() / F_val.std() * np.sqrt(12))
# print('Test Sharpe ensemble:', F_test.mean() / F_test.std() * np.sqrt(12))
#
# np.savez('F_ensemble.npz', F_train=F_train_ensemble.numpy(), F_val=F_val_ensemble.numpy(),
#          F_test=F_test_ensemble.numpy())


class Beta_Model(nn.Module):
    def __init__(self, mac_feature_dim, ind_feature_dim, fcs):
        super().__init__()
        fc_layers = []
        for i in range(len(fcs) - 1):
            if i == 0:
                fc_layers.append(nn.Linear(mac_feature_dim + ind_feature_dim, fcs[0]))
            else:
                fc_layers.append(nn.Linear(fcs[i - 1], fcs[i]))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(0.1))
        fc_layers.append(nn.Linear(fcs[-2], fcs[-1]))
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, I_mac, I_ind, mask):
        N = mask.shape[1]
        I_mac = I_mac.unsqueeze(1)
        I_mac_tile = I_mac.repeat(1, N, 1)
        I_mac_masked = I_mac_tile[mask]
        I_ind_masked = I_ind[mask]
        I_concat = torch.cat(tensors=[I_mac_masked, I_ind_masked], dim=1)
        beta = self.fc(I_concat)
        return beta

F = np.load('F_ensemble.npz')
F_train = F['F_train']
F_val = F['F_val']
F_test = F['F_test']
R_train = torch.from_numpy(ret.loc['19860101':'20101201'].to_numpy() * F_train[:,np.newaxis]).float().to(device)
R_val = torch.from_numpy(ret.loc['20110101':'20141201'].to_numpy() * F_val[:,np.newaxis]).float().to(device)
R_test = torch.from_numpy(ret.loc['20150101':].to_numpy() * F_test[:,np.newaxis]).float().to(device)


def beta_trial(k):
    beta_model = Beta_Model(mac_feature_num, ind_feature_num, fcs=[128, 64, 1]).to(device)
    optimizer_beta = torch.optim.Adam(beta_model.parameters(), lr=lr)

    def monthly_IC_loss(x, y):
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
        z = torch.cat([x, y], axis=0)
        r = torch.corrcoef(z)[0][1]
        return 1 - r

    def train_beta(I_mac, I_ind, mask, R):
        N_i = list(torch.sum(mask, axis=1).detach().cpu().numpy())
        beta_model.train()
        for _ in range(4):
            optimizer_beta.zero_grad()
            beta = beta_model(I_mac, I_ind, mask)
            beta = beta.reshape(-1)
            R_masked = R[mask]
            beta_list = torch.split(beta, split_size_or_sections=N_i)
            R_masked_list = torch.split(R_masked, split_size_or_sections=N_i)
            R_masked_list = [(item - torch.mean(item)) / torch.std(item) for item in R_masked_list]
            loss = torch.cat(
                [monthly_IC_loss(R_masked_list[i], beta_list[i]).unsqueeze(0) for i in range(len(beta_list))]).sum()
            loss.backward()
            optimizer_beta.step()

    def get_IC(I_mac, I_ind, mask, R):
        N_i = list(torch.sum(mask, axis=1).detach().cpu().numpy())
        beta_model.eval()
        beta = beta_model(I_mac, I_ind, mask)
        beta = beta.reshape(-1)
        beta_list = torch.split(beta, split_size_or_sections=N_i)
        R_masked = R[mask]
        R_masked_list = torch.split(R_masked, split_size_or_sections=N_i)
        IC = torch.cat(
            [1 - monthly_IC_loss(R_masked_list[i], beta_list[i]).unsqueeze(0) for i in range(len(beta_list))]).mean()
        return IC.item()

    best_IC = -1
    patience = 50
    cnt = 0
    for _ in tqdm(range(2048)):
        train_beta(I_mac_train, I_ind_train, mask_train, R_train)
        IC = get_IC(I_mac_val, I_ind_val, mask_val, R_val)
        print(f'Train IC: {get_IC(I_mac_train, I_ind_train, mask_train, R_train)}')
        if IC > best_IC:
            cnt = 0
            best_IC = IC
            print(f'Val IC: {IC}')
            torch.save(beta_model.state_dict(), f'beta_model_{k}.pth')
        else:
            cnt += 1
        if cnt == patience:
            break

    beta_model.load_state_dict(torch.load(f'beta_model_{k}.pth'))
    factor = beta_model(I_mac_test, I_ind_test, mask_test).detach().cpu()
    print('Test IC:', get_IC(I_mac_test, I_ind_test, mask_test, R_test))
    return factor

factors = []

for i in range(5):
    factor = beta_trial(i)
    factors.append(factor.squeeze().unsqueeze(0))

factor = torch.cat(factors, axis=0).numpy()
np.save('factor.npy', factor.mean(axis=0))

