import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.distributions as D

torch.manual_seed(1)



class LstmEncDeltaStackedFullPredMultyGaus(nn.Module):
    """
    model that actually predicts delta movements (but output last timestamp + predicted deltas),
    with encoding person history and neighbors relative positions.
    """

    def __init__(self, lstm_hidden_dim, num_layers=1, bidir=True, dropout_p=0.5, num_modes=20):
        super(LstmEncDeltaStackedFullPredMultyGaus, self).__init__()
        self.name = "LstmEncDeltaStackedFullPredMultyGaus"
        self.num_modes = num_modes
        self.embedding_dim = 0
        self.lstm_hidden_dim = lstm_hidden_dim
        self.num_layers = num_layers
        self.dir_number = 1
        if bidir:
            self.dir_number = 2
        self.node_hist_encoder = nn.LSTM(input_size=6,
                                         hidden_size=lstm_hidden_dim,
                                         num_layers=num_layers,
                                         bidirectional=bidir,
                                         batch_first=True,
                                         dropout=0.5)

        self.node_hist_encoder_vel = nn.LSTM(input_size=2,
                                         hidden_size=lstm_hidden_dim,
                                         num_layers=num_layers,
                                         bidirectional=bidir,
                                         batch_first=True,
                                         dropout=0.5)

        self.node_hist_encoder_acc = nn.LSTM(input_size=2,
                                             hidden_size=lstm_hidden_dim,
                                             num_layers=num_layers,
                                             bidirectional=bidir,
                                             batch_first=True,
                                             dropout=0.5)

        self.node_hist_encoder_poses = nn.LSTM(input_size=2,
                                             hidden_size=lstm_hidden_dim,
                                             num_layers=num_layers,
                                             bidirectional=bidir,
                                             batch_first=True,
                                             dropout=0.5)


        self.edge_encoder = nn.LSTM(input_size=2,
                                    hidden_size=lstm_hidden_dim,
                                    num_layers=num_layers,
                                    bidirectional=bidir,
                                    batch_first=True,
                                    dropout=0.5)

        self.gru = nn.GRUCell(2 * lstm_hidden_dim * self.dir_number + 2, num_modes*2)

        self.action = nn.Linear(2, 2)
        self.state = nn.Linear(2 * lstm_hidden_dim * self.dir_number, num_modes*2)

        self.proj_to_GMM_log_pis = nn.Linear(num_modes*2, num_modes)
        self.proj_to_GMM_mus = nn.Linear(num_modes*2, num_modes*2)
        self.proj_to_GMM_log_sigmas = nn.Linear(num_modes*2, num_modes*2)
        self.proj_to_GMM_corrs = nn.Linear(num_modes*2, num_modes)
        self.dropout_p = dropout_p

    def project_to_GMM_params(self, tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Projects tensor to parameters of a GMM with N components and D dimensions.

        :param tensor: Input tensor.
        :return: tuple(log_pis, mus, log_sigmas, corrs)
            WHERE
            - log_pis: Weight (logarithm) of each GMM component. [N]
            - mus: Mean of each GMM component. [N, D]
            - log_sigmas: Standard Deviation (logarithm) of each GMM component. [N, D]
            - corrs: Correlation between the GMM components. [N]
        """
        log_pis =  F.dropout(self.proj_to_GMM_log_pis(tensor), self.dropout_p)
        mus = F.dropout(self.proj_to_GMM_mus(tensor), self.dropout_p)
        log_sigmas = F.dropout(self.proj_to_GMM_log_sigmas(tensor), self.dropout_p)
        corrs = F.dropout(torch.tanh(self.proj_to_GMM_corrs(tensor)), self.dropout_p)
        return log_pis, mus, log_sigmas, corrs

    def forward(self, scene: torch.Tensor):
        """
        :param scene: tensor of shape num_peds, history_size, data_dim
        :return: predicted poses distributions for each agent at next 12 timesteps
        """
        bs = scene.shape[0]
        poses = scene[:, :, :2]
        pv = scene[:, :, 2:6]
        vel = scene[:, :, 2:4]
        acc = scene[:, :, 4:6]
        pav = scene[:, :, :6]

        # lstm_out, hid = self.node_hist_encoder(pav)  # lstm_out shape num_peds, timestamps ,  2*hidden_dim
        lstm_out_acc, hid = self.node_hist_encoder_acc(acc)  # lstm_out shape num_peds, timestamps ,  2*hidden_dim
        lstm_out_vell, hid = self.node_hist_encoder_vel(vel)  # lstm_out shape num_peds, timestamps ,  2*hidden_dim
        lstm_out_poses, hid = self.node_hist_encoder_poses(poses)
        lstm_out = lstm_out_vell + lstm_out_poses + lstm_out_acc
        # lstm_out = lstm_out_poses  # + lstm_out_poses

        current_pose = scene[:, -1, :2]  # num_people, data_dim
        current_state = poses[:, -1, :]
        np, data_dim = current_pose.shape
        stacked = current_pose.flatten().repeat(np).reshape(np, np * data_dim)
        deltas = (stacked - current_pose.repeat(1, np)).reshape(np, np, data_dim)  # np, np, data_dim

        distruction, _ = self.edge_encoder(deltas)
        encoded_history = torch.cat((lstm_out[:, -1:, :], distruction[:, -1:, :]), dim=1)
        a_0 = F.dropout(self.action(current_state.reshape(bs, -1)), self.dropout_p)
        state = F.dropout(self.state(encoded_history.reshape(bs, -1)), self.dropout_p)

        current_state = current_state.unsqueeze(1)
        gauses = []
        inp = F.dropout(torch.cat((encoded_history.reshape(bs, -1), a_0), dim=-1), self.dropout_p)
        for i in range(12):
            h_state = self.gru(inp.reshape(bs, -1), state)

            log_pis, deltas, log_sigmas, corrs = self.project_to_GMM_params(h_state)
            deltas = torch.clamp(deltas, max=1.5, min=-1.5)

            log_pis = log_pis.reshape(bs, -1)
            log_pis = log_pis - torch.logsumexp(log_pis, dim=-1, keepdim=True)
            deltas = deltas.reshape(bs, -1, 2)
            log_sigmas = log_sigmas.reshape(bs, -1, 2)
            corrs = corrs.reshape(bs, -1, 1)

            mus = deltas + current_state
            current_state = mus
            variance = torch.clamp(torch.exp(log_sigmas).unsqueeze(2) ** 2, max=1e3)

            m_diag = variance * torch.eye(2).to(variance.device)
            sigma_xy = torch.clamp(torch.prod(torch.exp(log_sigmas), dim=-1), min=1e-8, max=1e3)

            mix = D.Categorical(log_pis)
            comp = D.MultivariateNormal(mus, m_diag)
            gmm = D.MixtureSameFamily(mix, comp)
            t = (sigma_xy * corrs.squeeze()).reshape(-1, 1, 1)
            cov_matrix = m_diag  # + anti_diag
            gauses.append(gmm)
            a_t = gmm.sample()  # possible grad problems?
            a_tt = F.dropout(self.action(a_t.reshape(bs, -1)), self.dropout_p)
            state = h_state
            inp = F.dropout(torch.cat((encoded_history.reshape(bs, -1), a_tt), dim=-1), self.dropout_p)

        return gauses
