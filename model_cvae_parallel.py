import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.distributions as D
from torch.autograd import Variable
from typing import List
import torch.nn.utils.rnn as rnn

def to_one_hot(y, n_dims=None):
    """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
    y_tensor = y.data if isinstance(y, Variable) else y
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return Variable(y_one_hot) if isinstance(y, Variable) else y_one_hot



class CvaeFuture(nn.Module):
    """
    model that actually predicts delta movements (but output last timestamp + predicted deltas),
    with encoding person history and neighbors relative positions.
    """

    def __init__(self, lstm_hidden_dim, num_layers=1, bidir=True, dropout_p=0.5, num_modes=20):
        super(CvaeFuture, self).__init__()
        self.name = "CvaeFuture_parallel"
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

        self.bn1 = nn.BatchNorm1d(8)
        self.bn2 = nn.BatchNorm1d(8)
        self.bn3 = nn.BatchNorm1d(num_modes*2)
        self.bn4 = nn.BatchNorm1d(num_modes*2)
        # self.bn5 = nn.BatchNorm1d()
        # self.bn6 = nn.BatchNorm1d()

        self.ln1 = nn.LayerNorm([8, lstm_hidden_dim])
        self.ln2 = nn.LayerNorm([8, lstm_hidden_dim])
        self.ln3 = nn.LayerNorm([num_modes*2])
        self.ln4 = nn.LayerNorm([num_modes*2])

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


        self.edge_encoder = nn.LSTM(input_size=12,
                                    hidden_size=lstm_hidden_dim,
                                    num_layers=num_layers,
                                    bidirectional=bidir,
                                    batch_first=True,
                                    dropout=0.5)



        self.node_future_encoder = nn.LSTM(input_size=6,
                                    hidden_size=lstm_hidden_dim,
                                    num_layers=num_layers,
                                    bidirectional=bidir,
                                    batch_first=True,
                                    dropout=0.5)

        self.action = nn.Linear(2, 12)
        self.gru = nn.GRUCell(2 * lstm_hidden_dim * self.dir_number + self.action.out_features, num_modes*2)


        self.state = nn.Linear(2 * lstm_hidden_dim * self.dir_number, num_modes*2)
        self.proj_p_to_log_pis = nn.Linear(lstm_hidden_dim*2*self.dir_number, num_modes)
        self.proj_to_GMM_log_pis = nn.Linear(lstm_hidden_dim*3*self.dir_number, num_modes)
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
        # log_pis =  F.dropout(self.proj_to_GMM_log_pis(tensor), self.dropout_p)
        mus = F.dropout(self.proj_to_GMM_mus(tensor), self.dropout_p)
        log_sigmas = F.dropout(self.proj_to_GMM_log_sigmas(tensor), self.dropout_p)
        corrs = F.dropout(torch.tanh(self.proj_to_GMM_corrs(tensor)), self.dropout_p)
        return None, mus, log_sigmas, corrs

    def sample_q(self):
        pass

    def sample_p(self):
        pass


    def obtain_encoded_tensors(self, scene: torch.Tensor, neighbors,  train):
        bs = scene.shape[0]
        poses = scene[:, :8, :2]
        pv = scene[:, :8, 2:6]
        vel = scene[:, :8, 2:4]
        acc = scene[:, :8, 4:6]
        pav = scene[:, :8, :6]
        if train:
            future = scene[:, 8:, :6]

        # lstm_out, hid = self.node_hist_encoder(pav)  # lstm_out shape num_peds, timestamps , 2*hidden_dim
        lstm_out_acc, hid = self.node_hist_encoder_acc(acc)  # lstm_out shape num_peds, timestamps,  2*hidden_dim
        lstm_out_vell, hid = self.node_hist_encoder_vel(vel)  # lstm_out shape num_peds, timestamps,  2*hidden_dim
        lstm_out_poses, hid = self.node_hist_encoder_poses(poses)
        # lstm_out = self.bn1(lstm_out_vell + lstm_out_poses + lstm_out_acc)
        lstm_out = F.dropout(self.ln1(lstm_out_vell + lstm_out_poses + lstm_out_acc), self.dropout_p)

        y_e = None
        if train:
            future_enc, hid = self.node_future_encoder(future)
            y_e = future_enc[:, -1, :]

        # lstm_out = lstm_out_poses  # + lstm_out_poses

        current_pose = scene[:, 7, :2]  # num_people, data_dim

        # np, data_dim = current_pose.shape
        # stacked = current_pose.flatten().repeat(np).reshape(np, np * data_dim)
        # deltas = (stacked - current_pose.repeat(1, np)).reshape(np, np, data_dim)  # np, np, data_dim
        #
        # distruction, _ = self.edge_encoder(deltas)
        # encoded_history = torch.cat((lstm_out[:, -1:, :], distruction[:, -1:, :]), dim=1)
        # encoded_edges = self.bn2(self.encode_edge(neighbors, pav, train))
        encoded_edges = self.ln2(self.encode_edge(neighbors, pav, train)),self.dropout_p
        encoded_history = torch.cat([lstm_out[:, -1, :], encoded_edges[:, -1, :]], dim=-1)

        return encoded_history, y_e

    def encode_edge(self, neighbors, node_history_st, train):

        # edge_states_list = []
        # for i, neighbor_states in enumerate(neighbors):  # Get neighbors for timestep in batch
        #     if len(neighbor_states) == 0:  # There are no neighbors for edge type # TODO necessary?
        #         pass
        #         # neighbor_state_length = int(
        #         #     np.sum([len(entity_dims) for entity_dims in self.state[edge_type[1]].values()])
        #         # )
        #         # edge_states_list.append(torch.zeros((1, max_hl + 1, neighbor_state_length), device=self.device))
        #     else:
        #         edge_states_list.append(torch.stack(neighbor_states, dim=0).to(self.device))

        # if self.hyperparams['edge_state_combine_method'] == 'sum':
            # Used in Structural-RNN to combine edges as well.
        op_applied_edge_states_list = list()
        for neighbors_state in neighbors:
            op_applied_edge_states_list.append(torch.sum(neighbors_state, dim=0))
        combined_neighbors = torch.stack(op_applied_edge_states_list, dim=0).to(node_history_st.device)

        joint_history = torch.cat([combined_neighbors, node_history_st], dim=-1)

        outputs, _ = self.run_lstm_on_variable_length_seqs(
            self.edge_encoder,
            original_seqs=joint_history,
            lower_indices=None
        )

        outputs = F.dropout(outputs,
                            p=self.dropout_p,
                            training=train)  # [bs, max_time, enc_rnn_dim]
        return outputs

    def run_lstm_on_variable_length_seqs(self, lstm_module, original_seqs, lower_indices=None, upper_indices=None,
                                         total_length=None):
        bs, tf = original_seqs.shape[:2]
        if lower_indices is None:
            lower_indices = torch.zeros(bs, dtype=torch.int)
        if upper_indices is None:
            upper_indices = torch.ones(bs, dtype=torch.int) * (tf - 1)
        if total_length is None:
            total_length = max(upper_indices) + 1
        # This is done so that we can just pass in self.prediction_timesteps
        # (which we want to INCLUDE, so this will exclude the next timestep).
        inclusive_break_indices = upper_indices + 1

        pad_list = list()
        for i, seq_len in enumerate(inclusive_break_indices):
            pad_list.append(original_seqs[i, lower_indices[i]:seq_len])

        packed_seqs = rnn.pack_sequence(pad_list, enforce_sorted=False)
        packed_output, (h_n, c_n) = lstm_module(packed_seqs)
        output, _ = rnn.pad_packed_sequence(packed_output,
                                            batch_first=True,
                                            total_length=total_length)

        return output, (h_n, c_n)
    def sample(self, p_distrib: D.Categorical):
        pass
        z = p_distrib.sample()
        return z

    def encoder(self, x, y_e=None, train=True):
        # q_distrib -> onehot?
        # p_distrib -> onehot?
        # kl dist
        # sample z
        bs = x.shape[0]
        p_distrib = D.Categorical(logits=F.dropout(self.proj_p_to_log_pis(x.reshape(bs, -1)), self.dropout_p))
        kl = 0
        q_distrib = None

        if train:
            h = torch.cat((x.reshape(bs, -1), y_e), dim=1)
            q_distrib = D.Categorical(logits=F.dropout(self.proj_to_GMM_log_pis(h), self.dropout_p))
            z = self.sample(q_distrib)
            kl_separated = D.kl_divergence(q_distrib, p_distrib)
            # kl_lower_bounded = torch.clamp(, min=0.1, max=1e3)
            kl = - torch.clamp(torch.sum(kl_separated), min=0.01, max=1e10)#- torch.sum(q_distrib.entropy())
        else:
            z = self.sample(p_distrib)
        return p_distrib, q_distrib, z, kl

    def decoder(self, z, encoded_history, current_state, y_e=None, train=False):
        pass

        bs = encoded_history.shape[0]
        a_0 = F.dropout(self.action(current_state.reshape(bs, -1)), self.dropout_p)

        # state = self.bn3(F.dropout(self.state(encoded_history.reshape(bs, -1)), self.dropout_p))
        state = self.ln3(F.dropout(self.state(encoded_history.reshape(bs, -1)), self.dropout_p))

        current_state = current_state.unsqueeze(1)
        gauses = []
        inp = F.dropout(torch.cat((encoded_history.reshape(bs, -1), a_0), dim=-1), self.dropout_p)

        for i in range(12):
            h_state = self.ln4(self.gru(inp.reshape(bs, -1), state))
            # h_state = self.bn4(self.gru(inp.reshape(bs, -1), state))

            _, deltas, log_sigmas, corrs = self.project_to_GMM_params(h_state)
            deltas = torch.clamp(deltas, max=1.5, min=-1.5)
            deltas = deltas.reshape(bs, -1, 2)
            log_sigmas = log_sigmas.reshape(bs, -1, 2)
            corrs = corrs.reshape(bs, -1, 1)

            mus = deltas + current_state
            current_state = mus
            variance = torch.clamp(torch.exp(log_sigmas).unsqueeze(2) ** 2, max=1e3, min=1e-6)

            m_diag = variance * torch.eye(2).to(variance.device)
            sigma_xy = torch.clamp(torch.prod(torch.exp(log_sigmas), dim=-1), min=1e-8, max=1e3)

            if train:
                # log_pis = z.reshape(bs, 1) * torch.ones(bs, self.num_modes).cuda()
                log_pis = to_one_hot(z, n_dims=self.num_modes).to(encoded_history.device)

            else:
                log_pis = to_one_hot(z, n_dims=self.num_modes).to(encoded_history.device)
            log_pis = log_pis - torch.logsumexp(log_pis, dim=-1, keepdim=True)
            mix = D.Categorical(logits=log_pis)
            try:
                comp = D.MultivariateNormal(mus, m_diag)
            except RuntimeError as e:
                print(m_diag)
                raise e
            gmm = D.MixtureSameFamily(mix, comp)
            t = (sigma_xy * corrs.squeeze()).reshape(-1, 1, 1)
            cov_matrix = m_diag  # + anti_diag
            gauses.append(gmm)
            a_t = gmm.sample()  # possible grad problems?
            a_tt = F.dropout(self.action(a_t.reshape(bs, -1)), self.dropout_p)
            state = h_state
            inp = F.dropout(torch.cat((encoded_history.reshape(bs, -1), a_tt), dim=-1), self.dropout_p)
        return gauses


    def loss(self, gmm, kl):
        # ELBO = log_likelihood - self.kl_weight * kl + 1. * mutual_inf_p
        pass

    def forward(self, scene: torch.Tensor, neighbours: List ,train=False):
        """
        :param scene: tensor of shape num_peds, history_size, data_dim
        :return: predicted poses distributions for each agent at next 12 timesteps
        """

        encoded_history, enc_future = self.obtain_encoded_tensors(scene, neighbours, train)
        p_distrib, q_distrib, z, kl = self.encoder(encoded_history, enc_future, train)
        gmm = self.decoder(z, encoded_history, scene[:, 7, :2], train=train)
        if train:
            return gmm, kl
        else:
            return gmm
