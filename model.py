import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
torch.manual_seed(1)

input_dim = 2 # x, y
hidden_dim = 10
n_layers = 1
tagset_size = 2


class LSTMTagger(nn.Module):

    def __init__(self, hidden_dim, input_dim, tagset_size, num_pred=12):
        super(LSTMTagger, self).__init__()
        self.num_pred = num_pred
        self.hidden_dim = hidden_dim

#         self.pose_embeddings = nn.Linear(input_dim, embedding_dim)

        self.node_future_encoder = nn.LSTM(input_size=input_dim,
                              hidden_size=hidden_dim,
                              bidirectional=True,
                              batch_first=True)

        self.edge_encoder = nn.LSTM(input_size=input_dim*2,
                                  hidden_size=hidden_dim,
                                  bidirectional=True,
                                  batch_first=True)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(2*hidden_dim, tagset_size)

    def forward(self, self_pose, others_pose):

        lstm_out, _ = self.node_future_encoder(self_pose)
        for person in range(others_pose.shape[1]):
            other_pose = others_pose[:, person, :, :]
            concated = torch.cat((self_pose, other_pose), dim=-1)
            distr , _ = self.edge_encoder(concated)
#             distr , edge_enc_hiden = self.edge_encoder(concated, edge_enc_hiden)

        tag_space = self.hidden2tag(lstm_out + distr)

        return tag_space[:, -1:, :]


class LSTM_hid(nn.Module):

    def __init__(self, hidden_dim, input_dim, tagset_size, num_pred=12):
        super(LSTM_hid, self).__init__()
        self.num_pred = num_pred

        self.node_future_encoder = nn.LSTM(input_size=input_dim,
                              hidden_size=hidden_dim,
                              bidirectional=True,
                              batch_first=True)

        self.edge_encoder = nn.LSTM(input_size=input_dim*2,
                                  hidden_size=hidden_dim,
                                  bidirectional=True,
                                  batch_first=True)

        # The linear layer that maps from hidden state space to tag space
        self.edge_hidden = torch.zeros(2, 1, hidden_dim)
        self.edge_state = torch.zeros(2, 1, hidden_dim)
        self.hidden2tag = nn.Linear(2*hidden_dim, tagset_size)

    def forward(self, self_pose, others_pose):

        lstm_out, _ = self.node_future_encoder(self_pose)
        self.edge_hidden = torch.zeros(2, 1, hidden_dim).to(self_pose.device)
        self.edge_state = torch.zeros(2, 1, hidden_dim).to(self_pose.device)
        for person in range(others_pose.shape[1]):
            other_pose = others_pose[:, person, :, :]
            concated = torch.cat((self_pose, other_pose), dim=-1)
            distr , (self.edge_hidden, self.edge_state) = self.edge_encoder(concated, (self.edge_hidden, self.edge_state))
#             distr , edge_enc_hiden = self.edge_encoder(concated, edge_enc_hiden)

        tag_space = self.hidden2tag(lstm_out + distr)

        return tag_space[:, -1:, :]




class LSTM_simple(nn.Module):

    def __init__(self, hidden_dim, input_dim, tagset_size, num_pred=12):
        super(LSTM_simple, self).__init__()
        self.num_pred = num_pred

        self.node_future_encoder = nn.LSTM(input_size=input_dim,
                              hidden_size=hidden_dim,
                              bidirectional=True,
                              batch_first=True)


        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(2*hidden_dim, tagset_size)

    def forward(self, self_pose):

        lstm_out, _ = self.node_future_encoder(self_pose)

        tag_space = self.hidden2tag(lstm_out)

        return tag_space[:, -1:, :]



class LSTM_single(nn.Module):

    def __init__(self, hidden_dim, input_dim, tagset_size, num_pred=12):
        super(LSTM_single, self).__init__()
        self.num_pred = num_pred

        self.node_future_encoder = nn.LSTM(input_size=input_dim,
                              hidden_size=hidden_dim,
                              bidirectional=True,
                              batch_first=True)

        self.edge_encoder = nn.LSTM(input_size=input_dim,
                                  hidden_size=hidden_dim,
                                  bidirectional=True,
                                  batch_first=True)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(2*hidden_dim, tagset_size)

    def forward(self, scene):
        # scene shape = num_peds, timestamps, data_dim
        lstm_out, _ = self.node_future_encoder(scene) # lstm_out shape num_peds, timestamps ,  2*hidden_dim
        distr, _ = self.edge_encoder(scene)  # shape num_peds, timestamps ,  2*hidden_dim
        tag_space = self.hidden2tag(lstm_out + distr)

        return tag_space[:, -1:, :]



class LSTM_single_with_emb(nn.Module):

    def __init__(self, hidden_dim, input_dim, tagset_size, num_pred=12):
        super(LSTM_single_with_emb, self).__init__()
        self.num_pred = num_pred
        self.input_dim = input_dim
        self.inp_emb = nn.Linear(in_features=2, out_features=input_dim)
        self.current_inp_emb = nn.Linear(in_features=2, out_features=input_dim)
        self.node_future_encoder = nn.LSTM(input_size=input_dim,
                                           hidden_size=hidden_dim,
                                           num_layers=3,
                                           bidirectional=True,
                                           batch_first=True,
                                           dropout=0.5)

        self.edge_encoder = nn.LSTM(input_size=input_dim,
                                  hidden_size=hidden_dim,
                                  num_layers=3,
                                  bidirectional=True,
                                  batch_first=True,
                                  dropout=0.5)
        self.hidden_dim = hidden_dim
        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(18*hidden_dim, tagset_size)

    def forward(self, scene):
        inp = scene
        embedded = F.relu(self.inp_emb(inp).reshape(scene.shape[0], -1, self.input_dim))
        # scene shape = num_peds, timestamps, data_dim
        lstm_out, _ = self.node_future_encoder(embedded)  # lstm_out shape num_peds, timestamps ,  2*hidden_dim
        current = scene[:, -1, :]
        current_emb = F.relu(self.current_inp_emb(current))
        distr, _ = self.edge_encoder(current_emb.unsqueeze(1))  # shape num_peds, timestamps ,  2*hidden_dim
        catted = torch.cat((lstm_out, distr), dim=1).reshape(-1, 18 * self.hidden_dim)
        tag_space = self.hidden2tag(catted).unsqueeze(1)

        return tag_space


class LSTM_delta(nn.Module):

    def __init__(self, lstm_hidden_dim, embedding_dim, target_size, num_layers=1):
        super(LSTM_delta, self).__init__()

        self.embedding_dim = embedding_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.inp_emb = nn.Linear(in_features=2, out_features=embedding_dim)
        self.current_inp_emb = nn.Linear(in_features=2, out_features=embedding_dim)
        self.node_hist_encoder = nn.LSTM(input_size=embedding_dim,
                                           hidden_size=lstm_hidden_dim,
                                           num_layers=num_layers,
                                           bidirectional=True,
                                           batch_first=True,
                                           dropout=0.5)

        self.edge_encoder = nn.LSTM(input_size=embedding_dim,
                                  hidden_size=lstm_hidden_dim,
                                  num_layers=num_layers,
                                  bidirectional=True,
                                  batch_first=True,
                                  dropout=0.5)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2pose = nn.Linear(18*lstm_hidden_dim, target_size)

    def forward(self, scene: torch.Tensor) -> torch.Tensor:
        """
        :param scene: tensor of shape num_peds, history_size, data_dim
        :return: predicted poses for each agent at next timestep
        """

        inp = scene
        embedded = F.relu(self.inp_emb(inp))
        embedded = embedded.reshape(scene.shape[0], -1, self.embedding_dim)
        # scene shape = num_peds, timestamps, data_dim
        lstm_out, _ = self.node_hist_encoder(embedded)  # lstm_out shape num_peds, timestamps ,  2*hidden_dim
        current = scene[:, -1, :]
        current_emb = F.relu(self.current_inp_emb(current))
        distr, _ = self.edge_encoder(current_emb.unsqueeze(1))  # shape num_peds, timestamps ,  2*hidden_dim
        catted = torch.cat((lstm_out, distr), dim=1).reshape(-1, 18 * self.lstm_hidden_dim)
        tag_space = (scene.clone()[:, -1, :] + self.hidden2pose(catted)).unsqueeze(1)

        return tag_space



class LSTM_enc_delta(nn.Module):

    def __init__(self, lstm_hidden_dim, embedding_dim, target_size, num_layers=1):
        super(LSTM_enc_delta, self).__init__()
        self.name = "LSTM_enc_delta"
        self.embedding_dim = embedding_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.num_layers = num_layers
        self.inp_emb = nn.Linear(in_features=2, out_features=embedding_dim)
        self.current_inp_emb = nn.Linear(in_features=2, out_features=embedding_dim)
        self.node_hist_encoder = nn.LSTM(input_size=embedding_dim,
                                           hidden_size=lstm_hidden_dim,
                                           num_layers=num_layers,
                                           bidirectional=True,
                                           batch_first=True,
                                           dropout=0.5)

        self.edge_encoder = nn.LSTM(input_size=embedding_dim,
                                  hidden_size=lstm_hidden_dim,
                                  num_layers=num_layers,
                                  bidirectional=True,
                                  batch_first=True,
                                  dropout=0.5)

        self.decoder = nn.LSTM(input_size=2 * lstm_hidden_dim,
                                    hidden_size=embedding_dim,
                                    num_layers=num_layers,
                                    bidirectional=True,
                                    batch_first=True,
                                    dropout=0.5)

        # The linear layer that maps from hidden state space to tag space

        self.hidden2pose = nn.Linear(9*2*embedding_dim, target_size)

    def forward(self, scene: torch.Tensor) -> torch.Tensor:
        """
        :param scene: tensor of shape num_peds, history_size, data_dim
        :return: predicted poses for each agent at next timestep
        """
        bs = scene.shape[0]
        inp = scene
        embedded = F.relu(self.inp_emb(inp))
        embedded = embedded.reshape(scene.shape[0], -1, self.embedding_dim)
        # scene shape = num_peds, timestamps, data_dim
        lstm_out, _ = self.node_hist_encoder(embedded)  # lstm_out shape num_peds, timestamps ,  2*hidden_dim
        current = scene[:, -1, :]
        current_emb = F.relu(self.current_inp_emb(current))
        distr, _ = self.edge_encoder(current_emb.unsqueeze(1))  # shape num_peds, timestamps ,  2*hidden_dim
        catted = torch.cat((lstm_out, distr), dim=1)
        decoded, _ = self.decoder(catted)
        dec_reshaped = decoded.reshape(bs, -1)
        tag_space = (scene.clone()[:, -1, :] + self.hidden2pose(dec_reshaped)).unsqueeze(1)

        return tag_space



class LSTM_enc_delta_wo_emb(nn.Module):

    def __init__(self, lstm_hidden_dim, target_size, num_layers=1, embedding_dim=10):
        super(LSTM_enc_delta_wo_emb, self).__init__()
        self.name = "LSTM_enc_delta_wo_emb"
        self.embedding_dim = embedding_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.num_layers = num_layers
        # self.inp_emb = nn.Linear(in_features=2, out_features=embedding_dim)
        # self.current_inp_emb = nn.Linear(in_features=2, out_features=embedding_dim)
        self.node_hist_encoder = nn.LSTM(input_size=2,
                                           hidden_size=lstm_hidden_dim,
                                           num_layers=num_layers,
                                           bidirectional=True,
                                           batch_first=True,
                                           dropout=0.5)

        self.edge_encoder = nn.LSTM(input_size=2,
                                  hidden_size=lstm_hidden_dim,
                                  num_layers=num_layers,
                                  bidirectional=True,
                                  batch_first=True,
                                  dropout=0.5)

        self.decoder = nn.LSTM(input_size=2 * lstm_hidden_dim,
                                    hidden_size=embedding_dim,
                                    num_layers=num_layers,
                                    bidirectional=True,
                                    batch_first=True,
                                    dropout=0.5)

        # The linear layer that maps from hidden state space to tag space

        self.hidden2pose = nn.Linear(9*2*embedding_dim, target_size)

    def forward(self, scene: torch.Tensor) -> torch.Tensor:
        """
        :param scene: tensor of shape num_peds, history_size, data_dim
        :return: predicted poses for each agent at next timestep
        """
        bs = scene.shape[0]
        inp = scene
        # embedded = embedded.reshape(scene.shape[0], -1, self.embedding_dim)
        # scene shape = num_peds, timestamps, data_dim
        lstm_out, _ = self.node_hist_encoder(inp)  # lstm_out shape num_peds, timestamps ,  2*hidden_dim
        current = scene[:, -1, :]
        # current_emb = F.relu(self.current_inp_emb(current))
        distr, _ = self.edge_encoder(current.unsqueeze(1))  # shape num_peds, timestamps ,  2*hidden_dim
        catted = torch.cat((lstm_out, distr), dim=1)
        decoded, _ = self.decoder(catted)
        dec_reshaped = decoded.reshape(bs, -1)
        tag_space = (scene.clone()[:, -1, :] + self.hidden2pose(dec_reshaped)).unsqueeze(1)

        return tag_space


class LSTM_enc_delta_stacked(nn.Module):

    def __init__(self, lstm_hidden_dim, target_size, num_layers=1, embedding_dim=10, bidir=True):
        super(LSTM_enc_delta_stacked, self).__init__()
        self.name = "LSTM_enc_delta_stacked"
        self.embedding_dim = embedding_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.num_layers = num_layers
        self.dir_number = 1
        if bidir:
            self.dir_number = 2
        self.node_hist_encoder = nn.LSTM(input_size=2,
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

        self.decoder = nn.LSTM(input_size=self.dir_number * lstm_hidden_dim,
                                    hidden_size=embedding_dim,
                                    num_layers=num_layers,
                                    bidirectional=bidir,
                                    batch_first=True,
                                    dropout=0.5)

        # The linear layer that maps from hidden state space to pose space

        self.hidden2pose = nn.Linear(9 * self.dir_number * embedding_dim, target_size)

    def forward(self, scene: torch.Tensor) -> torch.Tensor:
        """
        :param scene: tensor of shape num_peds, history_size, data_dim
        :return: predicted poses for each agent at next timestep
        """
        bs = scene.shape[0]
        inp = scene

        lstm_out, hid = self.node_hist_encoder(inp)  # lstm_out shape num_peds, timestamps ,  2*hidden_dim
        current = scene[:, -1, :] # num_people, data_dim
        np, data_dim = current.shape
        stacked = current.flatten().repeat(np).reshape(np, np * data_dim)
        deltas = (stacked - current.repeat(1, np)).reshape(np, np, data_dim) # np, np, data_dim
        # non_self_deltas = deltas[deltas != 0]
        # hiden_state = torch.zeros_like(hid[0])
        # cell_state = torch.zeros_like(hid[1])
        # for person_id in range(np):

        # distruction, _ = self.edge_encoder(non_self_deltas.reshape(np,np-1,-1))
        distruction, _ = self.edge_encoder(deltas)
        catted = torch.cat((lstm_out, distruction[:, -1:, :]), dim=1)
        decoded, _ = self.decoder(catted)
        dec_reshaped = decoded.reshape(bs, -1)
        tag_space = (scene.clone()[:, -1, :] + self.hidden2pose(dec_reshaped)).unsqueeze(1)
        return tag_space

class OneLayer(nn.Module):
    def __init__(self):
        super(OneLayer, self).__init__()
        self.simple_linear = nn.Linear(16, 2)

    #         self.simple_linear1 = nn.Linear(20, 40)
    #         self.simple_linear2 = nn.Linear(40, 8)

    def forward(self, self_pose, others_pose):
        self_pose = self_pose.reshape(-1, 16)

        out = self.simple_linear(self_pose)

        return out.reshape(1, 1, 2)
if __name__ == "__main__":
    # model = LSTM_hid(10, 2, 2)
    model = LSTM_delta(10, 20, 2)

    self_pose = torch.rand(1, 1, 20, 2)
    others_pose = torch.rand(1, 2, 20, 2)
    predictions = torch.zeros(1, 1, 0, 2)
    for timestamp in range(0, 12):
        inp = torch.cat((self_pose[:, :, 0+timestamp:8+timestamp, :], others_pose[:, :, 0+timestamp:8+timestamp, :]), dim=1)[0, :, :, :]
        print(inp.shape)
        prediction = model(inp)
        print (prediction.shape)
        predictions = torch.cat((predictions, prediction.unsqueeze(0)), dim=2)
    print(predictions.shape)
