import torch
from dataloader import Dataset_from_pkl, is_filled
from model import LSTMTagger, LSTM_hid, LSTM_simple, OneLayer, LSTM_single, LSTM_single_with_emb, LSTM_delta, LSTM_enc_delta, LSTM_enc_delta_wo_emb
torch.manual_seed(1)
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np


def plot_traj(data, ax=None, color="red"):
    # data shape [n, 2]
    x_poses = np.zeros((data.shape[0], data.shape[1]))
    y_poses = np.zeros((data.shape[0], data.shape[1]))
    for person in range(data.shape[0]):
        x_poses[person] = data[person, :, 0].numpy()
        y_poses[person] = data[person, :, 1].numpy()

    if data.shape[0] < 2:
        for person in range(data.shape[0]):
            if ax is not None:
                ax.plot(x_poses[person], y_poses[person], 'o-', color=color)
            else:
                plt.plot(x_poses[person], y_poses[person], 'o-', color=color)
                plt.show()
    else:
        for person, color_index in enumerate(np.linspace(0, 1, data.shape[0])):
            if ax is not None:
                ax.plot(x_poses[int(person)], y_poses[int(person)], 'o-', color=plt.cm.RdYlBu(color_index))
            else:
                plt.plot(x_poses[int(person)], y_poses[int(person)], 'o-', color=plt.cm.RdYlBu(color_index))
                plt.show()
def visualize(model, gen, limit=10e7, device="cuda") :
    for batch_id, local_batch in enumerate(gen):
        if local_batch.shape[1] < 1:
            continue
        if batch_id > limit:
            # stop to next epoch
            break


        gt = local_batch.clone()
        local_batch = local_batch[:, :, :8, 2:4].to(device)
        local_batch[0, :, 8:, :] = torch.zeros_like(local_batch[0, :, 8:, :]).to(device)

        num_peds = local_batch.shape[1]
        predictions = torch.zeros(num_peds, 0, 2).requires_grad_(True).to(device)
        for t in range(0, 12):
            prediction = model(local_batch[0, :, 0 + t:8 + t, :])
            predictions = torch.cat((predictions, prediction), dim=1)
            local_batch = torch.cat((local_batch, prediction.unsqueeze(0)), dim=2)

        for ped_num in range(num_peds):
            if is_filled(local_batch[0, ped_num, :8, :]):
                fig = plt.figure()
                ax1 = fig.add_subplot(2, 1, 1)
                ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
                # fig, ax = plt.subplots(1, 3, figsize=(9, 3))
                plot_traj(local_batch[0, :, :8, :].detach().cpu(), ax=ax1, color="blue")

                # data = local_batch[0, :, :, 2:4].cpu()
                # plot_traj(data, ax=ax[0], color="blue")

                # data = torch.cat((gt[0,ped_num:ped_num+1,0:8,2:4], predictions[ped_num:ped_num+1,:,:].detach().cpu()),dim=1)\

                # plot_traj(gt[0, ped_num:ped_num + 1, 0:8, 2:4], ax[2], color="blue")

                plot_traj(predictions[ped_num:ped_num + 1, :, :].detach().cpu(), ax2)
                plot_traj(gt[0, ped_num:ped_num + 1, 8:, 2:4].detach().cpu(), ax2, color="black")
                plot_traj(local_batch[0, ped_num:ped_num + 1, :8, :].detach().cpu(), ax2, color="blue")
                pass
                plt.show()
                # ax1.axis('equal')
                # ax2.axis('equal')
                # ax[1].axis('equal')
                # ax[2].axis('equal')

                # ax[0].set_ylim(-10, 10)
                # ax[0].set_xlim(-10, 10)
                #
                # ax[1].set_ylim(-10, 10)
                # ax[1].set_xlim(-10, 10)


if __name__ == "__main__":

    dataset = Dataset_from_pkl("/home/robot/repos/trajectory-prediction/processed/", data_files=["eth_test.pkl"])
    generator = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    model = LSTM_enc_delta(10, 10, 2, 1).to(device)
    # model = LSTM_enc_delta(10, 10, 2, 10).to(device)
    path = "/home/robot/repos/trajectory-prediction/final LSTM_enc_delta0.002_hd_10_ed_10_nl_1@08.07.2020-16:17:16.pkl"
    model.load_state_dict(torch.load(path))
    # model = torch.load()

    visualize(model, generator)