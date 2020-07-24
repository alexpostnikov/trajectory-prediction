import torch
from dataloader import Dataset_from_pkl, is_filled
from model import LSTMTagger, OneLayer, LSTM_single, LSTM_single_with_emb, LSTM_delta, LSTM_enc_delta, LSTM_enc_delta_wo_emb
torch.manual_seed(1)
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np


def plot_prob(gt, gmm, ped_num):
    fig, ax = plt.subplots(4, 3, figsize=(18, 18))
    for i in range(12):
        plot_prob_step(gt[8+i], gmm[i], ped_num, ax[i % 4][i // 4])
        ax[i % 4][i // 4].set_title("timestamp " + str(i))

    fig.show()
    pass


counter = 0

def plot_prob_big(gt, gmm, ped_num):
    global counter

    x = torch.arange(torch.min(gt[8:, 0]) - 2, torch.max(gt[8:, 0]) + 2, 0.1)
    y = torch.arange(torch.min(gt[8:, 1]) - 2, torch.max(gt[8:, 1]) + 2, 0.1)
    prob = torch.zeros_like(y[None].T * x)
    for t in range(12):
        for i in range(len(x)):
            if abs(x[i] - gt[8+t, 0]) < 3:
                for j in range(len(y)):
                    if abs(y[j] - gt[8 + t, 1]) < 3:
                        prob[j][i] += torch.exp(gmm[t].log_prob(torch.tensor([x[i], y[j]]).cuda())[ped_num])

    prob = torch.clamp(prob, max=2)

    #(gt[:,0] - x.mean())/(torch.max(x) - torch.min(x)) * (len(x)) + (len(x)/2.0)
    gt[:, 0] = (gt[:, 0] - x.mean())/(torch.max(x) - torch.min(x)) * (len(x)) + (len(x)/2.0)
    gt[:, 1] = (gt[:, 1] - y.mean())/(torch.max(y) - torch.min(y)) * (len(y)) + (len(y)/2.0)

    fig, ax = plt.subplots(1, figsize=(18, 18))
    ax.set_xticks(np.round(np.linspace(0, len(x), 6), 1))
    ax.set_yticks(np.round(np.linspace(0, len(y), 6), 1))
    ax.set_xticklabels(
        np.round(((torch.linspace(0, len(x), 6) * (torch.max(x) - torch.min(x)) / len(x) + torch.min(x)).numpy()), 1))
    ax.set_yticklabels(
        np.round((((torch.linspace(0, len(x), 6)) * (torch.max(x) - torch.min(x)) / len(y) + torch.min(y)).numpy()), 1))
    ax.plot(gt[:, 0].detach().cpu(), gt[:, 1].detach().cpu())
    ax.plot(gt[8:, 0].detach().cpu(), gt[8:, 1].detach().cpu(), 'ro')
    ax.imshow(prob.detach().numpy())
    plt.savefig("./visualisations/traj_dirtr/"+str(counter)+".jpg", )
    counter += 1
    plt.close()
    pass


def plot_prob_step(gt, gmm, ped_num, ax):
        x = torch.arange(gt[0] - 2, gt[0] + 2, 0.1).cuda()
        y = torch.arange(gt[1] - 2, gt[1] + 2, 0.1).cuda()
        prob = torch.zeros_like(x[None].T * y)
        for i in range(len(x)):
            for j in range(len(y)):
                prob[i][j] = torch.exp(gmm.log_prob(torch.tensor([x[i], y[j]]).cuda())[ped_num])
        prob = prob.clamp(max=2)
        ax.imshow(prob.detach().cpu())

        # ax.set_xticks(np.arange(len(labels_x)))
        # ax.set_yticks(np.arange(len(labels_y)))

        # ax.set_xticklabels(labels_x.numpy())
        # ax.set_yticklabels(labels_y.numpy())
        ax.set_xticks(np.arange(0, len(x), 6))
        ax.set_yticks(np.arange(0, len(y), 6))
        ax.set_xticklabels(np.round((((torch.arange(0, len(x), 6))*(torch.max(x) - torch.min(x)) / len(x) + torch.min(x)).numpy()),1))
        ax.set_yticklabels(np.round(((( torch.arange(0, len(y), 6))*(torch.max(x) - torch.min(x)) / len(y) + torch.min(y)).numpy()),1))
        # for _ in range(20):
        #     mean_pr = gmm.sample()
        #     # mean_pr = torch.tensor([5.,5])
        #
        #     mean_pr[0] = (-torch.min(x) + mean_pr[0]) * len(x) / (torch.max(x) - torch.min(x))
        #     mean_pr[1] = (-torch.min(x) + mean_pr[1]) * len(x) / (torch.max(x) - torch.min(x))
        #
        #     circle = plt.Circle((mean_pr[0], mean_pr[1]), radius=0.4, color="r")
        #     ax.add_patch(circle)
        mean_pr = torch.zeros(2)
        mean_pr[0] = (-torch.min(x) + gt[0]) * len(x) / (torch.max(x) - torch.min(x))
        mean_pr[1] = (-torch.min(y) + gt[1]) * len(x) / (torch.max(y) - torch.min(y))

        circle = plt.Circle((mean_pr[0], mean_pr[1]), radius=0.4, color="r")
        ax.add_patch(circle)
        return ax


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
        local_batch = local_batch.to(device)
        if local_batch.shape[1] < 1:
            continue
        if batch_id > limit:
            # stop to next epoch
            break


        gt = local_batch.clone()
        # local_batch = local_batch[:, :, :8, 2:4].to(device)
        local_batch[0, :, 8:, :] = torch.zeros_like(local_batch[0, :, 8:, :]).to(device)

        num_peds = local_batch.shape[1]
        # predictions = torch.zeros(num_peds, 0, 2).requires_grad_(True).to(device)
        prediction = model(local_batch[0, :, 0:8, 2:8])
        predictions = torch.cat([prediction[i].mean for i in range(12)]).reshape(12, -1, 2).permute(1, 0, 2)
        for ped_num in range(num_peds):
            if is_filled(local_batch[0, ped_num, :8, :]):

                if not torch.any(torch.norm(gt[0, ped_num, 8:, 2:4],dim=-1)==torch.tensor([0]).cuda()):
                    plot_prob_big(gt[0, ped_num, :, 2:4], prediction, ped_num)
                    # plot_prob(gt[0, ped_num, :, 2:4], prediction, ped_num)

                # fig = plt.figure()
                # ax1 = fig.add_subplot(2, 1, 1)
                # ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
                # # fig, ax = plt.subplots(1, 3, figsize=(9, 3))
                # plot_traj(local_batch[0, :, :8, 2:4].detach().cpu(), ax=ax1, color="blue")
                #
                # # data = local_batch[0, :, :, 2:4].cpu()
                # # plot_traj(data, ax=ax[0], color="blue")
                #
                # # data = torch.cat((gt[0,ped_num:ped_num+1,0:8,2:4], predictions[ped_num:ped_num+1,:,:].detach().cpu()),dim=1)\
                #
                # # plot_traj(gt[0, ped_num:ped_num + 1, 0:8, 2:4], ax[2], color="blue")
                #
                # plot_traj(predictions[ped_num:ped_num + 1, :, :].detach().cpu(), ax2)
                # plot_traj(gt[0, ped_num:ped_num + 1, 8:, 2:4].detach().cpu(), ax2, color="black")
                # plot_traj(local_batch[0, ped_num:ped_num + 1, :8, 2:4].detach().cpu(), ax2, color="blue")
                # pass
                # plt.show()



if __name__ == "__main__":

    from tb_good.best.model import  LstmEncDeltaStackedFullPredMultyGaus


    dataset = Dataset_from_pkl("/home/robot/repos/trajectory-prediction/processed/", data_files=["eth_test.pkl"])
    generator = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    model = LstmEncDeltaStackedFullPredMultyGaus(lstm_hidden_dim=64, num_layers=1,
                                                 bidir=True, dropout_p=0.0, num_modes=30).to(device)

    # model = LSTM_enc_delta(10, 10, 2, 10).to(device)
    path = "/home/robot/repos/trajectory-prediction/tb_good/best/LstmEncDeltaStackedFullPredMultyGaus0.001_hd_64_ed_0_nl_1@21.07.2020-14:20:33+best.pth"
    model.load_state_dict(torch.load(path))
    # model = torch.load()

    visualize(model, generator)