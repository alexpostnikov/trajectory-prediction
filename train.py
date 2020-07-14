import math
import time
from datetime import datetime
from typing import Tuple, Union
import os

import torch
import torch.optim as optim
from dataloader import Dataset_from_pkl, is_filled
from model import LstmEncDeltaStacked, LstmEncDeltaStackedVel, LstmEncDeltaStackedFullPred
from torch.utils.tensorboard import SummaryWriter

from utils import compare_prediction_gt

torch.manual_seed(1)


def setup_experiment(title: str, logdir: str="./tb") -> Tuple[SummaryWriter, str, str]:
    """
    :param title: name of experiment
    :param logdir: tb logdir
    :return: writer object,  modified experiment_name, best_model path
    """
    experiment_name = "{}@{}".format(title, datetime.now().strftime("%d.%m.%Y-%H:%M:%S"))
    writer = SummaryWriter(log_dir=os.path.join(logdir, experiment_name))
    best_model_path = f"weights/{experiment_name}.best.pth"
    return writer, experiment_name, best_model_path


def get_ade_fde(generator: torch.utils.data.DataLoader, model: torch.nn.Module, limit: int = 1e10) -> Tuple[int, int]:
    """

    :param generator: torch generator to get data
    :param model:   torch module predicting poses. input shape are [numped, history, 2]
    :param limit: limit number of processed batches. Default - unlimited
    :return: tuple of ade, fde for given generator and limit of batches
    """
    ade = []
    fde = []
    for batch_id, local_batch in enumerate(generator):

        if batch_id > limit:
            break
        if local_batch.shape[1] < 1:
            continue
        gt = local_batch.clone()
        local_batch = local_batch.to(device)
        local_batch[0,:,8:,2:4] = torch.zeros_like(local_batch[0,:,8:,2:4])
        num_peds = local_batch.shape[1]
        predictions = torch.zeros(num_peds, 0, 2).requires_grad_(True).to(device)

        for t in range(0, 12):
            prediction = model(local_batch[0, :, 0 + t: 8 + t, 2:4])

            #                 prediction = prediction.unsqueeze(1)
            predictions = torch.cat((predictions, prediction), dim=1)
            local_batch[0, :, 8 + t:9 + t, 2:4] = prediction.detach()

        metrics = compare_prediction_gt(predictions.detach().cpu(), gt.detach().cpu()[0][:, :, 2:4])

        for local_ade in metrics["ade"]:
            ade.append(local_ade)
        for local_fde in metrics["fde"]:
            fde.append(local_fde)
    ade = sum(ade) / len(ade)
    fde = sum(fde) / len(fde)
    return (ade, fde)



def get_ade_fde_vel(generator: torch.utils.data.DataLoader, model: torch.nn.Module, limit: int = 1e10) -> Tuple[int, int]:
    """

    :param generator: torch generator to get data
    :param model:   torch module predicting poses. input shape are [numped, history, 2]
    :param limit: limit number of processed batches. Default - unlimited
    :return: tuple of ade, fde for given generator and limit of batches
    """
    ade = []
    fde = []
    for batch_id, local_batch in enumerate(generator):

        if batch_id > limit:
            break
        if local_batch.shape[1] < 1:
            continue
        gt = local_batch.clone()
        local_batch = local_batch.to(device)
        local_batch[0, :, 8:, :] = torch.zeros_like(local_batch[0, :, 8:, :])
        num_peds = local_batch.shape[1]
        predictions = torch.zeros(num_peds, 0, 2).requires_grad_(True).to(device)

        for t in range(0, 12):
            prediction = model(local_batch[0, :, 0 + t: 8 + t, 2:6])

            #                 prediction = prediction.unsqueeze(1)
            predictions = torch.cat((predictions, prediction), dim=1)
            new_vel = (prediction - local_batch[0, :, 7 + t:8 + t, :2]) / 0.4
            # local_batch = torch.cat((local_batch, torch.cat((prediction, new_vel), dim=2).unsqueeze(0)), dim=2)
            local_batch[0, :, 8 + t:9 + t, 2:6] = torch.cat((prediction, new_vel), dim=2).detach()

        metrics = compare_prediction_gt(predictions.detach().cpu(), gt.detach().cpu()[0][:, :, 2:4])

        for local_ade in metrics["ade"]:
            ade.append(local_ade)
        for local_fde in metrics["fde"]:
            fde.append(local_fde)
    ade = sum(ade) / len(ade)
    fde = sum(fde) / len(fde)
    return (ade, fde)

def get_batch_is_filled_mask(batch: torch.Tensor) -> Tuple[torch.Tensor, int]:
    """

    :param batch:  batch with shape bs, num_peds, seq, data_shape, currently works only with bs = 1
    :return: mask shape num_people, seq, data_shape with 0 filled data if person is not fully observable during seq
    """
    assert batch.shape[0] == 1

    num_peds = batch.shape[1]
    mask = torch.zeros(num_peds, 12, 2)
    full_peds = 0
    for ped in range(num_peds):
        if is_filled(batch[0][ped]):
            mask[ped] = torch.ones(12, 2)
            full_peds += 1
    return mask, full_peds


def train(model: torch.nn.Module, training_generator: torch.utils.data.DataLoader,
          test_generator: torch.utils.data.DataLoader, num_epochs:int, device: torch.device,
          lr: Union[int, float] = 0.02, limit: Union[int, float] = 10e7):
    """

    :param model: model for predicting the poses.  input shape are [numped, history, 2]
    :param training_generator:  torch training generator to get data
    :param test_generator: torch training generator to get data
    :param num_epochs: number of epochs to train
    :param device: torch device. (cpu/cuda)
    :param lr: learning rate
    :param limit: limit the number of training batches
    :return:
    """

    optimizer = optim.Adam(model.parameters(), lr=lr)
    drop_every_epochs = 10
    drop_rate = 1
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, drop_every_epochs,
                                                drop_rate)  # drop_every_epochs epochs drop by drop_rate lr
    writer, experiment_name, save_model_path = setup_experiment(model.name+str(lr) + "_hd_" + str(model.lstm_hidden_dim)
                                                                +"_ed_"  + str(model.embedding_dim)
                                                                + "_nl_"+str(model.num_layers))
    prev_epoch_loss = 0

    for epoch in range(0, num_epochs):
        model.train()
        epoch_loss = 0
        start = time.time()
        num_skipped = 0
        for batch_id, local_batch in enumerate(training_generator):
            if batch_id > limit:
                # skip to next epoch
                continue

            angle = torch.rand(1) * math.pi
            rot = torch.tensor([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
            local_batch[:, :, :, 2:4] = local_batch[:, :, :, 2:4] @ rot
            local_batch = local_batch.to(device)
            gt = local_batch.clone()
            model.zero_grad()
            num_peds = local_batch.shape[1]
            predictions = torch.zeros(num_peds, 0, 2).requires_grad_(True).to(device)
            mask, full_peds = get_batch_is_filled_mask(gt)
            if full_peds == 0:
                continue
            mask = mask.to(device)

            local_batch = local_batch[:, :, :8, 2:4]
            if torch.sum(mask) == 0.0:
                num_skipped += 1
                continue
            for t in range(0, 12):
                prediction = model(local_batch[0, :, 0 + t:8 + t, :])
                predictions = torch.cat((predictions, prediction), dim=1)
                local_batch = torch.cat((local_batch, prediction.unsqueeze(0)), dim=2)

            loss = torch.sum(torch.norm(mask * (predictions - gt[0, :, 8:, 2:4]), dim=-1))
            epoch_loss += loss.item() / full_peds
            loss.backward()
            optimizer.step()

        epoch_loss = epoch_loss / (batch_id - num_skipped)
        print("epoch {epoch} loss {el:0.4f}, time taken {t:0.2f}, delta {delta:0.3f}".format(epoch=epoch, el=epoch_loss,
                                                                                             t=time.time() - start,
                                                                                             delta=-prev_epoch_loss + epoch_loss))

        if writer is not None:
            writer.add_scalar(f"loss_epoch", epoch_loss, epoch)

        prev_epoch_loss = epoch_loss
        model.eval()
        start = time.time()
        with torch.no_grad():
            ade, fde = get_ade_fde(test_generator, model)

            if writer is not None:
                writer.add_scalar(f"test/ade", ade, epoch)
                writer.add_scalar(f"test/fde", fde, epoch)
            t_ade, t_fde = get_ade_fde(training_generator, model, 100)
            if writer is not None:
                writer.add_scalar(f"train/ade", t_ade, epoch)
                writer.add_scalar(f"train/fde", t_fde, epoch)
        print("\t epoch {epoch} val ade {ade:0.4f}, val fde {fde:0.4f} time taken {t:0.2f}".format(epoch=epoch, ade=ade,
                                                                                                   t=time.time()-start,
                                                                                                   fde=fde))
    torch.save(model.state_dict(), save_model_path)


def train_pose_vel(model: torch.nn.Module, training_generator: torch.utils.data.DataLoader,
          test_generator: torch.utils.data.DataLoader, num_epochs:int, device: torch.device,
          lr: Union[int, float] = 0.02, limit: Union[int, float] = 10e7):
    """

    :param model: model for predicting the poses.  input shape are [numped, history, 2]
    :param training_generator:  torch training generator to get data
    :param test_generator: torch training generator to get data
    :param num_epochs: number of epochs to train
    :param device: torch device. (cpu/cuda)
    :param lr: learning rate
    :param limit: limit the number of training batches
    :return:
    """

    optimizer = optim.Adam(model.parameters(), lr=lr)
    drop_every_epochs = 10
    drop_rate = 1
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, drop_every_epochs,
                                                drop_rate)  # drop_every_epochs epochs drop by drop_rate lr
    writer, experiment_name, save_model_path = setup_experiment(model.name+str(lr) + "_hd_" + str(model.lstm_hidden_dim)
                                                                +"_ed_"  + str(model.embedding_dim)
                                                                + "_nl_"+str(model.num_layers))
    prev_epoch_loss = 0

    for epoch in range(0, num_epochs):
        model.train()
        epoch_loss = 0
        start = time.time()
        num_skipped = 0
        for batch_id, local_batch in enumerate(training_generator):
            if batch_id > limit:
                # skip to next epoch
                continue

            # angle = torch.rand(1) * math.pi
            # rot = torch.tensor([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
            # local_batch[:, :, :, 2:6] = local_batch[:, :, :, 2:6] @ rot
            local_batch = local_batch.to(device)
            gt = local_batch.clone()
            model.zero_grad()
            num_peds = local_batch.shape[1]
            predictions = torch.zeros(num_peds, 0, 2).requires_grad_(True).to(device)
            mask, full_peds = get_batch_is_filled_mask(gt)
            if full_peds == 0:
                continue
            mask = mask.to(device)

            local_batch = local_batch[:, :, :8, 2:6]
            if torch.sum(mask) == 0.0:
                num_skipped += 1
                continue
            for t in range(0, 12):
                prediction = model(local_batch[0, :, 0 + t:8 + t, :])
                predictions = torch.cat((predictions, prediction), dim=1)
                new_vel = gt[0, :, 8+t:9+t:, 4:6]
                local_batch = torch.cat((local_batch, torch.cat((prediction, new_vel), dim=2).unsqueeze(0)), dim=2)

            loss = torch.sum(torch.norm(mask * (predictions - gt[0, :, 8:, 2:4]), dim=-1))
            epoch_loss += loss.item() / full_peds
            loss.backward()
            optimizer.step()

        epoch_loss = epoch_loss / (batch_id - num_skipped)
        print("epoch {epoch} loss {el:0.4f}, time taken {t:0.2f}, delta {delta:0.3f}".format(epoch=epoch, el=epoch_loss,
                                                                                             t=time.time() - start,
                                                                                             delta=-prev_epoch_loss + epoch_loss))

        if writer is not None:
            writer.add_scalar(f"loss_epoch", epoch_loss, epoch)

        prev_epoch_loss = epoch_loss
        model.eval()
        start = time.time()
        with torch.no_grad():
            ade, fde = get_ade_fde_vel(test_generator, model)

            if writer is not None:
                writer.add_scalar(f"test/ade", ade, epoch)
                writer.add_scalar(f"test/fde", fde, epoch)
            t_ade, t_fde = get_ade_fde_vel(training_generator, model, 100)
            if writer is not None:
                writer.add_scalar(f"train/ade", t_ade, epoch)
                writer.add_scalar(f"train/fde", t_fde, epoch)
        print("\t epoch {epoch} val ade {ade:0.4f}, val fde {fde:0.4f} time taken {t:0.2f}".format(epoch=epoch, ade=ade,
                                                                                                   t=time.time()-start,
                                                                                                   fde=fde))
    torch.save(model.state_dict(), save_model_path)



if __name__ == "__main__":
    training_set = Dataset_from_pkl("/home/robot/repos/trajectory-prediction/processed/", data_files=["eth_train.pkl"])
    training_generator = torch.utils.data.DataLoader(training_set, batch_size=1, shuffle=True)

    test_set = Dataset_from_pkl("/home/robot/repos/trajectory-prediction/processed/", data_files=["eth_test.pkl"])
    test_generator = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    # model  = LSTM_single(20, 2, 2).to(device)
    # model  = LSTM_simple(10, 2, 2).to(device)

    # model = LSTM_single_with_emb(40, 20, 2).to(device)

    # model = LSTM_delta(40, 20, 2, 1).to(device)

    # model = LstmEncDeltaStacked(lstm_hidden_dim=10, target_size=2, num_layers=1, embedding_dim=10, bidir=False).to(device)
    model = LstmEncDeltaStackedVel(lstm_hidden_dim=30, target_size=2, num_layers=1, embedding_dim=10, bidir=False).to(device)

    # model = LSTM_enc_delta_wo_emb(10, 2, embedding_dim=10).to(device)

    train_pose_vel(model, training_generator, test_generator, num_epochs=100, device=device, lr=0.002, limit=200)
