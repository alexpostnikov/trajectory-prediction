from torch.utils.data import Dataset
import os
import torch
import dill
from typing import Union, List


class Dataset_from_pkl(Dataset):
    """
        Class for loading data in torch format from preprocessed pkl
        files with pedestrian poses, velocities and accelerations
    """

    def __init__(self, data_folder: str, data_files: Union[str, List[str]] = "all",
                 train: bool = True, test: bool = False, validate: bool = False):
        """
        :param data_folder: path to folder with preprocessed pkl files
        :param data_files: list of files to be used or "all"
        :param train: if data_files is all, if train is false -> all *train*.pkl files will be ignored
        :param test: if data_files is all, if train is false -> all *test*.pkl files will be ignored
        :param validate: if data_files is all, if train is false -> all *val*.pkl files will be ignored
        """

        super().__init__()
        self.train_dataset = torch.tensor([])
        file_list = []
        if "all" not in data_files:
            file_list = data_files
        else:
            dd = os.listdir(data_folder)
            for file in dd:
                if train and "train" in file:
                    file_list.append(file)
                if test and "test" in file:
                    file_list.append(file)
                if validate and "val" in file:
                    file_list.append(file)
        data_dict = {}
        for x in file_list:
            data_dict[x] = data_folder + "/" + x
        self.data = {}
        for file_name in data_dict.keys():
            with open(data_dict[file_name], 'rb') as f:
                print("loading " + file_name)
                self.data[file_name] = dill.load(f)
                for i in range(len(self.data[file_name])):
                    self.data[file_name][i] = torch.cat(self.data[file_name][i])


        self.dataset_indeces = {}
        self.data_length = 0

        for key in self.data.keys():
            for id, sub_dataset in enumerate(self.data[key]):
                self.data_length += len(sub_dataset) - 20
                self.dataset_indeces[self.data_length] = [key, id]


    def find_next_data_by_index(self, index, dataset):

        self_data = dataset[index]
        data = []
        data.append(self_data)

        self_index = self_data[0].item()

        c = 1
        last_seen_timestamp = dataset[index][1].item()
        while index + c < len(dataset):

            if dataset[index + c][0].item() == self_index:
                data.append(dataset[index + c])
                last_seen_timestamp = dataset[index + c][1].item()

            if dataset[index + c][1].item() - last_seen_timestamp > 1.0:
                break
            c += 1
            if len(data) >= 20:
                break
        data = torch.stack(data)
        extended = torch.zeros(20, 14)
        extended[0:len(data)] = data
        return extended, last_seen_timestamp

    def get_neighbours_history(self, neighbours_indexes, start_index, end_index, data, start_timestamp):
        n_history_tensor = torch.zeros(len(neighbours_indexes), 20, 14)
        n_history = [torch.zeros(20, 14) for i in neighbours_indexes]
        for i in range(end_index - start_index + 1):
            person_id = data[start_index+i][0].item()
            timestamp = int(data[start_index + i][1].item())
            if person_id in neighbours_indexes:
                delta_ts = timestamp - start_timestamp
                n_history[neighbours_indexes.index(person_id)][delta_ts] = data[start_index + i]
        if len(n_history) != 0:
            n_history_tensor = torch.stack(n_history)
        return n_history_tensor

    def get_peds(self, index, dataset: List):
        """
        return stacked torch tensor of scene in specified timestamps from specified dataset.
        if at any given timestamp person is not found at dataset, but later (previously) will appear,
        that its tensor data is tensor of zeros.
        :param start: timestamp start
        :param end:  timestamp end
        :param dataset: list of data. shapesa are: 1: timestamp 1: num_peds, 2: RFU, 3: data np.array of 8 floats
        :return: torch tensor of shape : end-start, max_num_peds, , 20 , 8
        """
        self_data = dataset[index]
        start = int(self_data[1].item())
        self_index = self_data[0].item()

        person_history, end = self.find_next_data_by_index(index, dataset)
        # end = int(person_history[-1, 1].item())
        neighbours_indexes = set()
        counter = 0
        start_index = index
        end_index = index
        while index - counter >= 0:
            neighbours_indexes.add(dataset[index - counter][0].item())
            start_index = index - counter
            counter += 1
            if dataset[index - counter][1] != start:
                break

        counter = 0
        while dataset[index + counter][1] <= end:
            neighbours_indexes.add(dataset[index + counter][0].item())
            end_index = index + counter
            counter += 1
            if index + counter >= len(dataset):
                break
        pass
        neighbours_indexes.remove(self_index)
        neighbours_history = self.get_neighbours_history(list(neighbours_indexes), start_index, end_index, dataset, start)

        return person_history, neighbours_history

    def get_ped_data_in_time(self, start: int, end: int, dataset: List):
        """
        return stacked torch tensor of scene in specified timestamps from specified dataset.
        if at any given timestamp person is not found at dataset, but later (previously) will appear,
        that its tensor data is tensor of zeros.
        :param start: timestamp start
        :param end:  timestamp end
        :param dataset: list of data. shapesa are: 1: timestamp 1: num_peds, 2: RFU, 3: data np.array of 8 floats
        :return: torch tensor of shape : end-start, max_num_peds, , 20 , 8
        """
        indexes = self.get_peds_indexes_in_range_timestamps(start, end, dataset)

        max_num_of_peds = 0
        for key in indexes.keys():
            if len(indexes[key]) > max_num_of_peds:
                max_num_of_peds = len(indexes[key])
        prepared_data = 0 * torch.ones(end - start, max_num_of_peds, 20, 14)
        for start_timestamp in range(start, end):
            for duration in range(0, 20):
                for ped in dataset[start_timestamp + duration]:
                    if type(ped) == torch.Tensor:
                        ped_id = indexes[start_timestamp].index(ped[0])
                        prepared_data[start_timestamp - start][ped_id][duration] = ped.clone()
                    else:
                        ped_id = indexes[start_timestamp].index(ped[0][0])
                        prepared_data[start_timestamp - start][ped_id][duration][0:len(ped[0])] = torch.tensor(ped[0])

        return prepared_data

    def limit_len(self, new_len):
        self.data_length = new_len


    def get_peds_indexes_in_range_timestamps(self, start: int, end: int, dataset: List):
        """
        :param start: timestamp start
        :param end:  timestamp end
        :param dataset: list of data. shapes are: 0: timestamp 1: num_peds, 2: RFU, 3: data np.array of 8 floats
        :return: dict of  predestrians ids at each scene (one scene is 20 timestamps)
        """
        indexes = []
        for time_start in range(start, end):

            for duration in range(0, 20):
                peoples = dataset[time_start + duration]
                indexes += list(self.get_peds_indexes_in_timestamp(peoples))

        return set(indexes)

    def get_peds_indexes_in_timestamp(self, person: List):
        """
        :param person:
        :return: Set of  peds ids at scene
        """
        indexes = []

        if type(person) == torch.Tensor:
            indexes.append(float(person[0]))
        else:
            indexes.append(float(person[0][0]))
        return set(indexes)

    def get_dataset_from_index(self, data_index: int):
        """
        given index return dataset name and sub_dataset id, corresponding index in sub_dataset
        :param data_index: data sample number
        :return: file_name, sub_dataset id, corresponding index in sub_dataset
        """
        upper_bounds = list(self.dataset_indeces.keys())
        upper_bounds.append(0)
        upper_bounds.sort()
        index = [upper_bound > data_index for upper_bound in upper_bounds].index(True)
        index_in_sub_dataset = data_index - upper_bounds[index-1]
        return self.dataset_indeces[upper_bounds[index]], index_in_sub_dataset

    def __len__(self):
        return self.data_length

    def __getitem__(self, index):

        [file, sub_dataset], index_in_sub_dataset = self.get_dataset_from_index(index)
        self.packed_data = []  # num_samples * 20 * numped * 8
        data = self.get_peds(index_in_sub_dataset, self.data[file][sub_dataset])
        return data


def my_fn(data):
    node_hist = []
    neighbours = []
    for i in range(len(data)):
        node_hist.append(data[i][0])
        neighbours.append(data[i][1])
    node_hist = torch.stack(node_hist)
    return node_hist, neighbours

def is_filled(data):
    return not (data[:, 1] == 0).any().item()


if __name__ == "__main__":
    dataset = Dataset_from_pkl("/home/robot/repos/trajectory-prediction/processed_with_forces/", data_files=["eth_train.pkl", "zara2_test.pkl"])
    print(len(dataset))
    t = dataset[0]
    print(dataset[0][0].shape)

    # training_set = Dataset_from_pkl("/home/robot/repos/trajectories_pred/processed/", data_files=["eth_train.pkl"])
    training_generator = torch.utils.data.DataLoader(dataset, batch_size=512, collate_fn=my_fn)# , num_workers=10
    #
    import time
    start = time.time()
    for local_batch in training_generator:
        # print(local_batch[0].shape)
        # print(local_batch[0][-1, 0, 1])
        pass

    print(time.time() - start)
        # print(local_batch[1].shape)
        # for ped in range(local_batch.shape[1]):
        #     observed_pose = local_batch[0, ped, 0:8, :]
        #     if is_filled(observed_pose):
        #         print("ped: ", ped, "observed_pose: ", observed_pose.shape)
        #     else:
        #         print("unfilled")
        #         break


