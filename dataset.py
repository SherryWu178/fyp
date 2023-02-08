import numpy as np
import config
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image


class MapDataset(Dataset):
    def __init__(self, root_dir, component = 'w', dataset_type = "train"):
        self.root_dir = root_dir
        self.component = component
        self.dataset_type = dataset_type
        self.train_inputs = np.load(self.root_dir + '/urban_input_train.npz')['arr_0']
        self.val_inputs = np.load(self.root_dir + '/urban_input_valid.npz')['arr_0']
        self.test_inputs = np.load(self.root_dir + '/urban_input_test.npz')['arr_0']   

        if self.component == 'u':
            self.train_labels = np.load(self.root_dir + '/urban_u_train.npz')['arr_0']
            self.val_labels = np.load(self.root_dir + '/urban_u_valid.npz')['arr_0']
            self.test_labels = np.load(self.root_dir + '/urban_u_test.npz')['arr_0']    

        elif self.component == 'v':
            self.train_labels = np.load(self.root_dir + '/urban_v_train.npz')['arr_0']
            self.val_labels = np.load(self.root_dir + '/urban_v_valid.npz')['arr_0']
            self.test_labels = np.load(self.root_dir + '/urban_v_test.npz')['arr_0']    

        elif self.component == 'w':
            self.train_labels = np.load(self.root_dir + '/urban_w_train.npz')['arr_0']
            self.val_labels = np.load(self.root_dir + '/urban_w_valid.npz')['arr_0']
            self.test_labels = np.load(self.root_dir + '/urban_w_test.npz')['arr_0']          

    def __len__(self):
        return len(np.load(self.root_dir + '/urban_w_train.npz')['arr_0'])

    def __getitem__(self, index):
        # load input layout
        print(index)

        if self.dataset_type == 'train':
            input_image = self.train_inputs[index]
            target_image = self.train_labels[index]

        elif self.dataset_type == 'valid':
            input_image = self.valid_inputs[index]
            target_image = self.valid_labels[index] 

        elif self.dataset_type == 'test':
            input_image = self.test_inputs[index]
            target_image = self.test_labels[index]

        input_image = input_image.reshape((256,256))
        target_image = target_image.reshape((256,256))
        print("input_image dimension", input_image.shape)
        print("target_image dimension", target_image.shape)
        return input_image, target_image


if __name__ == "__main__":
    dataset = MapDataset("/Users/sherrywu1999/Desktop/FYP/astar_data/urban_plane_data")
    loader = DataLoader(dataset, batch_size=10)
    print("init successfully")
    for x, y in loader:
        print(x.shape)
        print(x)
        print(y)
        break
        # save_image(x, "testx.png")
        # save_image(y, "testy.png")
        # import sys
        # sys.exit()