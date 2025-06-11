import os
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import pickle
import random
import shutil 


class LoadDataset(Dataset):
    
    def __init__(self, training_dir) -> None:
        super().__init__()

        self.training_dir = training_dir
        self.input_files = self.load_images_stack()


    def __len__(self):
        return len(self.input_files) // 2
    
    def load_images_stack(self):
    
        input_files = sorted([f for f in os.listdir(self.training_dir) if 'input' in f])

        # input_files_negative = sorted([f for f in os.listdir(self.training_dir.replace("positive", "negative")) if 'input' in f])
        
        # random_items = random.sample(input_files_negative, len(input_files))

        # for file in random_items:
        #     shutil.move(
        #         os.path.join(self.training_dir.replace("positive", "negative"), file),
        #         os.path.join(self.training_dir, file),
        #     )
        #     shutil.move(
        #         os.path.join(self.training_dir.replace("positive", "negative"), file.replace("input", "depth")),
        #         os.path.join(self.training_dir, file.replace("input", "depth")),
        #     )
        # sdf

        return input_files
    
    def __getitem__(self, index):
        
        files = self.input_files
        
        input_vector_path = files[index]
        depth_vector_path = input_vector_path.replace('input_vector', 'depth_vector')

        file = open(os.path.join(self.training_dir, depth_vector_path), 'rb')
        depth_vector = pickle.load(file)
        file.close()

        file = open(os.path.join(self.training_dir, input_vector_path), 'rb')
        input_vector = pickle.load(file)
        file.close()
        return input_vector, depth_vector

if __name__ == "__main__":

    dataloader = LoadDataset("data/spatial_data/train_data/training_data_44/positive")

    input_vector, depth_vector = dataloader.__getitem__(2)

