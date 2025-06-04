import os
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import pickle

class LoadDataset(Dataset):
    
    def __init__(self, training_dir) -> None:
        super().__init__()

        self.training_dir = training_dir

        self.files = os.listdir(training_dir)
    

    def __len__(self):
        return len(self.files) // 2
    
    def load_images_stack(self):
    
        input_files = sorted([f for f in os.listdir(self.training_dir) if 'input' in f])
        
        return input_files
    
    def __getitem__(self, index):
        
        files = self.load_images_stack()
        
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

    dataloader = LoadDataset("data/spatial_data/training_data")

    input_vector, depth_vector = dataloader.__getitem__(2)

