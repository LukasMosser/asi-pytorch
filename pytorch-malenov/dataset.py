from torch.utils.data.dataset import Dataset

class MalenovDataset(Dataset):
    def __init__(self, seismic, indices, labels, cube_size):
        self.seismic = seismic
        self.cube_size = cube_size
        self.indices = indices
        self.labels = labels
        self.count = len(self.labels)
        
    def __getitem__(self, index):
        idx = self.indices[index]
        img = self.seismic[:, idx[0]-self.cube_size:idx[0]+self.cube_size+1, 
                            idx[1]-self.cube_size:idx[1]+self.cube_size+1,
                            idx[2]-self.cube_size:idx[2]+self.cube_size+1]
        label = self.labels[index]
        return (img, label)

    def __len__(self):
        return self.count # of how many examples(images?) you have