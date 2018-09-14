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


class MalenovTestDataset(Dataset):
    def __init__(self, seismic, inline, cube_size):
        self.seismic = seismic
        self.cube_size = cube_size

        self.indices = []
        self.amplitudes = []
        for i in range(self.seismic.shape[2]):
            for j in range(self.seismic.shape[3]):
                if cube_size <= i < seismic.shape[2]-cube_size and cube_size <= j < seismic.shape[3]-cube_size:
                    idx = [inline, i, j]
                    self.indices.append(idx)
                    self.amplitudes.append(self.seismic[:, idx[0], idx[1], idx[2]])
        self.count = len(self.indices)
        
    def __getitem__(self, index):
        idx = self.indices[index]
        img = self.seismic[:, idx[0]-self.cube_size:idx[0]+self.cube_size+1, 
                            idx[1]-self.cube_size:idx[1]+self.cube_size+1,
                            idx[2]-self.cube_size:idx[2]+self.cube_size+1]
        return img

    def __len__(self):
        return self.count # of how many examples(images?) you have