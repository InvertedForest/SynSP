import numpy as np

class RandomIter():
    '''
    used to randomly iterate over multiple pytorch dataloaders.

    args:
        loaders: list of pytorch dataloaders
        crop: if True, the length of each loader is cropped to the mimimum length
    returns:
        iterator over the combined dataloader 
    '''
    def __init__(self, loaders: list, crop: bool=False):
        self.loaders = loaders
        self.num = len(loaders)
        self.crop = crop
        self.sub_length = [len(loader) for loader in self.loaders]
        if self.crop == True:
            min_length = min(self.sub_length)
            self.sub_length = [min_length for _ in self.loaders]
            print('these datasets are cropped to the length of the mimimum dataset')
        self.length = sum(self.sub_length)
        # self.__iter_init__()


    def __iter__(self):
        self.id = 0
        self.iter_loaders = [iter(loader) for loader in self.loaders]
        
        self.seq = [i for i in range(self.num) for j in range(self.sub_length[i])]
        np.random.shuffle(self.seq)
        return self

    @property
    def now_loader_id(self):
        # use after the iter
        return self.seq[self.id-1]

    def __len__(self):
        return self.length


    def __next__(self):
        if self.id == self.length:
            raise StopIteration
        # if self.id == 200:
        #     raise StopIteration
        data = next(self.iter_loaders[self.seq[self.id]])
        self.id += 1
        return data

