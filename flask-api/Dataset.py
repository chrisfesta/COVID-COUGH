from tensorflow.keras import utils
import numpy as np


class CustomDataset(utils.Sequence):
    def __init__(self, data_x, labels, batch_size=128, shuffle=False, n_classes=1):
        self.batch_size = batch_size
        self.features = data_x
        self.dim = (self.features.shape[1], self.features.shape[2])
        self.n_channels = self.features.shape[3]
        self.labels = labels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.features) / self.batch_size))

    def __getitem__(self, index):
        # returns one batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # generate data
        X, y = self.__data_generation(indexes)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.features))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        for i, ID in enumerate(indexes):
            X[i, ] = self.features[ID]
            y[i, ] = self.labels[ID]

        return X, utils.to_categorical(y, num_classes=self.n_classes)


class CustomPipeline(utils.Sequence):
    def __init__(self, data_x, data_y, batch_size=48, shuffle=False, n_classes=1):
        self.features = data_x
        self.labels = data_y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_features = self.features.shape[1]
        self.n_classes = n_classes
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.features) / self.batch_size))

    def __getitem__(self, index):
        # returns one batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # generate data
        X, y = self.__data_generation(indexes)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.features))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        X = np.empty((self.batch_size, self.n_features))
        y = np.empty((self.batch_size), dtype=int)

        for i, ID in enumerate(indexes):
            X[i, ] = self.features[ID]
            y[i, ] = self.labels[ID]

        return X, utils.to_categorical(y, num_classes=self.n_classes)


class SingleInputGenerator(utils.Sequence):
    """
    Wrapper of two generators for the combined input model
    """

    def __init__(self, X1, Y, batch_size, target_size=(128, 128), n_classes=1, shuffle=False):
        self.genX1 = CustomDataset(
            X1, Y, batch_size=batch_size, shuffle=shuffle, n_classes=n_classes)
        self.n_classes = n_classes

    def __len__(self):
        return self.genX1.__len__()

    def get_shapes(self):
        return (*self.genX1.dim, self.genX1.n_channels)

    def get_n_classes(self):
        return self.n_classes

    def __getitem__(self, index):
        X1_batch, Y_batch = self.genX1.__getitem__(index)
        X_batch = [X1_batch]
        return X_batch, Y_batch
