import os
import collections
import numpy as np

class Dataset():
    def __init__(self, data_name):
        self.data = np.load(data_name)
        self._num_examples = self.data.shape[0]
        self._dimension = self.data.shape[1]

        self._epochs_completed = 0
        self._index_in_epoch = 0

    def next_batch(self, batch_size, shuffle = True):
        start = self._index_in_epoch

        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self.data = self.data[perm0]

        if start + batch_size > self._num_examples:
            self._epochs_completed += 1

            rest_num_exmamples = self._num_examples - start
            rest_part = self.data[start:self._num_examples]

            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self.data = self.data[perm]
            start = 0
            self._index_in_epoch = batch_size - rest_num_exmamples
            end = self._index_in_epoch
            new_part = self.data[start:end]
            return np.concatenate((rest_part, new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self.data[start:end]

    @property
    def num_examples(self):
        return self._num_examples
        
    @property
    def dimension(self):
        return self._dimension

class DroppedDataset(Dataset):
    def __init__(self, idx_name, *args, **kwargs):
        super(DroppedDataset, self).__init__(*args, **kwargs)
        self.idx = np.load(idx_name) 

    def next_batch_with_idx(self, batch_size, shuffle = True):
        start = self._index_in_epoch

        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self.data = self.data[perm0]
            self.idx = self.idx[perm0]

        if start + batch_size > self._num_examples:
            self._epochs_completed += 1

            rest_num_exmamples = self._num_examples - start
            data_rest_part = self.data[start:self._num_examples]
            idx_rest_part = self.idx[start:self._num_examples]

            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self.data = self.data[perm]
                self.idx = self.idx[perm]
            start = 0
            self._index_in_epoch = batch_size - rest_num_exmamples
            end = self._index_in_epoch
            data_new_part = self.data[start:end]
            idx_new_part = self.idx[start:end]
            return np.concatenate((data_rest_part, data_new_part), axis=0), np.concatenate((idx_rest_part, idx_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self.data[start:end], self.idx[start:end]

def load_music_data():
    train_data_name = 'Music/data/train_data.npy'
    valid_data_name = 'Music/data/val_data.npy'
    #test_data_name = 'Music/data/test_data.npy'

    positive_dropped_test_data_name = 'Music/data/positive_dropped_test_data.npy'
    positive_dropped_test_idx_name = 'Music/data/positive_dropped_test_idx.npy'

    datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])

    #return datasets(train=Dataset(train_data_name), validation=Dataset(valid_data_name), test=Dataset(test_data_name))
    return datasets(train=Dataset(train_data_name), validation=Dataset(valid_data_name), test=DroppedDataset(data_name=positive_dropped_test_data_name, idx_name=positive_dropped_test_idx_name))


if __name__ == '__main__':

    data = np.arange(110).reshape(11, 10)
    np.save('test_utils_data.npy', data)

    dataset = Dataset('test_utils_data.npy')
    batch_size = 2
    total_batch = int(dataset.num_examples / batch_size)
    for i in range(total_batch + 1):
        batch_data = dataset.next_batch(batch_size)
        print(i, batch_data)

    os.remove('test_utils_data.npy')

    datasets = load_music_data()
