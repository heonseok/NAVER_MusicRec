import numpy as np
import os 
import sys

def drop_random_element_in_row(data, target):
    dropped_data = np.copy(data)
    dropped_indices = np.zeros(data.shape[0])

    for i in range(data.shape[0]):
        candidate_idx = np.where(data[i] == target)
        selected_idx = np.random.choice(candidate_idx[0], 1)[0]

        dropped_data[i][selected_idx] = 0
        dropped_indices[i] = selected_idx

    return dropped_data, dropped_indices


if __name__ == "__main__":

    if sys.argv[1] == 'dummy': 
        data = np.random.binomial(1, 0.7, [10, 10]) 
        dropped_data, dropped_idx = drop_random_element_in_row(data, 1)
        print(data)
        print(dropped_data)
        print(dropped_idx)

    elif sys.argv[1] == 'music':
        data_dir = 'Music/data'

        test_data_name = os.path.join(data_dir, 'test_data.npy')
        data = np.load(test_data_name)

        positive_dropped_data_name = os.path.join(data_dir, 'positive_dropped_test_data.npy')
        positive_dropped_idx_name = os.path.join(data_dir, 'positive_dropped_test_idx.npy')

        positive_dropped_data, positive_dropped_idx = drop_random_element_in_row(data, 1)
        np.save(positive_dropped_data_name, positive_dropped_data)
        np.save(positive_dropped_idx_name, positive_dropped_idx)

    #filename = 'test_utils_data.npy'
    #np.save(filename, data)
    #os.remove(filename)

