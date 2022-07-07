'''
This module contains a self-contained class for loading the training
and test sets from the CIFAR-10 data batch files (stored as pickle files).
It was written for the unit CITS5508 Machine Learning.

Author: Du Huynh
Date created: April 19th 2021
Last modified: April 19th 2021

Copyright Department of Computer Science and Software Engineering
The University of Western Australia
'''
import numpy as np
import pickle

class DataLoader:
    @staticmethod
    def __load_pickle_file(filename):
        '''
        This function reads the given pickle file in the current directory
        and returns a dictionary object.
        :param filename - the pickle file name.
        '''
        with open(filename, 'rb') as f:
            dict = pickle.load(f, encoding='bytes')
            return dict  

    @staticmethod
    def load_batch(filename_prefix, Nbatches=5,
                   height=32, width=32, Nchannels=3):
        '''
        This function reads the Nbatches files and returns two numpy arrays X and
        y. To save memory space, the type of X is set to 'float32' (single
        precision floating point numbers) and the type of y is set to
        uint8. This is the same data type as used in the MNIST dataset, so it
        should be fine.

        @param filename_prefix (type: str): this should be a string, e.g.,
            'data_batch' or 'test_batch'. If Nbatches=1, then filename_prefix
            is assumed to be the single pickle file name; otherwise if
            Nbatches > 1, then "_1", "_2", etc, would be appended to
            filename_prefix to form the pickle file names.
        @param Nbatches (type: int; default=5): the number of batch files to
            read. Note that the batch index starts at 1.
        @param height (type: int; default=32): the height of each image in the
            data batch.
        @param width (type: int; default=32): the width of each image in the
            data batch.
        @param Nchannels (type: int; default=3): the number of channels of
            each image in the data batch.
        @return X (type: float32): this would be a N x height x width x Nchannels
            tensor, where N is the total number of instances in the data batch
            files. The pixel values are normalized to the range 0..1.
        @return y (type: uint8): this would be a numpy array containing N
            items of type uint8, which are just the class IDs. One should
            check the website of the dataset so as to correctly map the
            class IDs to the class names.
        '''

        if Nbatches == 1:
            batch = DataLoader.__load_pickle_file(filename_prefix)
            X = np.rollaxis(np.reshape(batch[b'data'], (-1,Nchannels,height,width)),
                            1, 4).astype('float32')
            y = np.array(batch[b'labels'], dtype='uint8')
        else:
            batch_no = range(1, Nbatches+1)  # the batch numbers start at 1
            # Read all the batch pickle files
            batch = [DataLoader.__load_pickle_file(filename_prefix+'_'+str(b)) for b in batch_no]

            batch_size = [len(batch[i][b'labels']) for i in range(Nbatches)]
            dataset_size = np.sum(batch_size)
    
            X = np.zeros((dataset_size, height, width, Nchannels), dtype='float32')
            y = np.zeros(dataset_size, dtype='uint8')
            loc = 0
            for i in range(Nbatches):
                X[loc:(loc+batch_size[i])] = np.rollaxis(
                    np.reshape(batch[i][b'data'], (-1,Nchannels,height,width)), 1, 4)
                y[loc:(loc+batch_size[i])] = batch[i][b'labels']
                loc += batch_size[i]
        return X/255, y
