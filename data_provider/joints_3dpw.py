import numpy as np
import random
import pdb

class InputHandle:
    def __init__(self, input_param):
        self.paths = input_param['paths']
        self.num_paths = len(input_param['paths'])
        self.name = input_param['name']
        self.input_data_type = input_param.get('input_data_type', 'float32')
        self.output_data_type = input_param.get('output_data_type', 'float32')
        self.minibatch_size = input_param['minibatch_size']
        self.is_output_sequence = input_param['is_output_sequence']
        self.data = {}
        self.indices = {}
        self.current_position = 0
        self.current_batch_size = 0
        self.current_batch_indices = []
        self.input_length = input_param['input_length']
        self.output_length = input_param['sequence_length']-input_param['input_length']
        self.seq_length = input_param['sequence_length']
        self.load()

    def load(self):
        #print self.paths,'\n data:\n',self.paths[0]
        self.data = np.load(self.paths[0])+1
        
        
        #pdb.set_trace()
        print(self.data.shape)

    def total(self):
        return len(self.data)

    def begin(self, do_shuffle = True):
        self.indices = np.arange(self.total(),dtype="int32")
        if do_shuffle:
            random.shuffle(self.indices)
        self.current_position = 0
        if self.current_position + self.minibatch_size <= self.total():
            self.current_batch_size = self.minibatch_size
        else:
            self.current_batch_size = self.total() - self.current_position
        self.current_batch_indices = self.indices[
            self.current_position:self.current_position + self.current_batch_size]
        #self.current_input_length = max(self.data['clips'][0, ind, 1] for ind
        #                                in self.current_batch_indices)
        #self.current_output_length = max(self.data['clips'][1, ind, 1] for ind
        #                                 in self.current_batch_indices)

    def next(self):
        self.current_position += self.current_batch_size
        if self.no_batch_left():
            return None
        if self.current_position + self.minibatch_size <= self.total():
            self.current_batch_size = self.minibatch_size
        else:
            self.current_batch_size = self.total() - self.current_position
        self.current_batch_indices = self.indices[
            self.current_position:self.current_position + self.current_batch_size]
        #self.current_input_length = max(self.data['clips'][0, ind, 1] for ind
        #                                in self.current_batch_indices)
        #self.current_output_length = max(self.data['clips'][1, ind, 1] for ind
        #                                 in self.current_batch_indices)

    def no_batch_left(self):
        if self.current_position >= self.total() - self.current_batch_size:
            return True
        else:
            return False

    def input_batch(self):
        if self.no_batch_left():
            return None

        #pdb.set_trace()
        input_batch = np.zeros(
            (self.current_batch_size, self.seq_length,24,3)).astype(self.input_data_type)
        #input_batch = np.transpose(input_batch,(0,1,3,4,2))
        for i in range(self.current_batch_size):
            batch_ind = self.current_batch_indices[i]
            #begin = self.data['clips'][0, batch_ind, 0]
            #end = self.data['clips'][0, batch_ind, 0] + \
            #        self.data['clips'][0, batch_ind, 1]
            data_slice = self.data[batch_ind]
            data_slice = data_slice[0:self.seq_length]
            #print input_batch.shape,'\ndata_slice',data_slice.shape
            #data_slice = np.transpose(data_slice,(0,2,3,1))
            input_batch[i, :self.seq_length, :] = data_slice
        input_batch = input_batch.astype(self.input_data_type)
        return input_batch
    def get_batch(self):
        batch = self.input_batch()
        #output_seq = self.output_batch()
        #batch = np.concatenate((input_seq, output_seq), axis=1)
        return batch
