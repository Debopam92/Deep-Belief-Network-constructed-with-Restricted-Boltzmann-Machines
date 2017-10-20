import os

import numpy

import theano
import theano.tensor as T

def unpickle(file):
  import cPickle
  fo = open(file, 'rb')
  dict = cPickle.load(fo)
  fo.close()
  return dict

def shared_dataset(data_xy, borrow=True):
  data_x, data_y = data_xy
  shared_x = theano.shared(numpy.asarray(data_x,
                                         dtype=theano.config.floatX),
                                         borrow=borrow)
  shared_y = theano.shared(numpy.asarray(data_y,
                                         dtype=theano.config.floatX),
                                         borrow=borrow)
  return shared_x, T.cast(shared_y, 'int32')

def load_data_cifar():
  dirpath = os.path.join(os.path.split(__file__)[0], 'cifar-10-batches-py')

  file = os.path.join(dirpath, 'data_batch_1')
  train_sets_1 = unpickle(file)

  file = os.path.join(dirpath, 'data_batch_2')
  train_sets_2 = unpickle(file)

  #file = os.path.join(dirpath, 'data_batch_3')
  #train_sets_3 = unpickle(file)

  #file = os.path.join(dirpath, 'data_batch_4')
  #train_sets_4 = unpickle(file)

  file = os.path.join(dirpath, 'data_batch_5')
  valid_sets = unpickle(file)

  #file = os.path.join(dirpath, 'test_batch')
  #test_sets = unpickle(file)
 
  train_sets_data = numpy.concatenate((train_sets_1['data'], train_sets_2['data']), axis=0)
  train_sets_labels = numpy.concatenate((train_sets_1['labels'], train_sets_2['labels']), axis=0)

  train_set_x, train_set_y = shared_dataset((train_sets_data / 256.0, train_sets_labels))
  valid_set_x, valid_set_y = shared_dataset((valid_sets['data'] / 256.0, valid_sets['labels']))
  #test_set_x, test_set_y = shared_dataset((test_sets['data'] / 256.0, test_sets['labels']))

  return [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (valid_set_x, valid_set_y)]


if __name__ == '__main__':
  load_data()

