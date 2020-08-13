import tensorflow as tf
import numpy as np

def lr_schedule(a0,M,size):
    T=M*size
    def a(t,a0,T,M):
        return a0/2*(np.cos((np.pi*np.mod(t-1,T/M))/(T/M))+1)
    lr=[a(t,a0,T,M) for t in range(1,T+1)]
    index=[t-1  for t in range(1,T+1) if t % size ==0]
    return lr,index

class CyclicLR(tf.keras.callbacks.Callback):
    def __init__(self,a0,M,size):
        super(CyclicLR,self).__init__()
        self.weight_list=[]
        self.running_weights=None
        self.lr_list,self.lr_index=lr_schedule(a0,M=M,size=size)
        self.n_epochs=M*size

    def on_epoch_begin(self,epoch,logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        logs = logs or {}
        tf.keras.backend.set_value(self.model.optimizer.lr, self.lr_list[epoch])
        print('\nLearning rate set to %6.5f at epoch %d' % (self.lr_list[epoch],epoch))

    def on_epoch_end(self,epoch,logs=None):

        self.running_weights=self.get_weights()

        if epoch in self.lr_index:
            self.weight_list.append(self.running_weights)
            print("\nSaving weight at epoch %d with val_loss: %4.3f" % (epoch,logs['val_loss']))
    def get_weights(self):
        return [w.numpy() for w in self.model.weights]




















