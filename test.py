# Lab 11 MNIST and Convolutional Neural Network
import tensorflow as tf
from mnist_cnn import MnistCNN


class XXX (MnistCNN):
    def init_network(self):
        self.set_placeholder(784, 10, 28, 28)

        # 1, 2
        L1 = self.convolution_layer(self.X_2d, 3, 3, 1, 32, 1, 1)
        L1 = self.relu(L1)
        L1_maxpool = self.max_pool(L1, 2, 2, 2, 2)

        #3, 4
        L2 = self.convolution_layer(L1_maxpool, 3, 3, 32, 64, 1, 1)
        L2 = self.relu(L2)
        L2_maxpool = self.max_pool(L2, 2, 2, 2, 2)

        # 5
        reshaped = tf.reshape(L2_maxpool, [-1, 7*7*64])
        hypothesis = self.fully_connected_layer(reshaped, 7*7*64, 10, 'input_l')
        self.set_hypothesis(hypothesis)

        self.set_cost_function()
        self.set_optimizer(0.001)


bob = XXX()
bob.learn_mnist(15, 100)
bob.evaluate()
bob.classify_random()
bob.show_error()
