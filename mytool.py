import sys
import random

def print_dot():
    sys.stdout.write('.')
    sys.stdout.flush()

def load_mnist():
    from tensorflow.examples.tutorials.mnist import input_data
    # Check out https://www.tensorflow.org/get_started/mnist/beginners for more information about the mnist dataset
    return input_data.read_data_sets("MNIST_data/", one_hot=True)

def get_random_int(max):
    return random.randint(0, max - 1)
