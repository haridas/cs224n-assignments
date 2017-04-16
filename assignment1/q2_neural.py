#!/usr/bin/env python
import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive
#from test_neural import gradcheck_naive


def forward_backward_prop(data, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network

    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.

    Arguments:
    data -- M x Dx matrix, where each row is a training example.
    labels -- M x Dy matrix, where each row is a one-hot vector.
    params -- Model parameters, these are unpacked for you.
    dimensions -- A tuple of input dimension, number of hidden units
                  and output dimension


    Network topology.

    Input layer ( 10 Neurons ) -> Hidden ( 5 Neurons ) -> Output ( 10 Neurons )

    W1 (10 x 5) + b1 (1 x 5)  = 55
    W2 (5 x 10) + b1 (1 x 10) = 60 
    Total params = 115.
    Backpropagation has to update total of 115 params.

    Cost fun is Cross entropy fun.
    """

    ### Unpack network parameters (do not modify)
    ofs = 0

    # input layer size, hidden layer size, output layer size.
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    ### YOUR CODE HERE: forward propagation
    # 20 x 10 * 10 x 5 => (20 x 5) + 1 x 5
    layer1_activation = sigmoid(data.dot(W1) + b1)

    # Usually the final layer shouldn'tapplied the non-linearity, as after
    # that there is no extra learning process, also some non-linearity changes
    # or transform the neuron output for further learning. As the output
    # layer is not going to do any of this, just do the linear transformation
    # done inside each output neurons.
    layer2_activation = layer1_activation.dot(W2) + b2
    model_pred = softmax(layer2_activation) # Squash to probabilities.

    # cost calculated using cross entropy function. Average wrt #samples.
    cost = - np.multiply(labels, np.log(model_pred)).sum() / labels.shape[0]
    ### END YOUR CODE

    ### YOUR CODE HERE: backward propagation
    ### END YOUR CODE

    # Derivative of the final layer(softmax) output wrt sigmoid neuron output
    d0 = (model_pred - labels) / labels.shape[0]


    # Derivative of sigmoid nueron output layer wrt previous weights.
    # Derivative wrt to W2s
    # 5 X 20 O 20 X 10 => 5 X 10
    gradW2 = np.dot(layer1_activation.transpose(), d0)
    # b2 shape is 1 x 10, d0 = N(20) x 10
    gradb2  = d0.sum(axis=0)

    # Derivative of sigmoid nueron output layer wrt previous weights.
    # Derivative wrt to W1s
    d1 = np.multiply(sigmoid_grad(layer1_activation), np.dot(d0, W2.transpose()))

    gradW1 = np.dot(data.transpose(), d1)
    gradb1 = d1.sum(axis=0)

    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
        gradW2.flatten(), gradb2.flatten()))

    return cost, grad


def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print "Running sanity check..."

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in xrange(N):
        labels[i, random.randint(0,dimensions[2]-1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    # All other variables are fixed, we are finding gradients for Weights, so
    # We only need to check the numerical check for that only keeping all other
    # variables constant.
    gradcheck_naive(lambda params: forward_backward_prop(data, labels, params, dimensions), params)


def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print "Running your sanity checks..."
    ### YOUR CODE HERE
    #raise NotImplementedError
    ### END YOUR CODE


if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()
