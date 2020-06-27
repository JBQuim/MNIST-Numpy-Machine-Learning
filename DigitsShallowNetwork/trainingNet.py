import numpy as np
import neuralNetwork as nnet
import time
from scipy.ndimage import rotate
import random
import matplotlib.pyplot as plt




with np.load('Dataset/mnist.npz') as data:
    trainingImages = data["training_images"]
    trainingLabels = data["training_labels"]
    testImages = data["test_images"]
    testLabels = data["test_labels"]
    validationImages = data["validation_images"]
    validationLabels = data["validation_labels"]

voting = nnet.Ensemble(9)
# voting.load("9Ensemble")
voting.initNetworks([784, 100, 10])
voting.trainAll(trainingImages, trainingLabels, 10, 10, 0.5, 0.1)
voting.save("9Ensemble")
print(voting.getAccuracy(testImages, testLabels))

"""""
nn = nnet.NeuralNetwork([784, 100, 10], nnet.CrossEntropyCost)
t1 = time.clock()
nn.stochasticGradientDescent(trainingImages, trainingLabels, 30, 10, 0.5, 0.1)
t2 = time.clock()


print(nn.getAccuracy(testImages, testLabels))
print("Time taken: {}m {}s".format(*divmod(int(t2-t1), 60)))

nn.save("neuralNet.txt")
"""