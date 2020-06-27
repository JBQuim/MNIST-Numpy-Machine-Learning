import numpy as np
import json
import sys
import random
from scipy.ndimage import rotate
import os
from pathlib import Path
from collections import Counter

def sigma(z):
    return 1 / (1 + np.exp(-z))


def sigmaPrime(z):
    s = sigma(z)
    return s * (1 - s)


class QuadraticCost:
    @staticmethod
    def getCost(batchLabels, activations):
        # np.linalg.norm finds the square root of the sum of the squares
        return 0.5 * np.linalg.norm(activations - batchLabels) ** 2

    @staticmethod
    def lastLayerError(batchLabels, activations, activationDerivatives):
        return (activations - batchLabels) * activationDerivatives


class CrossEntropyCost:
    @staticmethod
    def getCost(batchLabels, activations):
        return np.sum(
            np.nan_to_num(-batchLabels * np.log(activations) - (1 - batchLabels) * np.log(batchLabels - activations)))

    @staticmethod
    def lastLayerError(batchLabels, activations, activationDerivatives):
        # accepts 3 arguments for compatibility with the quadratic cost
        return activations - batchLabels


class NeuralNetwork:
    def __init__(self, layerSizes, cost=CrossEntropyCost):
        self.imageShape = [layerSizes[0]**0.5]*2

        self.layerCount = len(layerSizes)
        self.layerSizes = layerSizes
        self.matrixDims = [(i, j) for i, j in zip(layerSizes[1:], layerSizes[:-1])]

        self.biases = [np.random.randn(layerSize, 1) for layerSize in self.layerSizes[1:]]
        self.weights = [np.random.standard_normal(shape) / shape[1] ** 0.5 for shape in self.matrixDims]

        self.cost = cost

    def __repr__(self):
        string = "Network with layers of sizes {} \nCost function: {}".format(self.layerSizes, (self.cost).__name__)
        return string

    @staticmethod
    def unison_shuffled_copies(a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]

    def getActivation(self, a):
        for w, b in zip(self.weights, self.biases):
            a = sigma(np.dot(w, a) + b)
        return a

    def getPrediction(self, image):
        return np.argmax(self.getActivation(image), axis=0)[0]

    def getAccuracy(self, images, labels):
        predictions = np.array([self.getPrediction(image) for image in images])
        labels = np.array([np.argmax(label) for label in labels])
        correct = np.sum(labels == predictions)
        return correct

    def randRotateBatch(self, batch):
        randomNums = np.random.random_sample(batch.size)
        newBatch = np.array([rotate(image.reshape(self.imageShape), angle).reshape(self.layerSizes[0], 1) for image, angle in zip(batch, randomNums)])
        return newBatch

    def forwardPropagate(self, batch_a, batchSize):

        layerActivations = [np.zeros((layerSize, batchSize)) for layerSize in self.layerSizes]
        layerActivations[0] = batch_a

        layerSigmaDerivatives = [np.zeros((layerSize, batchSize)) for layerSize in self.layerSizes[1:]]

        for layer, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(w, layerActivations[layer]) + b
            layerActivations[layer + 1] = sigma(z)
            layerSigmaDerivatives[layer] = sigmaPrime(z)
        return layerActivations, layerSigmaDerivatives

    def backPropagate(self, activations, batchLabels, activationDerivatives, batchSize):
        nablaB = [np.zeros(bias.shape) for bias in self.biases]
        nablaW = [np.zeros(weight.shape) for weight in self.weights]
        layerErrors = [np.zeros((layerSize, batchSize)) for layerSize in self.layerSizes[1:]]

        layerErrors[-1] = (self.cost).lastLayerError(batchLabels, activations[-1], activationDerivatives[-1])
        nablaB[-1] = np.sum(layerErrors[-1], axis=1).reshape((self.layerSizes[-1], 1))
        nablaW[-1] = np.dot(layerErrors[-1], activations[-2].T)

        for i in range(-2, -self.layerCount, -1):
            layerErrors[i] = np.dot(self.weights[i + 1].T, layerErrors[i + 1]) * activationDerivatives[i]
            nablaB[i] = np.sum(layerErrors[i], axis=1).reshape((self.layerSizes[i], 1))
            nablaW[i] = np.dot(layerErrors[i], activations[i - 1].T)

        return nablaB, nablaW

    def updateBatch(self, batchImages, batchLabels, eta, lmbda, batchSize, N):
        activations, activationDerivatives = self.forwardPropagate(batchImages, batchSize)
        nablaBiases, nablaWeigths = self.backPropagate(activations, batchLabels, activationDerivatives, batchSize)

        self.biases = [b - dB * eta / batchSize for b, dB in zip(self.biases, nablaBiases)]
        self.weights = [(1 - eta * (lmbda / N)) * w - dW * eta / batchSize for w, dW in zip(self.weights, nablaWeigths)]

    def stochasticGradientDescent(self, trainingImages, trainingLabels, epochs, batchSize, eta, lmbda, testImages=None,
                                  testLabels=None):
        N = len(trainingImages)

        for k in range(epochs):
            trainingImages, trainingLabels = self.unison_shuffled_copies(trainingImages, trainingLabels)

            batchesImages = np.array([trainingImages[i:i + batchSize].reshape(batchSize, self.layerSizes[0]).T for i in
                                      range(0, N, batchSize)])
            batchesLabels = np.array([trainingLabels[i:i + batchSize].reshape(batchSize, self.layerSizes[-1]).T for i in
                                      range(0, N, batchSize)])
            for batchImages, batchLabels in zip(batchesImages, batchesLabels):
                self.updateBatch(batchImages, batchLabels, eta, lmbda, batchSize, N)

            if not (testImages is None or testLabels is None):
                assert len(testImages) == len(testLabels)
                accurate = self.getAccuracy(testImages, testLabels)
                print("Epoch {}: {}/{}".format(k, accurate, len(testImages)))
            else:
                print("Epoch {} completed".format(k))

    def save(self, filename):
        data = {"layerSizes": self.layerSizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        with open(filename, "w") as outfile:
            json.dump(data, outfile)


class Ensemble:
    def __init__(self, amount):
        self.amount = amount
        self.neuralNets = []

    def initNetworks(self, layerSizes, cost=CrossEntropyCost):
        self.neuralNets = [NeuralNetwork(layerSizes, cost) for _ in range(self.amount)]

    def __repr__(self):
        string = "Ensemble of {} networks with types: \n".format(self.amount)
        for i, net in enumerate(self.neuralNets):
            string += "{}: {}".format(i, repr(net))
            string += "\n"
        return string

    def trainAll(self, *args):
        for k, net in enumerate(self.neuralNets):
            net.stochasticGradientDescent(*args)
            print("Training of network {} completed".format(k))

    def vote(self, images):
        decisions = np.full(len(images), np.nan)
        for i, image in enumerate(images):
            outcomeList = [net.getPrediction(image) for net in self.neuralNets]
            decisions[i] = Counter(outcomeList).most_common(1)[0][0]
        return decisions

    def getAccuracy(self, images, labels):
        predictions = self.vote(images)
        labels = np.array([np.argmax(label) for label in labels])
        correct = np.sum(labels == predictions)
        return correct

    def save(self, path):
        if not os.path.isdir(path):
            os.mkdir(path)

        for i, net in enumerate(self.neuralNets):
            net.save("{}/NeuralNet{}".format(path, i))

    def load(self, directory):
        self.neuralNets = [loadNetwork("{}/NeuralNet{}".format(directory, i)) for i in range(self.amount)]


def loadNetwork(filename):
    with open(filename, "r") as inputfile:
        data = json.load(inputfile)
    cost = getattr(sys.modules[__name__], data["cost"])
    net = NeuralNetwork(data["layerSizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net
