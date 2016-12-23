from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.structure import FullConnection
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.xml.networkwriter import NetworkWriter
from pybrain.tools.xml.networkreader import NetworkReader
import math
import os.path
import numpy as np
import random
import matplotlib.pyplot as plt



def quadro_equation(coeficients, array):
    #print("step 1")

    a = coeficients[0]
    b = coeficients[1]
    c = coeficients[2] - coeficients[3]
    discriminant = b*b - 4 * a*c
    #print(discriminant)
    if a == 0:
        return "bad a"
    try:
        sqrtDiscriminant = float(math.sqrt(discriminant))
    except Exception:
        return "bad sqrt"
    else:
        x1 = float((-b + sqrtDiscriminant)/(2 * a))
        #print(x1)
        x2 = float((-b - sqrtDiscriminant) / (2 * a))
        #print(x2)
        array[0] = x1
        array[1] = x2

        #print array
        return ""



def prepareDataSet(rangeStart,rangeEnd , counter):
    dataSet =  SupervisedDataSet(4, 2)
    #print dataSet
    #for i in range(0,10) :
    i = 0
    while i != counter :
        inputValues = np.array([random.randint(rangeStart,rangeEnd),random.randint(rangeStart,rangeEnd),random.randint(rangeStart,rangeEnd),random.randint(rangeStart,rangeEnd)])
        resultArray = np.array([float(0), float(0)])
        err = quadro_equation(inputValues, resultArray)
        if err == "":
            dataSet.addSample(inputValues, resultArray)
            #print "#" + str(i) + " Coefficients: " + str(inputValues)
            #print "#" + str(i) + " Results: " + str(resultArray) + " \n"
            #print "current dataset: " + str(dataSet)
            i+=1
    #print "Whole dataset: " + str(dataSet)
    return dataSet


def OurTrainEpoch(title, ourRange, counts, currentLearningrate, network, axis, plotNumber):
    errs = np.zeros((counts,1))
    ordinats = np.zeros((counts,1))

    dataSet = prepareDataSet(ourRange[0], ourRange[1], counts)
    trainer = BackpropTrainer(network, dataSet,learningrate=currentLearningrate)
    for i in range (0,counts):
        errs[i] = trainer.train()
        ordinats[i] = i

    plt.subplot(220 + plotNumber)
    plt.plot(ordinats, errs)
    plt.axis([0, counts, axis[0], axis[1]])
    plt.yscale('linear')
    plt.title(title)
    plt.grid(True)
    print errs


def neuroNetworkAlgorithm():
    # if (os.path.isfile('weatherlearned.csv')) :
    #     n = NetworkReader.readFrom('weatherlearned.csv')
    # else:
    n = FeedForwardNetwork()
    inLayer = LinearLayer(4)
    hiddenLayer1 = LinearLayer(12)
    hiddenLayer2 = LinearLayer(24)
    hiddenLayer3 = LinearLayer(48)
    hiddenLayer4 = LinearLayer(24)
    hiddenLayer5 = LinearLayer(12)
    outLayer = LinearLayer(2)

    n.addInputModule(inLayer)
    n.addModule(hiddenLayer1)
    n.addModule(hiddenLayer2)
    n.addModule(hiddenLayer3)
    n.addModule(hiddenLayer4)
    n.addModule(hiddenLayer5)
    n.addOutputModule(outLayer)

    in_to_hidden1 = FullConnection(inLayer, hiddenLayer1)
    hidde1_to_hidden2 = FullConnection(hiddenLayer1, hiddenLayer2)
    hidde2_to_hidden3 = FullConnection(hiddenLayer2, hiddenLayer3)
    hidde3_to_hidden4 = FullConnection(hiddenLayer3, hiddenLayer4)
    hidde4_to_hidden5 = FullConnection(hiddenLayer4, hiddenLayer5)
    hidden5_to_out = FullConnection(hiddenLayer5, outLayer)

    n.addConnection(in_to_hidden1)
    n.addConnection(hidde1_to_hidden2)
    n.addConnection(hidde2_to_hidden3)
    n.addConnection(hidde3_to_hidden4)
    n.addConnection(hidde4_to_hidden5)
    n.addConnection(hidden5_to_out)
    n.sortModules()
    n.reset()
    #print n.activate([1,1])

    OurTrainEpoch("first epoch",  [0,20],  200, 0.00000000001,   n , [0,100], 1)
    OurTrainEpoch("second epoch", [0,20], 200,  0.000000000001,  n , [0,0.5], 2)
    OurTrainEpoch("third epoch",  [0,20],  200, 0.0000000000001,  n , [0,2],   3)
    OurTrainEpoch("fourth epoch", [0, 20], 200, 0.00000000000001, n,  [0, 2],  4)



    plt.show()
    # NetworkWriter.writeToFile(n, 'weatherlearned.csv')
    # print n.activate([0,1])



if __name__ == '__main__':
    #prepareDataSet(0, 10, 10)
    neuroNetworkAlgorithm()
    # resultArray = np.array([float(0),float(0)])
    # res = quadro_equation(np.array([105,25,15,30]),resultArray)
    # if(res !=""):
    #     print res
    # else:
    #     print (resultArray)


