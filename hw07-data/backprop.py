import numpy as np
import matplotlib.pyplot as plt
import pdb
from mnist import MNIST
import sklearn.metrics as metrics
import abc
from enum import Enum

class Layer(Enum):
    CONVOLUTION = "ConvolutionLayer"
    MAXPOOLING = "MaxPoolingLayer"
    DENSE = "DenseLayer"

# Preprocess the input data into features
class MNISTPreprocessor(object):

    def __init__(self, f):
        # f is the file address containing the images
        self.f = f

    def load_data(self):
        # Load the data
        mndata = MNIST(self.f)
        X_train, labels_train = map(np.array, mndata.load_training())
        # The test labels are meaningless,
        # since you're replacing the official MNIST test set with our own test set
        X_test, _ = map(np.array, mndata.load_testing())
        # Remember to center and normalize the data...
        X_train, X_test = X_train/1.0, X_test/1.0
        self.center_normal(X_train)
        self.center_normal(X_test)

        #pdb.set_trace()

        # Shuffle the data and split the data into training and validation
        training, training_labels, validation, validation_labels = self.shuffle_and_split(X_train, labels_train)

        #pdb.set_trace()

        return training, training_labels, validation, validation_labels, X_test


    # Given the labels, make them into one hot encoding
    @staticmethod
    def one_hot(y):
        '''Convert categorical labels 0,1,2,....9 to standard basis vectors in R^{10} '''
        one_hot_vector = np.zeros((y.shape[0], len(np.unique(y))))
        i = 0
        while i < len(y):
            one_hot_vector[i][y[i]] = 1
            i += 1
        return one_hot_vector


    def center_normal(self, data):
        #pdb.set_trace()
        for i in range(data.shape[0]):
            entry = data[i]
            mean = np.mean(entry)
            std = np.std(entry)
            data[i] = (entry - mean)/std


    def shuffle_and_split(self, X_train, labels_train):
        #print labels_train
        training = []
        training_labels = []
        validation = []
        validation_labels = []
        l = []
        for i in range(X_train.shape[0]):
            l.append((X_train[i], labels_train[i]))

        np.random.shuffle(l)
        # training 
        for i in range(X_train.shape[0]):
            if i < (X_train.shape[0] * 9 / 10):
                training.append(l[i][0])
                training_labels.append(l[i][1])
            else:
                validation.append(l[i][0])
                validation_labels.append(l[i][1])


        #data_augmentation(training, training_labels)
        #print len(training)


        return np.array(training), np.array(training_labels), np.array(validation), np.array(validation_labels)


# Gradient descent optimization
# The learning rate is specified by eta
class GDOptimizer(object):
    def __init__(self, eta, gamma, epochs):
        self.eta = eta
        self.gamma = gamma # the decaying rate.
        self.epochs = epochs # how many epochs do we apply the decaying factor
        self.count = 0

    def initialize(self, layers):
        pass

    # This function performs one gradient descent step
    # layers is a list of dense layers in the network
    # g is a list of gradients going into each layer before the nonlinear activation
    # a is a list of of the activations of each node in the previous layer going 
    def update(self, layers, g, a):
        # decay the eta every pre-defined number of epochs
        if self.count != 0 and self.count % self.epochs == 0: 
            self.decay()
            print "epochs: " + str(self.count)
            print "learning rate: " + str(self.eta)

        m = a[0].shape[1]
        for layer, curGrad, curA in zip(layers, g, a):
            # TODO #################################################################################
            # Compute the gradients for layer.W and layer.b using the gradient for the output of the
            # layer curA and the gradient of the output curGrad
            # Use the gradients to update the weight and the bias for the layer
            # ######################################################################################
            if layer.getType() == Layer.DENSE or layer.getType() == Layer.MAXPOOLING:
                # Gradient of W
                dW = curGrad.dot(curA.T)
                layer.updateWeights(-self.eta/m * dW)
                # Gradient of b
                db = np.sum(curGrad,1).reshape(layer.b.shape)
                layer.updateBias(-self.eta/m * db)
            if layer.getType() == Layer.CONVOLUTION:
                dW = np.zeros(layer.W.shape)
                # reshape curA
                curA_reshaped = curA.reshape((layer.numNodesPrevW, layer.numNodesPrevL, curA.shape[1]))
                # rotate each layer of curA by 180
                for k in range(curA.shape[1]):
                    batch = curA_reshaped[:,:,k]
                    curA_reshaped[:,:,k] = np.rot90(batch,2)

                # reshape the current gradient, this will serve as the window of convolution
                _curGrad = curGrad.reshape((layer.numNodesW, layer.numNodesL, curGrad.shape[1]))
                step = layer.numNodesW
                margin = layer.window
                # Convolution
                for k in range(curA.shape[1]):
                    dW_k = np.zeros(layer.W.shape)
                    for i in range(margin):
                        for j in range(margin):
                            batch = curA_reshaped[i:i+step,j:j+step,k]
                            convolution = np.sum(np.multiply(_curGrad[:,:,k], batch))
                            # Bookkeeping the result
                            dW_k[i][j] = convolution
                    dW += dW_k

                layer.updateWeights(-self.eta/m * dW)

                # update d
                db = np.sum(curGrad,1).reshape(layer.b.shape)
                layer.updateBias(-self.eta/m * db)


        self.count += 1

    # This function allows us to decay the learning rate
    def decay(self):
        self.eta *= self.gamma

# Cost function used to compute prediction errors
class QuadraticCost(object):

    # Compute the squared error between the prediction yp and the observation y
    # This method should compute the cost per element such that the output is the
    # same shape as y and yp
    @staticmethod
    def fx(yp,y):
        # TODO #################################################################################
        # Implement me
        # ######################################################################################
        return 0.5 * np.square(y-yp)

    # Derivative of the cost function with respect to yp
    @staticmethod
    def dx(yp,y):
        # TODO #################################################################################
        # Implement me
        # ######################################################################################
        return yp - y

class CrossEntropyCost(object):
    # Compute the cross entropy cost between the prediction yp and the observation y
    # This method should compute the cost per element such that the output is the
    # same shape as y and yp
    @staticmethod
    def fx(yp,y):
        # TODO #################################################################################
        # Implement me
        # ######################################################################################
        return -1.0 * np.sum(np.multiply(y, np.log(yp)))

    # Derivative of the cost function with respect to yp
    @staticmethod
    def dx(yp,y):
        # This is NOT the correct derivative of the cross entropy. The gradient of 
        # Cross entropy must be taken together with the Softmax. Therefore, the 
        # gradient of the cross entropy is handled together with the gradient of the softmax
        return yp - y

# Sigmoid function fully implemented as an example
class SigmoidActivation(object):
    @staticmethod
    def fx(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def dx(z):
        return SigmoidActivation.fx(z) * (1 - SigmoidActivation.fx(z))
        
# Hyperbolic tangent function
class TanhActivation(object):

    # Compute tanh for each element in the input z
    @staticmethod
    def fx(z):
        # TODO #################################################################################
        # Implement me
        # ######################################################################################
        return np.tanh(z)

    # Compute the derivative of the tanh function with respect to z
    @staticmethod
    def dx(z):
        # TODO #################################################################################
        # Implement me
        # ######################################################################################
        return 1 - np.square(np.tanh(z))

# Rectified linear unit
class ReLUActivation(object):
    @staticmethod
    def fx(z):
        # TODO #################################################################################
        # Implement me
        # ######################################################################################
        return np.maximum(0,z)

    @staticmethod
    def dx(z):
        # TODO #################################################################################
        # Implement me
        # ######################################################################################
        return (z>0).astype('float')

# Linear activation
class LinearActivation(object):
    @staticmethod
    def fx(z):
        # TODO #################################################################################
        # Implement me
        # ######################################################################################
        return z

    @staticmethod
    def dx(z):
        # TODO #################################################################################
        # Implement me
        # ######################################################################################
        return np.ones(z.shape)

# Softmax activation
class SoftmaxActivation(object):
    @staticmethod
    def fx(z):
        # TODO #################################################################################
        # Implement me
        # ######################################################################################
        """
        z_activated = np.zeros(z.shape)
        denominator = 0.0
        b = 0.0
        for i in range(z.shape[0]):
            b = max(z[i][0], b)
        for i in range(z.shape[0]):
            denominator += float(np.exp(z[i][0] - b))
        for i in range(z.shape[0]):
            numerator = float(np.exp(z[i][0] - b))
            z_activated[i][0] = float(numerator / denominator)
        return z_activated
        """
        exps = np.exp(z - np.max(z))
        activated = exps / np.sum(exps)
        # Lower bond the value
        for i in range(activated.shape[0]):
            for j in range(activated.shape[1]):
                if activated[i][j] < 0.0000000001:
                    activated[i][j] = 0.0000000001
        return activated

    @staticmethod
    def dx(z):
        # Assume that z has shape k x 1 where k is the number of classes
        """
        gradients = np.zeros(z.shape)
        num_classes = z.shape[0]
        for i in range(num_classes):
            gradient = 0.0
            for j in range(num_classes):
                if i == j:
                    gradient += float(z[j][0] * (1 - z[j][0]))
                else:
                    gradient += float(-z[j][0] * z[i][0])
            gradients[i][0] = gradient
        return gradients
        """
        # This is not the correct gradient of the Softmax. The gradient has been taken in the 
        # cross entropy loss function. Therefore, we don't handle any derivatives at this step.
        return np.ones(z.shape)

# This class represents a single hidden or output layer in the neural network
class DenseLayer(object):

    # numNodes: number of hidden units in the layer
    # activation: the activation function to use in this layer
    def __init__(self, numNodes, activation):
        self.numNodes = numNodes
        self.activation = activation
        self.type = Layer.DENSE

    def getType(self):
        return self.type

    def getNumNodes(self):
        return self.numNodes

    # Initialize the weight matrix of this layer based on the size of the matrix W
    def initialize(self, fanIn, scale=1.0):
        s = scale * np.sqrt(6.0 / (self.numNodes + fanIn))
        self.W = np.random.normal(0, s,
                                   (self.numNodes,fanIn))
        self.b = np.random.uniform(-1,1,(self.numNodes,1))

    # Apply the activation function of the layer on the input z
    def a(self, z):
        return self.activation.fx(z)

    # Compute the linear part of the layer
    # The input a is an n x k matrix where n is the number of samples
    # and k is the dimension of the previous layer (or the input to the network)
    def z(self, a):
        return self.W.dot(a) + self.b # Note, this is implemented where we assume a is k x n

    # Compute the derivative of the layer's activation function with respect to z
    # where z is the output of the above function.
    # This derivative does not contain the derivative of the matrix multiplication
    # in the layer.  That part is computed below in the model class.
    def dx(self, z):
        return self.activation.dx(z)

    # Update the weights of the layer by adding dW to the weights
    def updateWeights(self, dW):
        self.W = self.W + dW

    # Update the bias of the layer by adding db to the bias
    def updateBias(self, db):
        self.b = self.b + db


class ConvolutionLayer(object):
    # window: the size of each window
    # numNodesPrevW: number of nodes in the width side of the previous layer
    # numNodesPrevL: number of nodes in the length side of the previous layer
    def __init__(self, window, numNodesPrevW, numNodesPrevL, activation):
        self.window = window
        self.numNodesPrevW = numNodesPrevW
        self.numNodesPrevL = numNodesPrevL
        self.numNodesW = self.numNodesPrevW - self.window + 1
        self.numNodesL = self.numNodesPrevL - self.window + 1
        self.numNodes = self.numNodesW * self.numNodesL
        self.activation = activation
        self.type = Layer.CONVOLUTION

    def getType(self):
        return self.type

    def getNumNodes(self):
        return self.numNodes

    # Initialize the weight matrix of this layer based on the size of the matrix W
    def initialize(self, fanIn, scale=1.0):
        s = scale * np.sqrt(6.0 / (self.numNodes + fanIn))
        self.W = np.random.normal(0, s,
                                   (self.window,self.window))
        self.b = np.random.uniform(-1,1,(self.numNodes,1))

    # Apply the activation function of the layer on the input z
    # ConvolutionLayer layer doesn't have any activitation function
    def a(self, z):
        return self.activation.fx(z)

    # Compute the linear part of the layer
    # The input a is an n x k matrix where n is the number of samples
    # and k is the dimension of the previous layer (or the input to the network)
    def z(self, a):
        n = a.shape[0]
        self.W_Prev = np.zeros(a.shape)
        step = self.window
        output = np.zeros((self.numNodes, a.shape[1]))
        # Note, this is implemented where we assume a is k x n
        # Split the data in chunks based on the size of the max pooling layer
        a_copy = a.copy()
        # reshape a into a square image representation
        a_copy = a_copy.reshape((self.numNodesPrevW, self.numNodesPrevL, a_copy.shape[1]))
        l = 0
        for k in range(a.shape[1]):
            for i in range(self.numNodesW):
                for j in range(self.numNodesL):
                    batch = a_copy[i:i+step,j:j+step,k]
                    convolution = np.sum(np.multiply(self.W, batch))
                    # Bookkeeping the result
                    output[l][k] = convolution
                    l += 1
            l = 0
        """
        print "W"
        print self.W.reshape(self.window,self.window)
        print "output"
        print output.reshape(self.numNodesW, self.numNodesL)
        print "a"
        print a.reshape(self.numNodesPrevW, self.numNodesPrevL)
        """
        return output + self.b

    # Compute the derivative of the layer's activation function with respect to z
    # where z is the output of the above function.
    # This derivative does not contain the derivative of the matrix multiplication
    # in the layer.  That part is computed below in the model class.
    def dx(self, z):
        return self.activation.dx(z)

    # Update the weights of the layer by adding dW to the weights
    def updateWeights(self, dW):
        self.W = self.W + dW

    # Update the bias of the layer by adding db to the bias
    def updateBias(self, db):
        self.b = self.b + db



class MaxPoolingLayer(object):
    # window: the size of each window
    # numNodesPrev: number of nodes of the previous layer
    def __init__(self, window, numNodesPrevW, numNodesPrevL):
        self.window = window
        self.numNodesPrevW = numNodesPrevW
        self.numNodesPrevL = numNodesPrevL
        self.numNodesW = self.numNodesPrevW - self.window + 1
        self.numNodesL = self.numNodesPrevL - self.window + 1
        self.numNodes = self.numNodesW * self.numNodesL
        self.type = Layer.MAXPOOLING

    def getType(self):
        return self.type

    def getNumNodes(self):
        return self.numNodes

    # Initialize the weight matrix of this layer based on the size of the matrix W
    def initialize(self, fanIn, scale=1.0):
        # We don't initialize the weight yet
        # b is useless, but for conforming to the API we still keep it
        self.b = np.random.uniform(-1,1,(self.numNodes,1))

    # Apply the activation function of the layer on the input z
    # MaxPooling layer doesn't have any activitation function
    def a(self, z):
        return z

    # Compute the linear part of the layer
    # The input a is an n x k matrix where n is the number of samples
    # and k is the dimension of the previous layer (or the input to the network)
    def z(self, a):
        n = a.shape[0]
        step = self.window
        output = np.zeros((self.numNodes, a.shape[1]))
        # initialize the weight
        self.W = np.zeros((self.numNodes, a.shape[1]))
        self.W_Prev = np.zeros(a.shape)
        # Note, this is implemented where we assume a is k x n
        # Split the data in chunks based on the size of the max pooling layer
        a_copy = a.copy()
        # reshape a into a square image representation
        a_copy = a_copy.reshape((self.numNodesPrevW, self.numNodesPrevL, a_copy.shape[1]))
        l = 0
        for k in range(a.shape[1]):
            for i in range(self.numNodesW):
                for j in range(self.numNodesL):
                    maximum_ii, maximum_jj = 0, 0 
                    maximum = float("-inf")
                    # find the maximum index and value in this window
                    for ii in range(i,i+step):
                        for jj in range(j,j+step):
                            if a_copy[ii][jj][k] > maximum:
                                maximum = a_copy[ii][jj][k]
                                maximum_ii = ii
                                maximum_jj = jj
                    # We only sample the maximum. Propagate the output matrix with max pooling values
                    output[l][k] = maximum
                    # Bookkeeping in the weight matrix, remember which window was used to determine the winner
                    self.W[l][k] = maximum_ii * self.numNodesPrevW + maximum_jj
                    l += 1
            l = 0
        """
        print "W"
        print self.W.reshape(self.numNodesW,self.numNodesL)
        print "output"
        print output.reshape(self.numNodesW, self.numNodesL)
        print "a"
        print a.reshape(self.numNodesPrevW, self.numNodesPrevL)
        """

        return output

    # Compute the derivative of the layer's activation function with respect to z
    # where z is the output of the above function.
    # This derivative does not contain the derivative of the matrix multiplication
    # in the layer.  That part is computed below in the model class.
    def dx(self, z):
        # max pooling layer doesn't have any activitation functions
        return np.ones(z.shape)

    # Max pooling layer doesn't need to update the weights
    # Weight is 1 for the maximum node and 0 for everything else
    def updateWeights(self, dW):
        pass

    # Max pooling layer doesn't have bias
    def updateBias(self, db):
        pass


# This class handles stacking layers together to form the completed neural network
class Model(object):

    # inputSize: the dimension of the inputs that go into the network
    def __init__(self, inputSize):
        self.layers = []
        self.inputSize = inputSize

    # Add a layer to the end of the network
    def addLayer(self, layer):
        self.layers.append(layer)

    # Get the output size of the layer at the given index
    def getLayerSize(self, index):
        if index >= len(self.layers):
            return self.layers[-1].getNumNodes()
        elif index < 0:
            return self.inputSize
        else:
            return self.layers[index].getNumNodes()

    # Initialize the weights of all of the layers in the network and set the cost
    # function to use for optimization
    def initialize(self, cost, initializeLayers=True):
        self.cost = cost
        if initializeLayers:
            for i in range(0,len(self.layers)):
                if i == len(self.layers) - 1:
                    self.layers[i].initialize(self.getLayerSize(i-1))
                else:
                    self.layers[i].initialize(self.getLayerSize(i-1))

    # Compute the output of the network given some input a
    # The matrix a has shape n x k where n is the number of samples and
    # k is the dimension
    # This function returns
    # yp - the output of the network
    # a - a list of inputs for each layer of the newtork where
    #     a[i] is the input to layer i
    # z - a list of values for each layer after evaluating layer.z(a) but
    #     before evaluating the nonlinear function for the layer
    def evaluate(self, x):
        curA = x.T # This transfers the input design matrix a into shape k x n instead to conform to the nerual network
        a = [curA]
        z = []
        for layer in self.layers:
            # TODO #################################################################################
            # Store the input to each layer in the list a
            # Store the result of each layer before applying the nonlinear function in z
            # Set yp equal to the output of the network
            # ######################################################################################
            # Compute the linear part of the layer
            layer_z = layer.z(a[-1])
            # Store the result of each layer before applying the nonlinear function in z
            z.append(layer_z)
            # Apply nonlinear function and then store a as input to the next layer
            layer_a = layer.a(layer_z)
            a.append(layer_a)
        # Set yp equal to the output of the network
        yp = a.pop()
        return yp, a, z

    # Compute the output of the network given some input a
    # The matrix a has shape n x k where n is the number of samples and
    # k is the dimension
    def predict(self, a):
        a,_,_ = self.evaluate(a)
        return a.T

    # Train the network given the inputs x and the corresponding observations y
    # The network should be trained for numEpochs iterations using the supplied
    # optimizer
    def train(self, x, y, numEpochs, optimizer):

        # Initialize some stuff
        n = x.shape[0]
        x = x.copy()
        y = y.copy()
        hist = []
        optimizer.initialize(self.layers)
        
        # Run for the specified number of epochs
        for epoch in range(0,numEpochs):

            # Feed forward
            # Save the output of each layer in the list a
            # After the network has been evaluated, a should contain the
            # input x and the output of each layer except for the last layer
            yp, a, z = self.evaluate(x)

            # Compute the error
            C = self.cost.fx(yp,y.T)
            d = self.cost.dx(yp,y.T)
            grad = []

            # Backpropogate the error
            idx = len(self.layers)
            for layer, curZ in zip(reversed(self.layers),reversed(z)):
                # TODO #################################################################################
                # Compute the gradient of the output of each layer with respect to the error
                # grad[i] should correspond with the gradient of the output of layer i before
                # before the activation is applied (dMSE/dz_i)
                # ######################################################################################
                idx = idx - 1
                # Here, we compute dMSE/dz_i because in the update
                # function for the optimizer, we do not give it
                # the z values we compute from evaluating the network
                if layer.getType() == Layer.DENSE:
                    grad.insert(0,np.multiply(d,layer.dx(curZ)))
                    d = np.dot(layer.W.T,grad[0])
                if layer.getType() == Layer.MAXPOOLING:
                    grad.insert(0,np.multiply(d,layer.dx(curZ)))
                    _W = layer.W.copy()
                    d = layer.W_Prev.copy()
                    for k in range(_W.shape[1]):
                        for i in range(_W.shape[0]):
                            val = int(_W[i][k])
                            d[val][k] += grad[0][i][k] # We route the grdient to this weight
                    """
                    print "########GRAD#########"
                    print grad[0].shape
                    print grad[0]
                    print "########_W###########"
                    print _W.shape
                    print _W
                    print "########d###########"
                    print d.shape
                    print d
                    """
                if layer.getType() == Layer.CONVOLUTION:
                    grad.insert(0,np.multiply(d,layer.dx(curZ)))
                    # rotate the weight matrix by 180 degrees
                    _W = np.rot90(layer.W.copy(),2)
                    d = layer.W_Prev.copy()
                    # get the previous gradient
                    d_prev = grad[0].copy()
                    # reshape the previous gradient into a rectangle 
                    d_prev = d_prev.reshape(layer.numNodesW, layer.numNodesL, d_prev.shape[1])
                    margin = _W.shape[0] - 1
                    # To take the gradient, we need to pad the margin of previous gradient with zeros
                    d_prev_padding = np.zeros((d_prev.shape[0] + 2 * margin, d_prev.shape[1] + 2 * margin, d_prev.shape[1]))
                    d_prev_padding[margin:margin + layer.numNodesW, margin:margin + layer.numNodesL, :] = d_prev

                    # Then, the gradient is computed as taking convolution of the previous gradient with padding
                    step = layer.window
                    l = 0
                    # Iterate through each data point
                    for k in range(grad[0].shape[1]):
                        for i in range(layer.numNodesPrevW):
                            for j in range(layer.numNodesPrevL):
                                batch = d_prev_padding[i:i+step,j:j+step,k]
                                convolution = np.sum(np.multiply(_W, batch))
                                # Bookkeeping the result
                                d[l][k] = convolution
                                l += 1
                        l = 0

            # Update the errors
            optimizer.update(self.layers, grad, a)

            # Compute the error at the end of the epoch
            yh = self.predict(x)
            C = self.cost.fx(yh,y) # Just to conform to the convention 
            C = np.mean(C)
            hist.append(C)
        return hist

    def trainBatch(self, x, y, batchSize, numEpochs, optimizer):
        # Copy the data so that we don't affect the original one when shuffling
        x = x.copy()
        y = y.copy()
        hist = []
        n = x.shape[0]
        
        for epoch in np.arange(0,numEpochs):
            
            # Shuffle the data
            r = np.arange(0,x.shape[0])
            x = x[r,:]
            y = y[r,:]
            e = []

            # Split the data in chunks and run SGD
            for i in range(0,n,batchSize):
                end = min(i+batchSize,n)
                batchX = x[i:end,:]
                batchY = y[i:end,:]
                e += self.train(batchX, batchY, 1, optimizer)
            #hist.append(np.mean(e))
            hist += e

        return hist

"""
if __name__ == '__main__':

    # Generate the training set
    np.random.seed(9001)
    x=np.random.uniform(-np.pi,np.pi,(1000,1))
    y=np.sin(x)

    activations = dict(ReLU=ReLUActivation,
                       tanh=TanhActivation,
                       linear=LinearActivation)
    lr = dict(ReLU=0.02,tanh=0.02,linear=0.005)

    for key in activations:

        # Build the model
        activation = activations[key]
        model = Model(x.shape[1])
        model.addLayer(DenseLayer(100,activation()))
        model.addLayer(DenseLayer(100,activation()))
        model.addLayer(DenseLayer(1,LinearActivation()))
        model.initialize(QuadraticCost())

        # Train the model and display the results
        hist = model.train(x,y,500,GDOptimizer(eta=lr[key],gamma=1, epochs=10000))
        yHat = model.predict(x)
        error = np.mean(np.square(yHat - y))/2
        print(key+' MSE: '+str(error))
        plt.plot(hist)
        plt.title(key+' Learning curve')
        plt.show()

        # TODO #################################################################################
        # Plot the approximation of the sin function from all of the models
        # ######################################################################################

"""
if __name__ == '__main__':
    training, training_labels, validation, validation_labels, X_test = MNISTPreprocessor('./data/').load_data()

    x,y = training, MNISTPreprocessor.one_hot(training_labels)

    activations = dict(ReLU=ReLUActivation,
                       tanh=TanhActivation,
                       linear=LinearActivation,
                       softmax=SoftmaxActivation)
    lr = dict(ReLU=0.02,tanh=0.02,linear=0.005,softmax=0.001)

    model = Model(x.shape[1])
    model.addLayer(DenseLayer(784, ReLUActivation()))
    model.addLayer(ConvolutionLayer(window=3,numNodesPrevW=28,numNodesPrevL=28,activation=ReLUActivation()))
    #model.addLayer(DenseLayer(400, ReLUActivation()))
    model.addLayer(MaxPoolingLayer(window=3,numNodesPrevW=26,numNodesPrevL=26))
    model.addLayer(DenseLayer(400, ReLUActivation()))
    model.addLayer(DenseLayer(10, SoftmaxActivation()))
    model.initialize(CrossEntropyCost())

    # Train the model and display the results
    # hist = model.train(x,y,100, GDOptimizer(eta=0.001, gamma=0.9))
    hist = model.trainBatch(x,y,1, 1, GDOptimizer(eta=0.001, gamma=1, epochs=10000))
    yHat = model.predict(x)

    pred_labels_train = []
    for i in range(yHat.shape[0]):
        number = np.argmax(yHat[i])
        pred_labels_train.append(number)

    yHat_validation = model.predict(validation)
    pred_labels_validation = []
    for i in range(yHat_validation.shape[0]):
        number = np.argmax(yHat_validation[i])
        pred_labels_validation.append(number)

    print("Stochastic gradient descent")
    print("Train accuracy: {0}".format(metrics.accuracy_score(training_labels, pred_labels_train)))
    print("Validation accuracy: {0}".format(metrics.accuracy_score(validation_labels, pred_labels_validation)))

    plt.plot(hist)
    plt.title('Learning curve')
    plt.show()




