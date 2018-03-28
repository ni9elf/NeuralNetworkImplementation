import numpy as np
import random
import time
import data_loader
import sys

class Network:
    '''
    Class definition for a feedforward neural network or MLP.
    '''
    def __init__(self, neuron_list, no_inputs):
        #neuron list contains the number of neuron for each layer (input, multiple hidden and output)
        self.neuron_list = neuron_list
        self.no_of_layers = len(neuron_list)
        #weight[i] stores the weights between layer i and layer i+1
        #weights are initialized from a normal distribution with mean 0 and variance 2/no_inputs
        self.weights = [np.random.normal(0, np.sqrt(2./no_inputs), (n0*n1)).reshape(n1, n0) for n0, n1 in zip(neuron_list[:-1], neuron_list[1:])]
        #biases are stored from first hidden layer onwards
        self.biases = [np.random.normal(0, np.sqrt(2./no_inputs), neurons).reshape(neurons, 1) for neurons in neuron_list[1:]]


    def feedforward(self, z):    
        '''
        Computes the output of the MLP given an input vector z. Returns z = weights.activations + biases at each layer.
        '''  
        z_list = []
        z_list.append(z)
        for i in xrange(self.no_of_layers - 1):
            #dot function perform matrix multiplication in numpy
            #z is the input to layer i, biases[i] is the bias vector for the i+1 layer            
            z = self.weights[i].dot(z) + self.biases[i]
            z_list.append(z)
            a = sigmoid(z)
            z = a
        return z_list

    
    def stochastic_gradient_descent(self, tr_data, no_epochs, mini_batch_size, eta, test_data = None):
        '''
        Performs stochastic gradient descent and indirectly calls backpropogation to train the neural network.
        '''
        for i in xrange(no_epochs):
            start = time.time()
            #shuffle data for stochastic sampling
            random.shuffle(tr_data)
            #index mini batches
            mini_batches = [tr_data[j : j+mini_batch_size] for j in xrange(0, len(tr_data), mini_batch_size)]
            for mini_batch in mini_batches:
                #train network for each mini batch with learning rate eta
                self.update_network(mini_batch, eta, mini_batch_size)            
            print 'Epoch {} complete'.format(i)
            if(test_data is not None):
                print 'Evaluation results: {}/{}'.format(self.evaluate(test_data), len(test_data))
            end = time.time()
            print 'Time taken: {}s'.format(end - start)


    def update_network(self, mini_batch, eta, mini_batch_size):
        #stores running sum of nabla_w and nabla_b obtained from backpropogation for all data points in mini batch
        nabla_w_sum = []
        nabla_b_sum = []
        for x, y in mini_batch:
            #the learning step
            nabla_w_x, nabla_b_x = self.back_propogation(x, y)
            #store running sum
            if(len(nabla_w_sum) == 0 and len(nabla_b_sum) == 0):
                nabla_w_sum = nabla_w_x
                nabla_b_sum = nabla_b_x
            else:
                nabla_w_sum = [x+y for x, y in zip(nabla_w_sum, nabla_w_x)]
                nabla_b_sum = [x+y for x, y in zip(nabla_b_sum, nabla_b_x)]                                       
        #update weights and biases of neural network with learning rate and mean of nabla_w_sum and nabla_b_sum
        self.weights = [weight - ((eta/mini_batch_size) * nabla_w_i) for weight, nabla_w_i in zip(self.weights, nabla_w_sum)]
        self.biases = [bias - ((eta/mini_batch_size) * nabla_b_i) for bias, nabla_b_i in zip(self.biases, nabla_b_sum)]    


    def back_propogation(self, x, y):
        z_list = self.feedforward(x)   
        #construct activation outputs excluding last layer (since not needed for nabla_w calculation)
        #don't apply activation function to input neurons in first layer
        a_list = [z_list[0]] + [sigmoid(z) for z in z_list[1:-1]] 
        #calculate error at output layer
        output_error = (sigmoid(z_list[-1]) - y) * sigmoid_prime(z_list[-1])               
        error = [output_error]
        #back propogate error to previous layers
        for i in xrange(self.no_of_layers - 2, 0, -1):            
            err =  self.weights[i].transpose().dot(error[0]) * sigmoid_prime(z_list[i])            
            error = [err] + error
        #calculate change in weights from error
        nabla_w_x = [err.dot(a.transpose()) for err, a in zip(error, a_list)]
        #change in biases (nabla_b_x) is equal to the error
        #this will return nabla_w_x, nabla_b_x
        return nabla_w_x, error
        
    
    def evaluate(self, test_data):       
        #count no of correct predictions
        #argmax used since the labels are in the form of one hot vectors        
        return sum([(np.argmax(sigmoid(self.feedforward(x)[-1])) == np.argmax(y)) for (x, y) in test_data])
    
    
def sigmoid(z):
    #activation function
    return 1.0 / (1 + np.exp(-z))


def sigmoid_prime(z):
    #derivative of activation function
    return sigmoid(z) * (1.0 - sigmoid(z))        


if __name__ == '__main__':
    #this prints the entire numpy array, useful for debugging
    np.set_printoptions(threshold=np.inf)  
    tr_data, val_data, test_data = data_loader.load_data()       
    neuron_list = sys.argv[1][1:-1].split(',')
    neuron_list = [int(neurons) for neurons in neuron_list]
    n = Network(neuron_list, len(tr_data))
    n.stochastic_gradient_descent(tr_data, int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), test_data)
