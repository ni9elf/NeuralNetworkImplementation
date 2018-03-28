import cPickle
import gzip
import numpy as np

def load_data():
    '''
    Returns a tuple of three lists for training, validation and test data repectively. Each list contains tuples of two elements where the first element is a numpy ndarray of 784 (28X28 pixels) rows and 1 column (a column vector) and the second element is a one hot vector representing the label (0-9) (1 in the index corresponding to the label and 0 elsewhere). The second element has dimensions of 10 rows and 1 column and is also a column vector.
    
    Note: The input file contains a tuple of three lists for training, validation and test data repectively. Each list contains two nested lists with the first one corresponding to the input (numpy ndarray of 784 (28X28 pixels) rows and 1 column) and the second one to the output (label output values ranging from 0 to 9). The length of the lists are 50000, 10000 and 10000 corresponding to the number of data points.
    '''    
    #this prints the entire numpy array, useful for debugging
    np.set_printoptions(threshold=np.inf)
    #MNIST data set is contained in mnist.pkl.gz
    with gzip.open('./mnist.pkl.gz', 'rb') as f:
        tr_data, va_data, te_data = cPickle.load(f)
    #store final data in this list to return
    #this list helps create a for loop avoiding running the same code 3 time for training, validation and test data
    final_data = []
    #modifying data formats to be more suitable to the input format of our MLP
    for data in [tr_data, va_data, te_data]:
        #784 X 1 column vector as input
        data_x = [x.reshape((784, 1)) for x in data[0]]
        data_y = np.array([x for x in data[1]])                    
        rows = data_y.shape[0]
        temp = np.zeros((rows, 10))
        #using fancy indexing of numpy to create one hot vectors of the labels
        temp[np.arange(rows), data_y] = 1
        data_y = temp.tolist()
        #10 X 1 column vector as output
        data_y = [np.array(y).reshape((10, 1)) for y in data_y]   
        #tuples of (x, y)               
        final_data.append(zip(data_x, data_y))        
    #this will return training_data, validation_data, test_data 
    return (final_data[0], final_data[1], final_data[2])        
