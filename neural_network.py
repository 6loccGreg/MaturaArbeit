import numpy as np
import os
import cv2
from cv2 import imwrite
import pickle
import copy  
import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import spiral_data, sine_data

nnfs.init()

#n_inputs is same as input dimension and n_neurons is same as output dimensions

#Dense Layer 
class Layer_Dense:
    
    #Layer initialization
    def __init__(self, n_inputs, n_neurons, weight_regularizer_l1=0, weight_regularizer_l2=0, bias_regularizer_l1=0, bias_regularizer_l2=0):
        #intialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        
        #Set regularization strength
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2
        
    
    #Forward pass
    def forward(self, inputs, training):
        self.output = np.dot(inputs, self.weights) + self.biases
        self.inputs = inputs
    #Backward pass 
    def backward(self, dvalues):
        
        #Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis =0, keepdims=True)
        
        #Gradients on regularization
        #L1 on weights
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights <0] = -1
            self.dweights += self.weight_regularizer_l1 *dL1
        
        #L2 on weights
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights
        
        #L1 on biases
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1
        
        #L2 on biases
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2* self.bias_regularizer_l2 * self.biases
        
        #Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)

    #Retrieve layer parameters
    def get_parameters(self):
        return self.weights, self.biases
    
    #Set weights and biases in a layer instance
    def set_parameters(self, weights, biases):
        self.weights = weights
        self.biases = biases
        
#Dropout 
class Layer_Dropout:
    
    #Init
    def __init__(self,rate):
        #Store rate, we invert it as for example for dropout
        #of 0.1 we need success rate of 0.9
        self.rate = 1-rate
    
    #Forward pass
    def forward(self, inputs, training):
        #Save input valuesss
        self.inputs = inputs
        
        #If not in the training mode - return values
        if not training:
            self.output = inputs.copy()
            return
            
        #Generate and save scaled mask
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        #Apply mask to output values and
        self.output = inputs * self.binary_mask
    
    #Backward pass
    def backward(self, dvalues):
        #Gradient on values
        self.dinputs = dvalues * self.binary_mask

#Input "layer"
class Layer_Input:
    
    #Forward pass
    def forward(self, inputs, training):
        self.output = inputs

class Layer_Max_Pooling:

    def __init__(self, step_size=2, kernel_size=2, mode="max"):
        self.step_size = step_size
        self.kernel_size = kernel_size
        self.last_max_row = []
        self.last_max_col = []
        self.mode = mode
                

    def forward(self, input):
        self.input = input

        self.output = []
        
        for img in range(len(input)):
            self.output.append(self.pool(input[img]))
        
        return np.squeeze(np.array(self.output))

    def pool(self, input):
                
        self.input_rows = len(input)
        self.input_cols = len(input[0])
                
        self.output_rows = int((self.input_rows - self.kernel_size)/ self.step_size + 1)
        self.output_cols = int((self.input_cols - self.kernel_size)/ self.step_size + 1)
        
        output = np.ndarray(shape=(self.output_rows, self.output_cols))
        
        max_rows = np.ndarray(shape=(self.output_rows, self.output_cols))
        max_cols = np.ndarray(shape=(self.output_rows, self.output_cols))
                        
        for row in range(0, self.output_rows, self.step_size):
            for col in range(0, self.output_cols, self.step_size):
                
                
                max = 0.0
                max_rows[row][col] = -1
                max_cols[row][col] = -1
                # here if max or avg
                avg = 0.0
                for x in range(self.kernel_size):
                    for y in range(self.kernel_size):
                        if max < input[row + x][col + y]:
                            max = input[row + x][col + y]
                            max_rows[row][col] = row + x
                            max_cols[row][col] = col + y
                
                output[row][col] = max

        self.last_max_row.append(max_rows)
        self.last_max_col.append(max_cols)
        
        return output

    def backward(self, dvalues):

        self.dinputs = []

        l = 0
        
        for array in dvalues:
            
            error = np.ndarray(shape=(self.input_rows, self.input_cols))
            
            for row in range(self.output_rows):
                for col in range(self.output_cols):
                    max_x = self.last_max_row[l][row][col]
                    max_y = self.last_max_col[l][row][col]
                    
                    if max_x != -1:
                        error[max_x][max_y] += array[row][col]
                        

            self.dinputs.append(error)
            
            l+=1 
        
        self.dinputs = np.squeeze(np.array(self.dinputs))
    
        

class Layer_Convolution:
    
    def __init__(self, num_filters, kernel_size=3, step_size=1, padding=0, seed=0.001):
        
        self.filters = [] 
        self.kernel_size = kernel_size
        self.step_size = step_size
        self.seed = seed
        self.num_filters = num_filters
        self.padding = padding
        self.generate_random_filters(num_filters)
            
    def forward(self, input):

        self.last_input = input

        self.output = []    
        if len(input.shape) == 3:
            for matrix in range(len(input)):
                for filter in self.filters:
                    self.output.append(self.convolve(input[matrix], filter, self.step_size))
        
        if len(input.shape) == 2:
            for filter in self.filters:
                self.output.append(self.convolve(input, filter, self.step_size))
            
        self.input = input        

        return np.squeeze(np.array(self.output))
            
    def convolve(self, input, filter, step_size):
        if self.padding != 0:
            input_padded = np.zeros((len(input) + self.padding *2, len(input[0]) + self.padding*2))
            input_padded[int(self.padding):int(-1 * self.padding), int(self.padding):int(-1 * self.padding)] = input
        
        else: 
            input_padded = input
        
        output_rows = int((len(input_padded) - len(filter))/step_size + 1)  
        output_cols = int((len(input_padded[0]) - len(filter[0]))/ step_size + 1)
            
        self.input_rows = len(input_padded)
        self.input_cols = len(input_padded[0])
        
        filter_rows = len(filter)
        filter_cols = len(filter[0])
        
        output = np.ndarray(shape=(output_rows, output_cols))
        
        output_row = 0
        
        for row_index in range(0, self.input_rows-filter_rows+ 1, step_size):
            
            output_col = 0
            
            for col_index in range(0, self.input_cols-filter_cols + 1, step_size):
                
                sum = 0.0
                
                #Apply the filter
                for row_x  in range(filter_rows):
                    for col_y in range(filter_cols):
                        input_row_index = row_index + row_x
                        input_col_index = col_index + col_y
                        
                        value = filter[row_x][col_y] * input_padded[input_row_index][input_col_index]
                        sum+= value
                        
                
                output[output_row][output_col] = sum
                output_col +=1
            
            output_row +=1
                    
        return output

    
    
    
    def generate_random_filters(self, num_filters):
        output = []
        
        random = randint(1,100) * self.seed
                    
        for filter in range(num_filters):
            new_filter = np.ndarray(shape=(self.kernel_size, self.kernel_size))
            
            for i in range(self.kernel_size):
                for j in range(self.kernel_size):
                    
                    value = gauss(-1., 1.)
                    
                    new_filter[i][j] = value
                    
            self.filters.append(new_filter)
        
        
    def space_array(self, input):
        
        if self.step_size == 1:
            return input
        
        
        output_rows = (len(input) -1)*self.step_size +1
        output_cols = (len(input[0]) -1)*self.step_size +1
        
        output = np.ndarray(shape=(output_rows, output_cols))
        
        for row in range(len(input)):
            for col in range(len(input[0])):
                output[row*self.step_size][col*self.step_size] = input[row][col]
        
        return output

    def backward(self,dvalues):
        
        filters_delta = []
        
        self.dinputs = []
        
        for filter_index in range(self.kernel_size): 
            filters_delta.append(np.ndarray(shape=(self.kernel_size, self.kernel_size)))
        
        for input in range(self.last_input):
            
            error_for_input = np.ndarray(shape=(self.input_rows, self.input_cols))
            for filter in range(len(self.filters)):
                current_filter = self.filters[filter]
                error = dvalues[input*self.kernel_size + filter]
                
                spaced_error = self.space_array(error)
                self.dweights = np.array(self.convolve(self.last_input[input], spaced_error, step_size=1))
                
                flipped_error = self.flip_array_horizontal(self.flip_array_vertical(spaced_error))
                error_for_input = np.append(error_for_input, self.full_convolve(current_filter,flipped_error))
                
            self.dinputs.append(error_for_input)
        self.dbiases = dvalues
        
    def flip_array_horizontal(self, array):
        rows = len(array)
        cols = len(array[0])

        output = np.ndarray(shape=(rows, cols))

        for row in range(rows):
            for col in range(cols):
                output[rows-row-1][col] = array[row][col]
        
        return output
    
    def flip_array_vertical(self, array):
        rows = len(array)
        cols = len(array[0])
        
        output = np.ndarray(shape=(rows, cols))
        
        for row in range(rows):
            for col in range(cols):
                output[row][cols-col-1] = array[row][col]
        
        return output
    
    def full_convolve(self, input, filter):
            
        if self.padding != 0:
            input_padded = np.zeros((len(input) + self.padding *2, len(input[0]) + self.padding*2))
            input_padded[int(self.padding):int(-1 * self.padding), int(self.padding):int(-1 * self.padding)] = input
        
        else: 
            input_padded = input
        
        output_rows = int((len(input_padded) + len(filter)) + 1)  
        output_cols = int((len(input_padded[0]) + len(filter[0])) + 1)
            
        input_rows = len(input_padded)
        input_cols = len(input_padded[0])
        
        filter_rows = len(filter)
        filter_cols = len(filter[0])
        
        output = np.ndarray(shape=(output_rows, output_cols))
        
        output_row = 0
        
        for row_index in range(-filter_rows + 1, input_rows):
            
            output_col = 0
            
            for col_index in range(-filter_cols + 1, input_cols):
                
                sum = 0.0
                
                #Apply the filter
                for row_x  in range(filter_rows):
                    for col_y in range(filter_cols):
                        input_row_index = row_index + row_x
                        input_col_index = col_index + col_y
                        
                        if input_row_index >= 0 and input_col_index >= 0 and input_row_index < input_rows and input_col_index < input_cols:
                            value = filter[row_x][col_y] * input_padded[input_row_index][input_col_index]
                            sum+= value
                            
                
                output[output_row][output_col] = sum
                output_col +=1
            
            output_row +=1
                    
        return output
    
    def matrix_to_vector(self, input):

        length = len(input)
        rows = len(input[0])
        cols = len(input[0][0])
        
        vector = np.ndarray(shape=(length*rows*cols, 1))
        
        i = 0
        for l in range(length):
            for row in range(rows):
                for col in range(cols):
                    vector[i] = input[l][row][col]
                    i+=1
        
        return vector
    
    def vector_to_matrix(self, input: list, length: int, rows:int, cols:int):

        output = []
        
        i=0
        for l in range(length):
            matrix = np.ndarray(shape=(rows, cols))
            for row in range(rows):
                for col in range(cols):
                    matrix[row][col] = input[i]
                    i+=1
            
            output.append(matrix)


#ReLu Activation 
class Activation_ReLU:
    
    #Forward pass 
    def forward(self, inputs, training): 
        
        #calculate output values from inputs with the formula
        self.output = np.maximum(0, inputs)
        self.inputs = inputs
    
    #Backward pass 
    def backward(self, dvalues):
        #Since weneed to modify the original variable,
        # let's make a copy of the values first
        self.dinputs = dvalues.copy()
        
        #Zero gradient where inputs values were negative
        self.dinputs[self.inputs <= 0] = 0
    
    #Calculate predictions for outputs
    def predictions(self, outputs):
        return outputs

#Softmax Activation
class Activation_Softmax:
    
    #Forward pass 
    def forward(self, inputs, training):
        #Remember input values
        self.inputs = inputs
        #Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        
        #Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        
        
        self.output = probabilities
    
    #Backward pass
    def backward(self, dvalues):
        
        #Create uninitialized array
        self.dinputs = np.empty_like(dvalues)
        
        #Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            #Flatten output array
            single_output = single_output.reshape(-1, 1)
            #Calculate Jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            #Calculate sample-wise gradient
            #and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

    #Calculate predictions for outputs
    def predictions(self, outputs):
        return np.argmax(outputs, axis=1)
#Sigmoid activation
class Activation_Sigmoid:
    
    #Forward pass
    def forward(self, inputs, training):
        #Save input and calculate/ save output
        #of the sigmoid function
        self.inputs = inputs
        self.output = 1/ (1+ np.exp(-inputs))
    
    #Backward pass
    def backward(self, dvalues):
        #Derivative - calculates from the output of the sigmoid function
        self.dinputs = dvalues * (1 - self.output) * self.output
    
    #Calculate predictions for outputs
    def predictions(self, outputs):
        return (outputs > 0.5) * 1

#Linear activation
class Activation_Linear:
    
    #Forward pass
    def forward(self, inputs, training):
        
        #Just remember values
        self.inputs = inputs
        self.output = inputs
    
    #Backward pass
    def backward(self, dvalues):
        #Derivative is 1, 1* dvalues = dvalues - the chain rule
        self.dinputs = dvalues.copy()
    
    #Calculate predictions for outputs
    def predictions(self, outputs):
        return outputs
    
#Common loss class
class Loss:
    
    #Regularization loss calculation
    def regularization_loss(self):
        
        #0 by default
        regularization_loss = 0
        
        #Calculate regularization loss
        #iterate all trainable layers
        for layer in self.trainable_layers:
            
            #L1 regularization - weights
            #calculate only when factor is greater than 0
            if layer.weight_regularizer_l1 > 0:
                regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))
            
            #L2 regularization - weights
            if layer.weight_regularizer_l2 > 0:
                regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights * layer.weights)
            
            #L1 regularization - biases
            #calculate only when factor greater than 0
            if layer.bias_regularizer_l1 > 0:
                regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))
            
            #L2 regularization - biases
            if layer.bias_regularizer_l2 > 0:
                regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases * layer.biases)
            
        return regularization_loss

    #Set/ remember trainable layers
    def remember_trainable_layers(self, trainable_layers): 
        self.trainable_layers = trainable_layers
        
    #Calculates the data and regularization losses
    #given model output and ground truth
    def calculate(self, output, y, *, include_regularization=False):
        
        #Calculate sample loss
        sample_losses = self.forward(output, y)
        
        #Calculate mean loss
        data_loss = np.mean(sample_losses)
        
        #Add accumulated sum of losses and sample count
        self.accumulated_sum += np.sum(sample_losses)
        self.accumulated_count += len(sample_losses)
        
        #If just data loss - return it
        if not include_regularization:
            return data_loss
        
        #Return loss and regularization_loss
        return data_loss, self.regularization_loss()
    
    #Calculates accumulated loss
    def calculate_accumulated(self, *, include_regularization=False): 
        
        #Calculate mean loss
        data_loss = self.accumulated_sum / self.accumulated_count 
        
        #If just data loss - return it
        if not include_regularization:
            return data_loss
        
        #Return the data and regularization loss
        return data_loss, self.regularization_loss()
    
    #Reset variables for accumulated loss
    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

#Cross-entropy loss
class Loss_CategoricalCrossentropy(Loss):
    
    #Forward pass
    def forward(self, y_pred, y_true ):
        
        #Number of samples in a batch
        samples = len(y_pred)
        
        #Clip data to prevent division by 0
        #Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        
        #Probabilities for target values - 
        #only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        
        #Mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1,)  
        
        #losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    #Backward pass
    def backward(self, dvalues, y_true):
        
        #Number of samples
        samples = len(dvalues)
        #Number of labesl in every sample
        #We'll use the first sample to count them
        labels = len(dvalues[0])
        
        #If labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        
        #Calculate gradient
        self.dinputs = -y_true / dvalues
        #Normalize gradient
        self.dinputs = self.dinputs / samples
        
#Binary Cross-entropy loss
class Loss_BinaryCrossEntropy(Loss):
    
    #Forward pass
    def forward(self, y_pred, y_true):
        
        #Clip data to prevent division by zero
        #Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1- 1e-7)
        
        #calculate sample wise loss
        sample_losses = -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
        sample_losses = np.mean(sample_losses, axis=1)
        
        #return loss
        return sample_losses

    #Backward pass
    def backward(self, dvalues, y_true):
        
        #Number of samples
        samples = len(dvalues)
        
        #Number of outputs in every sample
        #We'll use the first sample to count  them
        outputs = len(dvalues[0])
        
        #Clip data to prevent division by zero
        #Clip both sides to not drag mean towards any value
        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)
        
        #Calculate gradient
        self.dinputs = - (y_true / clipped_dvalues - (1- y_true) / (1- clipped_dvalues)) / outputs
        
        #Normalize gradient
        self.dinputs = self.dinputs / samples

#Mean Squarred Error loss
class Loss_MeanSquarredError(Loss):
    
    #Forward pass
    def forward(self, y_pred, y_true):
        
        #Calculate loss
        sample_losses = np.mean((y_true - y_pred)**2, axis=-1)
        
        #return losses
        return sample_losses
    
    #Backward pass
    def backward(self, dvalues, y_true):
        
        #Number of samples
        samples = len(dvalues)
        
        #Number of outputs in every sample
        #We'll use the first sample to count the,
        outputs = len(dvalues[0])
        
        #Gradient on values
        self.dinputs = -2 * (y_true *dvalues) / outputs
        #Normalize gradient
        self.dinputs = self.dinputs / samples

#Mean Absolute Error loss
class Loss_MeanAbsoluteError(Loss):
    
    #Forward pass
    def forward(self, y_pred, y_true):
        
        #Calculate loss
        sample_losses = np.mean((y_true - y_pred)**2, axis=-1)
        
        #return losses
        return sample_losses
    
    #Backward pass
    def backward(self, dvalues, y_true):
        
        #Number of samples
        samples = len(dvalues)
        
        #Number of outputs in every sample
        #We'll use the first sample to count the,
        outputs = len(dvalues[0])
        
        #Gradient on values
        self.dinputs = np.sign(y_true - dvalues) / outputs
        #Normalize gradient
        self.dinputs = self.dinputs / samples
                 
#Softmax classifier - combined Softmax activation
#and cross-entropy loss for faster backward step
class Activation_Softmax_Loss_CategoricalCrossentropy:
    
    #Backward pass
    def backward(self, dvalues, y_true):
        
        #Number of samples
        samples = len(dvalues)
        
        #If labels are one-hot encoded,
        #turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
            
        #Copy so we can safely modify
        self.dinputs = dvalues.copy()
        #Calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        #Normalize gradient
        self.dinputs = self.dinputs / samples

#SGD optimizer
class Optimizer_SGD:
    
    #Initialize optimizer - set settings,
    #learning rate of 1. is default for this optimizer
    def __init__(self, learning_rate=1., decay=0., momentum=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0 
        self.momentum = momentum
        
    #Call once before any parameters updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1 + self.decay * self.iterations))
            
    #Update parameters
    def update_params(self, layer):
        
        #If use momentum
        if self.momentum:
            
            #If layer does not conatin momentum arrays, create them
            #filled with zeros
            if not hasattr(layer, "weight_momentums"):
                layer.weight_momentums = np.zeros_like(layer.weights)
                #If there is no momentum array for weights
                #The array doesn't exist for biases yet either.
                layer.bias_momentums = np.zeros_like(layer.biases)
            
            #Build weight updates with momentum - take previous
            #updates multiplied by retain factor and update with
            #current gradients
            weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates
            
            #Build bias updates
            bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates
            
        #Vanilla SGD updates (as before momentum update)
        else:
            weight_updates = -self.current_learning_rate * layer.dweights
            
            bias_updates = -self.current_learning_rate * layer.dbiases
        
        #Update weights and biases using either
        #vanilla or momentum
        layer.weights += weight_updates
        layer.biases += bias_updates
        
    #Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1

#AdaGrad optimizer
class Optimizer_Adagrad:
    
    #Initialize optimizer - set settings,
    #learning rate of 1. is default for this optimizer
    def __init__(self, learning_rate=1., decay=0., epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0 
        self.epsilon = epsilon
        
    #Call once before any parameters updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1 + self.decay * self.iterations))
            
    #Update parameters
    def update_params(self, layer):
        
        #If layer does not conatin cache arrays,
        #create them filled with zeros
        if not hasattr(layer, "weight_cache"):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
        
        #Update cache with squared current gradients
        layer.weight_cache += layer.dweights ** 2
        layer.bias_cache += layer.dbiases ** 2
        
        #Vanilla SGD parameter update + normalization
        #with square rooted cache
        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)
        
        
    #Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1

#RMSprop optimizer
class Optimizer_RMSprop:
    
    #initialize optimizer - set settings
    def __init__(self, learning_rate=0.001, decay=0, epsilon=1e-7, rho=0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho
    
    #Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))
    
    #Update parameters
    def update_params(self, layer):
        
        #If layer does not contain cache arrays
        #create them filled with zeros
        if not hasattr(layer, "weight_cache"):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
        
        #Update cache woth squared current Gradients
        layer.weight_cache = self.rho * layer.weight_cache + (1 - self.rho) * layer.dweights**2
        layer.bias_cache = self.rho * layer.bias_cache + (1 - self.rho) * layer.dbiases **2
        
        #Vanilla SGD parameter update + normalization
        #with square rooted cache
        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

    #Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1 

#Adam optimizer
class Optimizer_Adam:
    
    #Initialize optimizer - set settings
    def __init__(self,learning_rate= 0.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    #Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))
    
    #Update parameters
    def update_params(self, layer):
        
        #If layer does not contain cache arrays,
        #create them filled with zeros
        if not hasattr(layer, "weight_cache"):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)
        
        #Update momentum with current gradient
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases
        
        #Get correct momentum
        #self.iteration is 0 at first pass
        #and we need to start with 1 here
        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        
        #Update cache with squarred current gradients 
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights **2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases **2
        
        #Get corrected cache
        weight_cache_corrected = layer.weight_cache / ( 1 -self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / ( 1 - self.beta_2 ** (self.iterations + 1)) 
        
        #Vanilla SGD parameter udpate + normalization
        #with square rooted cache
        layer.weights += - self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += - self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)
    
    #Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1   

#Common accuracy class
class Accuracy:
    
    #Calculates an accuracy
    #given predictions and ground truth values
    def calculate(self, predictions, y):
        
        #Get comparison results
        comparison = self.compare(predictions, y)
        
        #Calculate an accuracy
        accuracy = np.mean(comparison)
        
        #Add accumulated sum of matching values and sample count
        self.accumulated_sum += np.sum(comparison)
        self.accumulated_count += len(comparison)
        
        #return accuracy
        return accuracy
    
    #Calculates accumulated accuracy
    def calculate_accumulated(self):

        #Calculate an accuracy
        accuracy = self.accumulated_sum / self.accumulated_count
        
        #Return the adata and regularization_loss
        return accuracy
    
    #Reset variables for accumulated accuracy
    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0
        
#Accuracy calculation for regression model
class Accuracy_Regression(Accuracy):
    
    def __init__(self):
        
        #Create precision property
        self.precision = None
        
    #Calculates precision value
    #based on passed-in ground truth
    def init(self, y, reinit=False):
        if self.precision is None or reinit:
            self.precision = np.std(y) / 250
            
    #Compares predictions to the ground truth values
    def compare(self, predictions, y):
        return np.absolute(predictions -y) < self.precision
        
#Accuracy calculation for classification model
class Accuracy_Categorical(Accuracy): 
    
    def __init__(self,*, binary=False): 
        #Binary mode ?
        self.binary = binary
    
    #No initialization is needed
    def init(self, y): pass
    
    #Compare predictions to ground truth values
    def compare(self, predictions, y):
        if not self.binary and len(y.shape) == 2:
            y = np.argmax(y, axis =1)
        return predictions == y
#Model class
class Model:
    
    def __init__(self):
        #Create a list of network objects
        self.layers = []
        
        #Softmax classifier's output object
        self.softmax_classifier_output = None
        
        #Create a list of lost values
        self.loss_values = []

        #Create a list of accuracy values
        self.accuracy_values = []
    
    #Add objects to the model
    def add(self, layer):
        self.layers.append(layer)
    
    #Set loss and optimizer
    def set(self, *, loss=None, optimizer=None, accuracy=None): 
        
        if loss is not None:
            self.loss = loss
        
        if optimizer is not None: 
            self.optimizer = optimizer

        if accuracy is not None:
            self.accuracy = accuracy
        
    
    #Train the model
    def train(self, X, y, *, epochs=1, batch_size=None, print_every=1, validation_data=None): 
        
        #Initialize accuracy object
        self.accuracy.init(y) 
        
        # Create a list to store all the accuracies
        self.accuracies = []
        
        # Create a list to store the epochs
        self.epochs = []
        
        #Default value if batch size is not being set
        train_steps = 1
            
        #Calculate number of steps
        if batch_size is not None:
            train_steps = len(X) //batch_size
            
            #Dividing rounds down. If there are some remaining
            #data, but not a full batch, this won't include it
            #Add `1` to include this not full batch
            if train_steps * batch_size < len(X):
                train_steps += 1        
        
        #Main training loop
        for epoch in range(1, epochs+1): 
            
            #Print epoch number
            print(f"epoch: {epoch}")
            
            #Reset accumulated values in loss and accuracy objects
            self.loss.new_pass()
            self.accuracy.new_pass()
            
            #Iterate over steps 
            for step in range(train_steps):

                #If batch size is not set - 
                #train using one step and full dataset
                if batch_size is None:
                    batch_X = X
                    batch_y = y

                #Otherwise slice a batch
                else:
                    batch_X = X[step*batch_size: (step+1)*batch_size]
                    batch_y = y[step*batch_size: (step+1)*batch_size]
            
                #Perform forward pass
                output = self.forward(batch_X, training=True)
                
                #Calculate loss
                data_loss, regularization_loss = self.loss.calculate(output, batch_y, include_regularization=True)
                loss = data_loss + regularization_loss
                
                #Get predictions and calculate an accuracy
                predictions = self.output_layer_activation.predictions(output)
                
                accuracy = self.accuracy.calculate(predictions, batch_y) 
                
                # Store the current accuracy
                self.accuracies.append(accuracy)
                
                # Store the current epoch
                self.epochs.append(epoch)
                
                #Perform backward pass
                self.backward(output, batch_y)
                
                #Optimize (update parameters)
                self.optimizer.pre_update_params()
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)
                self.optimizer.post_update_params()
                
                #Print a summary
                if not step % print_every or step == train_steps -1:
                    print(f"step: {step}, "+
                        f"acc: {accuracy:.3f}, "+
                        f"loss: {loss:.3f}, "+ 
                        f"data_loss: {data_loss:.3f}, "+ 
                        f"reg_loss: {regularization_loss:.3f}, "+
                        f"lr: {self.optimizer.current_learning_rate}")
            
            #Get and print epoch loss and accuracy
            epoch_data_loss, epoch_regularization_loss = self.loss.calculate_accumulated(include_regularization=True)
            epoch_loss = epoch_data_loss + epoch_regularization_loss
            epoch_accuracy = self.accuracy.calculate_accumulated()
            
            print(f"training, "+
                  f"acc: {epoch_accuracy:.3f}, "+
                  f"loss: {epoch_loss:.3f}, "+ 
                  f"data_loss: {epoch_data_loss:.3f}, "+ 
                  f"reg_loss: {epoch_regularization_loss:.3f}, "+
                  f"lr: {self.optimizer.current_learning_rate}")
            
            #If there is validation data
            if validation_data is not None:
                
                #Evaluate the model          
                self.evaluate(*validation_data, batch_size=batch_size)
            
    #Finalize the model
    def finalize(self):
        
        #Create and set the input layer
        self.input_layer = Layer_Input()

        #Create and set the input layer
        layer_count = len(self.layers)
        
        #Initialize  a list containing trainable layers
        self.trainable_layers = []

        #Iterate the objects
        for i in range(layer_count):
            
            #If it's the first layer,
            #the previous layer object is the input layer
            if i == 0:   
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]
                
            #All layers except for the first and the last
            elif i < layer_count-1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]
                
            #The last layer - the next object is the loss
            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]
            
            #If layer contains an attribute called "weights"
            #it's a trainable layer -
            #add it to the list of trainable layers
            #We don't need to check for biases -
            #checking for weights is enough
            if hasattr(self.layers[i], "weights"):
                self.trainable_layers.append(self.layers[i])
            
            #Update loss object with trainable layers
            if self.loss is not None:
                self.loss.remember_trainable_layers(self.trainable_layers)
                
            #If output activation is Softmax
            #loss function is Categorical Cross-Entropy
            #create an object of combined activation
            #and loss function containing
            #faster gradient calculation
            if isinstance(self.layers[-1], Activation_Softmax) and isinstance(self.loss, Loss_CategoricalCrossentropy):
                #Create an object of combined activation
                #and loss function
                self.softmax_classifier_output = Activation_Softmax_Loss_CategoricalCrossentropy()
    
    #Predicts on the samples
    def predict(self, X, *, batch_size=None):

        #Default value if batch size is not being set
        prediction_steps = 1

        #Calculate number of steps
        if batch_size is not None:
            prediction_steps = len(X) // batch_size
            #Dividing rounds down. If there are some remaining
            #data, but not a full batch, this won't include it
            #Add `1` to include this not full batch
            if prediction_steps * batch_size < len(X):
                prediction_steps += 1
            
        #Model outputs
        output = []
        
        #Iterate over steps
        for step in range(prediction_steps):
            
            #If batch size is not set - 
            #train using one step and full dataset
            if batch_size is None:
                batch_X = X

            #Otherwise slice a batch
            else:
                batch_X = X[step*batch_size: (step+1)*batch_size]
            
            #Perform the forward pass
            batch_output = self.forward(batch_X, training=False)
            
            #Append batch prediction to the list of predictions
            output.append(batch_output)
        
        #Stack and return results
        return np.vstack(output)

            
    
    #Performs forward pass
    def forward(self, inputs, training):
        
        #Call forward method on the input layer
        #this will set the output property that
        #the first layer in "prev" object is expecting
        self.input_layer.forward(inputs, training)
        
        #Call forward method of every object in a chain
        #Pass output of the previous object as a parameter
        for layer in self.layers:
            layer.forward(layer.prev.output, training)
        
        #"layer" is now the last object from the list,
        #return its output
        return layer.output     
    
    #Performs backward pass
    def backward(self, output, y): 
        
        #If softmax classifier
        if self.softmax_classifier_output is not None: 
            #Fist call backward method
            #on the combined activation/ loss
            #this will set dinputs property
            self.softmax_classifier_output.backward(output, y)
            
            #Since we'll not call backward method of the last layer
            #which is Softmax activation
            #as we used activation/ loss
            #object, let's set dinputs property
            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs
            
            #Call backward method going through
            #all the objects but last
            #in reversed order passing dinputs as parameter
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)
                
            return
        
        #First call backward method on the loss
        #this will set dinputs property that the last
        #layer will try to access shortly
        self.loss.backward(output, y)
        
        #Call backward method going through all the objects
        #in reversed order passing dinputs as parameter
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)  
    
    #Evaluates the model using passed-in dataset        
    def evaluate(self, X_val, y_val, *, batch_size=None): 
        
        #Default value if bacht size is not being set
        validation_steps = 1

        #Calculate number of steps
        if batch_size is not None:
            validation_steps = len(X_val) // batch_size
            #Dividing rounds down, If there are some remaining
            #data, but not a full batch, this won't include it
            #Add `1` to include this not full batch
            if validation_steps * batch_size < len(X_val):
                validation_steps += 1
            
        #Reset accumulated values in loss
        #and accuracy objects
        self.loss.new_pass()
        self.accuracy.new_pass()
        
        #Iterate over steps
        for step in range(validation_steps):
            
            #If batch size is not set - 
            #train using one step and full dataset
            if batch_size is None:
                batch_X = X_val
                batch_y = y_val

            #Otherwise slice a batch
            else:
                batch_X = X_val[step*batch_size:(step+1)*batch_size]
                batch_y = y_val[step*batch_size:(step+1)*batch_size]
            
            #Perform the forward pass
            output = self.forward(batch_X, training=False)
            
            #Calculate the loss
            self.loss.calculate(output, batch_y)
            
            #Get predictions and calculate an accuracy
            predictions = self.output_layer_activation.predictions(output)
            self.accuracy.calculate(predictions, batch_y)
            
        #Get and print validation loss and accuracy
        validation_loss = self.loss.calculate_accumulated()
        validation_accuracy = self.accuracy.calculate_accumulated()
        
        #Print a summary
        print(f"validation, "+
                f"acc: {validation_accuracy:.3f}, "+
                f"loss: {validation_loss:.3f}, ")
    
    #Retrieves and returns parameters of trainable layers
    def get_parameters(self):

        #Create a list for parameters
        parameters = [] 
        
        #Iterable tainable layers and get their parameters  
        for layer in self.trainable_layers: 
            parameters.append(layer.get_parameters())
        
        #Return a list
        return parameters

    #Updates the model with new parameters
    def set_parameters(self, parameters):
        
        #Iterate over the parameters and layers
        #and update each layers with each set of parameters
        for parameter_set, layer in zip(parameters, self.trainable_layers):
            layer.set_parameters(*parameter_set)
    
    #Saves the parameters to a file
    def save_parameters(self, path):

        #Open a file in the binary-write mode 
        #and save parameters to it
        with open(path, "wb") as f:
            pickle.dump(self.get_parameters(), f)
    
    #Loads the weights and updates a model instance with them
    def load_parameters(self, path):

        #Open file in the binary-read mode
        #load weights and update trainable layers
        with open(path, "rb") as f:
            self.set_parameters(pickle.load(f))
    
    #Saves the model
    def save(self, path):

        #Make a copy of current model instance 
        model = copy.deepcopy(self)
        
        #Reset accumulated values in loss and accuracy objects
        model.loss.new_pass()
        model.accuracy.new_pass()
        
        #Remove data from input layer
        #and gradients from the loss object
        model.input_layer.__dict__.pop("output", None)
        model.loss.__dict__.pop("dinputs", None)
        
        #For each layer remove inputs, output and dinputs properties
        for layer in model.layers:
            for property in ["inputs", "output", "dinputs", "dweights", "dbiases"]:
                layer.__dict__.pop(property, None)
                
        #Open a file  in the binary-write mode and save the model
        with open(path, "wb") as f:
            pickle.dump(model, f)
    
    #Loads and returns a model 
    @staticmethod 
    def load(path):

        #Open a file in the binary-read mode, load a model
        with open(path, "rb") as f:
            model = pickle.load(f)

        #Return a model 
        return model 

#Loads a MNIST dataset
def load_mnist_dataset(dataset, path):
    #Scan all the directories and create a list of labels 
    labels = os.listdir(os.path.join(path, dataset))

    #create lists for samples and labels
    X = []
    y = [] 

    #For each label folder
    for label in labels: 
        #And for each image in given folder
        for file in os.listdir(os.path.join(path, dataset, label)):
            
            #Read the image
            image = cv2.imread(os.path.join(path, dataset, label, file), cv2.IMREAD_UNCHANGED)
            
            #And append it and a label to the lists
            X.append(image)
            y.append(label)
    
    return np.array(X), np.array(y).astype("uint8")

#MNIST dataset (train + test)
def create_data_mnist(path):
    
    #Load both sets seperatly 
    X, y = load_mnist_dataset("train", path)
    X_test, y_test = load_mnist_dataset("test", path)
    
    #And return all the data
    return X,y, X_test, y_test
    
