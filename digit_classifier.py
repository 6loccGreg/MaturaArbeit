import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from keras.datasets import mnist 
from neural_network import Model, Layer_Dense, Activation_ReLU, Activation_Softmax, Loss_CategoricalCrossentropy, Accuracy_Categorical, Optimizer_Adam


TRAIN = False

if TRAIN:
    (X, y), (X_test, y_test) = mnist.load_data()

    #Shuffle training dataset
    keys = np.array(range(X.shape[0]))
    np.random.shuffle(keys)
    X = X[keys]
    y = y[keys]

    #Scale and reshape samples
    X = (X.reshape(X.shape[0], -1).astype(np.float32) -127.5) /127.5
    X_test = (X_test.reshape(X_test.shape[0], -1).astype(np.float32) -127.5) /127.5


    #Instantiate the model 
    model = Model()

    #Add layers
    model.add(Layer_Dense(X.shape[1], 128))
    model.add(Activation_ReLU())
    model.add(Layer_Dense(128,128))
    model.add(Activation_ReLU())
    model.add(Layer_Dense(128,10)) 
    model.add(Activation_Softmax()) 

    #Set loss and optimizer object
    model.set(loss=Loss_CategoricalCrossentropy(), optimizer=Optimizer_Adam(decay=1e-3), accuracy=Accuracy_Categorical())

    #Finalize the model
    model.finalize()

    #Train the model
    model.train(X,y, validation_data= (X_test, y_test),epochs=3, batch_size=128, print_every=100)

    #save the model 
    model.save("./models/mnist_digits.model")
    
    #save the parameters
    model.save_parameters("./models/mnist_digits.params")
    
    fig = plt.figure()
    plt.scatter(model.get_accuracy_and_loss()[0], model.get_accuracy_and_loss()[1])
    plt.show()

else:
    #take_picture("pic1.jpg")

    #Read an image
    image_data = cv2.imread("sample_image_4.png", cv2.IMREAD_GRAYSCALE)

    #Resize to the same size as Fashion MNIST images
    image_data = cv2.resize(image_data, (28, 28))

    #Invert image colors
    image_data = 255 - image_data

    plt.imshow(image_data, cmap="gray")
    plt.show()
    #Reshape and scale pixel data
    image_data = (image_data.reshape(1, -1).astype(np.float32) - 127.5)/127.5

    #Load model
    model = Model.load("./models/mnist_digits.model")
    
    model.load_parameters("./models/mnist_digits.params")

    #Predict on the image  
    predictions = model.predict(image_data)

    #Get prediction instead of confidence level
    predictions = model.output_layer_activation.predictions(predictions)

    #Get label name from label index
    prediction = predictions[0]

    print(prediction)
