import tflearn
import tensorflow.compat.v1 as tf

# Simple TFLearn network for testing
def create_test_network():
    net = tflearn.input_data(shape=[None, 784])
    net = tflearn.fully_connected(net, 64, activation='relu')
    net = tflearn.fully_connected(net, 10, activation='softmax')
    net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy')
    return net

print("TFLearn and TensorFlow setup with TensorFlow 2.x compatibility - START")

# Create and test the network
tf.reset_default_graph()
network = create_test_network()
model = tflearn.DNN(network)

print("TFLearn and TensorFlow setup with TensorFlow 2.x compatibility - END")
