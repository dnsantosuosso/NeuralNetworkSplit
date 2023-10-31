import tensorflow as tf
from tensorflow import keras

def create_simple_nn():
    """
    Creates an arbitrary simple neural network with an input layer, a hidden layer, and an output layer.
    """
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(32,)),  # Input layer
        keras.layers.Dense(32, activation='relu'),  # Hidden layer
        keras.layers.Dense(10, activation='softmax')  # Output layer
    ])

    return model

def split_network(model, split_idx):
    """
    Splits a model at a specified layer index.
    Returns two new models.
    """
    # First model (up to split_idx - inclusive)
    layers_1 = [layer for i, layer in enumerate(model.layers) if i <= split_idx]
    model_1 = keras.Sequential(layers_1)
    for i, layer in enumerate(layers_1):
        model_1.layers[i].set_weights(layer.get_weights())

    # Second model (from split_idx+1 to the end)
    layers_2 = [layer for i, layer in enumerate(model.layers) if i > split_idx]
    model_2 = keras.Sequential(layers_2)
    for i, layer in enumerate(layers_2):
        model_2.layers[i].set_weights(layer.get_weights())

    return model_1, model_2

def main():
    original_model = create_simple_nn()

    model_1, model_2 = split_network(original_model, 1)

    # Test the split models
    input_data = tf.random.normal((1, 32))  # Example random input data
    output_1 = model_1.predict(input_data)
    output_2 = model_2.predict(output_1)

    # Validate by passing the input through the original model
    original_output = original_model.predict(input_data)

    print("Original Model Output:", original_output)
    print("Combined Outputs from Split Models:", output_2)

if __name__ == "__main__":
    main()