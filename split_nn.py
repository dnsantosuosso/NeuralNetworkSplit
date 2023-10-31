import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import plot_model


# Load the iris dataset from CSV
data = pd.read_csv('iris_data.csv')
X = data.drop('target', axis=1).values
y = data['target'].values

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data for better training performance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def create_simple_nn():
    """
    Creates an arbitrary simple neural network with an input layer, a hidden layer, and an output layer.
    """
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(4,)),  # Input layer
        keras.layers.Dense(32, activation='relu'),  # Hidden layer
        keras.layers.Dense(3, activation='softmax')  # Output layer

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


def trainSplitModels(model1, model2, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
    # Define loss function and optimizer
    loss_function = tf.keras.losses.SparseCategoricalCrossentropy()
    accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    optimizer = tf.keras.optimizers.Adam()

    # Build optimizer with all trainable variables
    optimizer.build(model1.trainable_variables + model2.trainable_variables)
    
    # Determine number of batches
    num_batches = int(np.ceil(len(X_train) / batch_size))

    # Training loop
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        
        total_loss = 0
        total_accuracy = 0
        
        for i in range(num_batches):
            # Get batch data
            start = i * batch_size
            end = min(start + batch_size, len(X_train))
            x_batch, y_batch = X_train[start:end], y_train[start:end]

            with tf.GradientTape(persistent=True) as tape:
                # Forward pass for model1 and model2
                output_1 = model1(x_batch, training=True)
                output_2 = model2(output_1, training=True)
                
                # Compute loss
                loss = loss_function(y_batch, output_2)
                total_loss += loss.numpy()
                
                # Update accuracy
                accuracy_metric(y_batch, output_2)
                accuracy = accuracy_metric.result().numpy()
                total_accuracy += accuracy
                
            # Compute gradients for model2
            grads_2 = tape.gradient(loss, model2.trainable_variables)

            # Compute gradients for the output of model1
            output_1_grad = tape.gradient(loss, output_1)

            # Compute gradients for model1
            grads_1 = tape.gradient(output_1, model1.trainable_variables, output_gradients=output_1_grad)

            # Update weights using gradients
            optimizer.apply_gradients(zip(grads_2, model2.trainable_variables))
            optimizer.apply_gradients(zip(grads_1, model1.trainable_variables))

        # Calculate average loss and accuracy over the training set
        average_loss = total_loss / num_batches
        average_accuracy = total_accuracy / num_batches
        
        # Compute validation loss and accuracy
        output_1_val = model1(X_val, training=False)
        output_2_val = model2(output_1_val, training=False)
        
        val_loss = loss_function(y_val, output_2_val).numpy()
        accuracy_metric.reset_states()
        val_accuracy = accuracy_metric(y_val, output_2_val).numpy()

        # Print the results
        print(f"{num_batches}/{num_batches} [==============================] - loss: {average_loss:.4f} - accuracy: {average_accuracy:.4f} - val_loss: {val_loss:.4f} - val_accuracy: {val_accuracy:.4f}")
        
        # Reset states of metrics for the next epoch
        accuracy_metric.reset_states()

    del tape  # Clean up the persistent gradient tape

def main():
    # Setting random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)

    #Create models
    original_model = create_simple_nn()
    model_1, model_2 = split_network(original_model, 1)

    #Add dummy inputs: TODO Why?
    dummy_input = tf.random.uniform((1, 4))  # or any appropriate shape
    model_1(dummy_input)
    model_2(model_1(dummy_input))

    # Train the split models
    trainSplitModels(model_1, model_2, X_train, y_train, X_test, y_test, epochs=10)

    # Compile the original model
    original_model.compile(optimizer='adam', 
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                           metrics=['accuracy'])
    
    print("******************************************************************")

    # Train the original model with consistent batch size
    original_model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32)

    print("******************************************************************")

    # Test the split models on any of the following inputs:
    input_data = tf.random.normal((1, 4))  # Example random input data
    input_data_setosa = np.array([[5.1, 3.5, 1.4, 0.2]]) # Setosa = [1,0,0]
    input_data_versicolour = np.array([[6.0, 2.7, 4.2, 1.3]]) # Versicolour = [0,1,0]
    input_data_virginica = np.array([[6.3, 3.3, 5.6, 2.0]]) # Virginica = [0,0,1]

    # Scale the data using the scaler you trained earlier
    input_data_scaled = scaler.transform(input_data_virginica)    #Change to test on different input

    output_1 = model_1.predict(input_data_scaled)
    output_2 = model_2.predict(output_1)

    # Validate by passing the input through the original model
    original_output = original_model.predict(input_data_scaled)

    print("Original Model Output:", original_output)
    print("Combined Outputs from Split Models:", output_2)

    #Print weights and biases
    models_list = [original_model, model_1, model_2]
    for model in models_list:
        for layer in model.layers:
            print(layer.name)
            print("Weights:")
            print(layer.get_weights()[0])  # weights
            print("Biases:")
            print(layer.get_weights()[1])  # biases
        print("******************************************************************")
    
    #Plot the models
    plot_model(original_model, to_file='original_model.png', show_shapes=True)
    plot_model(model_1, to_file='model_1.png', show_shapes=True)
    plot_model(model_2, to_file='model_2.png', show_shapes=True)


if __name__ == "__main__":
    main()