# Neural Network Advanced Techniques Library

This library provides two advanced techniques for neural network implementation and optimization:

1. **Splitting Neural Networks (`split_nn.py`):** This allows for a neural network model to be split into two separate models at a specified layer.
2. **Knowledge Distillation (`knowledge_distillation_nn.py`):** Knowledge distillation trains a smaller student model using the knowledge of a larger teacher model, typically resulting in the student model achieving higher performance than if trained directly.

# Neural Network Splitter

This project demonstrates how to split a neural network into multiple sub-networks using TensorFlow and Keras.

## Installation

1. Clone this repository: `git clone https://github.com/dnsantosuosso/NeuralNetworkTechniques.git`
2. Install the required Python packages: `pip install -r requirements.txt`
3. Install Graphviz (for NN visualization):

- On macOS: `brew install graphviz`
- On Linux (Debian/Ubuntu): `sudo apt-get install graphviz`
- On Windows: Download and install from [Graphviz's Download Page](https://graphviz.gitlab.io/download/)

## Run the Code

1. Navigate to the project directory: `cd techniques`
2. Run the code: `python split_nn.py`
