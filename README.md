# Data-Mining-HW2
This project is based on creating a model with Keras and TenserFlow libraries to cluster the given dataset by their label. The evaluation of the model is checked by graphs and calculated accuracy. After creating an accurate model with an acceptable accuracy, the model used for a bigger dataset. 

Given tasks are:
-	Finding the best number of nodes in a layer
-	Finding the best number of hidden layers
-	Add the suitable activation function
-	Learn how to work with TenserFlow and Keras
-	Change the learning rate 
-	Add the suitable loss function

Tech stack: 
Programming language: Python
Libraries used in this project: Pandas, Numpy, Matplotlib, Sklearn, TenserFlow, Keras
 

# Section1: Neural Network for Binary Classification

This repository contains the implementation of a simple neural network for binary classification using the Keras library. The primary goal is to demonstrate the impact of different configurations on model performance, particularly focusing on activation functions.

## Getting Started

To run the code in this repository, make sure you have the necessary dependencies installed. You can install them using the following command:

```bash
pip install numpy pandas scikit-learn keras tensorflow matplotlib
```

## Usage

The main script is provided in the Jupyter Notebook named `DM_HW2_9731084_code_seaction1.ipynb`. The notebook is divided into sections, each illustrating a different aspect of the neural network implementation.

1. **Dataset Creation and Visualization**
   - Utilizes the `make_circles` function from `sklearn.datasets` to generate 200 circles randomly.
   - Splits the dataset into training and testing sets.
   - Visualizes the circles using a scatter plot.

2. **Model Creation and Training**
   - Builds a neural network with an input layer, two hidden layers, and an output layer.
   - Demonstrates the impact of different activation functions on model performance.
   - Plots training accuracy and loss over epochs.

3. **Model Evaluation**
   - Evaluates the trained model on a test set.
   - Calculates and prints the accuracy of the model on the test data.

## Results

The neural network is initially built without any activation functions, resulting in poor performance. Subsequent models are created by introducing activation functions, such as ReLU and Sigmoid, which lead to improvements in accuracy. Additionally, experimenting with the loss function further enhances the model's predictive capabilities.

## Conclusion

This repository serves as a practical guide for understanding the importance of activation functions and loss functions in neural networks. Feel free to explore and modify the code to experiment with different configurations and further improve the model's performance.

