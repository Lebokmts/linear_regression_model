# Linear Regression Model in Rust

# Project Overview
This project implements a simple Linear Regression Model in Rust using the burn library. The goal is to predict values for the equation: 
y = 2x + 1 (with added noise).

We use synthetic data for training and visualize results using text-based plots.

# Setup and installation
1: Prerequisites 
Ensure you have the following installed:

Rust & Cargo → Install Rust

Rust Rover (IDE) → Download Rust Rover

Git → Install Git

# How it works

- We generate random (x, y) pairs using:  let y = 2.0 * x + 1.0 + noise;

- Noise is added to simulate real-world data variance.

- We initialize weights & bias:

    struct LinearRegression {

       weight: Tensor<NdArray, 1>,
    
       bias: Tensor<NdArray, 1>,
    
    }

- Loss function: Mean Squared Error (MSE)

- Optimizer: Stochastic Gradient Descent (SGD)

- Training runs for 100 epochs, updating weights to minimize loss.

- The model predicts y for new x values.

- We compare actual vs. predicted values.

- The textplots crate creates a text-based graph comparing

# Sample output

Epoch 0: Loss = 8.473

Epoch 10: Loss = 2.142

Epoch 20: Loss = 0.987

Epoch 30: Loss = 0.458

Epoch 90: Loss = 0.012

Training complete!

# Model predictions

For x = 2.0, predicted y = 4.05

For x = 4.0, predicted y = 9.02

For x = 6.0, predicted y = 13.99

For x = 8.0, predicted y = 17.98

# Thoughts on this project

I had a bit of challenges especially getting the code to run. This weeks assignment was a hectic one considering that I am not familiar with rust, it was my very first time interacting with it

I tried my best to get it done
