"""
Physics-Informed Neural Network (PINN) implementation for solving the Allen-Cahn equation
using discrete time inference approach with Implicit Runge-Kutta (IRK) methods.

@author: Maziar Raissi
"""

import sys  # Import system-specific parameters and functions
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../Utilities'))  # Add the Utilities directory to Python path for importing custom modules

import tensorflow as tf  # Import TensorFlow for deep learning computations
import numpy as np  # Import NumPy for numerical computations
import matplotlib.pyplot as plt  # Import matplotlib for plotting
import time  # Import time module for timing operations
import scipy.io  # Import scipy.io for loading MATLAB data files
from plotting import newfig, savefig  # Import custom plotting functions
import matplotlib.gridspec as gridspec  # Import gridspec for subplot layout
from mpl_toolkits.axes_grid1 import make_axes_locatable  # Import for colorbar positioning

np.random.seed(1234)  # Set random seed for NumPy to ensure reproducibility
tf.random.set_seed(1234)  # Set random seed for TensorFlow to ensure reproducibility


class PhysicsInformedNN:  # Define the main Physics-Informed Neural Network class
    # Initialize the class
    def __init__(self, x0, u0, x1, layers, dt, lb, ub, q):  # Constructor with input parameters
        
        self.lb = lb  # Store lower bound of spatial domain
        self.ub = ub  # Store upper bound of spatial domain
        
        self.x0 = x0  # Store initial spatial points (training data)
        self.x1 = x1  # Store boundary spatial points
        
        self.u0 = u0  # Store initial condition values at x0
        
        self.layers = layers  # Store neural network architecture (list of layer sizes)
        self.dt = dt  # Store time step size
        self.q = max(q,1)  # Store number of IRK stages (ensure at least 1)
    
        # Initialize NN
        self.weights, self.biases = self.initialize_NN(layers)  # Initialize neural network weights and biases
        
        # Load IRK weights
        tmp = np.float32(np.loadtxt('../../Utilities/IRK_weights/Butcher_IRK%d.txt' % (q), ndmin = 2))  # Load IRK weights from file
        self.IRK_weights = np.reshape(tmp[0:q**2+q], (q+1,q))  # Reshape IRK weights matrix
        self.IRK_times = tmp[q**2+q:]  # Extract IRK time points
        
        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,  # Create TensorFlow session
                                                     log_device_placement=True))  # Allow device placement logging
        
        self.x0_tf = tf.placeholder(tf.float32, shape=(None, self.x0.shape[1]))  # Placeholder for initial spatial points
        self.x1_tf = tf.placeholder(tf.float32, shape=(None, self.x1.shape[1]))  # Placeholder for boundary spatial points
        self.u0_tf = tf.placeholder(tf.float32, shape=(None, self.u0.shape[1]))  # Placeholder for initial condition values
        self.dummy_x0_tf = tf.placeholder(tf.float32, shape=(None, self.q))  # Dummy variable for forward gradients computation
        self.dummy_x1_tf = tf.placeholder(tf.float32, shape=(None, self.q+1))  # Dummy variable for forward gradients computation
        
        self.U0_pred = self.net_U0(self.x0_tf)  # N x (q+1) - Predict initial state using neural network
        self.U1_pred, self.U1_x_pred = self.net_U1(self.x1_tf)  # N1 x (q+1) - Predict boundary state and its derivative

        # Data loss: difference between predicted and actual initial condition
        data_loss = tf.reduce_sum(tf.square(self.u0_tf - self.U0_pred))  # Compute sum of squared differences
        # Boundary condition loss: continuity at boundaries
        boundary_loss = tf.reduce_sum(tf.square(self.U1_pred[0, :] - self.U1_pred[1, :]))  # Enforce continuity at boundaries
        # Boundary derivative loss: derivative continuity at boundaries
        boundary_derivative_loss = tf.reduce_sum(tf.square(self.U1_x_pred[0, :] - self.U1_x_pred[1, :]))  # Enforce derivative continuity

        self.loss = data_loss + boundary_loss + boundary_derivative_loss  # Total loss function
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,  # Create L-BFGS-B optimizer
                                                                method = 'L-BFGS-B',  # Use L-BFGS-B method
                                                                options = {'maxiter': 50000,  # Maximum iterations
                                                                           'maxfun': 50000,  # Maximum function evaluations
                                                                           'maxcor': 50,  # Maximum corrections
                                                                           'maxls': 50,  # Maximum line search steps
                                                                           'ftol' : 1.0 * np.finfo(float).eps})  # Function tolerance
        
        self.optimizer_Adam = tf.train.AdamOptimizer()  # Create Adam optimizer
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)  # Create Adam training operation
        
        init = tf.global_variables_initializer()  # Initialize all TensorFlow variables
        self.sess.run(init)  # Run variable initialization
        
    def initialize_NN(self, layers):  # Method to initialize neural network weights and biases
        weights = []  # Initialize empty list for weights
        biases = []  # Initialize empty list for biases
        num_layers = len(layers)  # Get total number of layers
        for l in range(0,num_layers-1):  # Loop through all layers except the last one
            W = self.xavier_init(size=[layers[l], layers[l+1]])  # Initialize weight matrix using Xavier initialization
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)  # Initialize bias vector with zeros
            weights.append(W)  # Add weight matrix to weights list
            biases.append(b)  # Add bias vector to biases list
        return weights, biases  # Return initialized weights and biases
        
    def xavier_init(self, size):  # Xavier/Glorot initialization method for weights
        in_dim = size[0]  # Get input dimension
        out_dim = size[1]  # Get output dimension
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))  # Calculate Xavier standard deviation
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)  # Return weight matrix with Xavier initialization
    
    def neural_net(self, X, weights, biases):  # Forward pass through neural network
        num_layers = len(weights) + 1  # Calculate total number of layers
        
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0  # Normalize input to [-1,1] range
        for l in range(0,num_layers-2):  # Loop through hidden layers
            W = weights[l]  # Get weight matrix for current layer
            b = biases[l]  # Get bias vector for current layer
            H = tf.tanh(tf.add(tf.matmul(H, W), b))  # Apply linear transformation and tanh activation
        W = weights[-1]  # Get weight matrix for output layer
        b = biases[-1]  # Get bias vector for output layer
        Y = tf.add(tf.matmul(H, W), b)  # Apply linear transformation for output (no activation)
        return Y  # Return neural network output
    
    def fwd_gradients_0(self, U, x):  # Forward mode automatic differentiation for initial points
        g = tf.gradients(U, x, grad_ys=self.dummy_x0_tf)[0]  # Compute gradient of U with respect to x
        return tf.gradients(g, self.dummy_x0_tf)[0]  # Compute second derivative using forward mode
    
    def fwd_gradients_1(self, U, x):  # Forward mode automatic differentiation for boundary points
        g = tf.gradients(U, x, grad_ys=self.dummy_x1_tf)[0]  # Compute gradient of U with respect to x
        return tf.gradients(g, self.dummy_x1_tf)[0]  # Compute second derivative using forward mode
    
    def net_U0(self, x):  # Neural network for predicting initial state U0
        U1 = self.neural_net(x, self.weights, self.biases)  # Forward pass through neural network
        U = U1[:,:-1]  # Extract all columns except the last one (IRK stages)
        U_x = self.fwd_gradients_0(U, x)  # Compute first spatial derivative
        U_xx = self.fwd_gradients_0(U_x, x)  # Compute second spatial derivative
        F = 5.0*U - 5.0*U**3 + 0.0001*U_xx  # Allen-Cahn equation: F = 5u - 5u^3 + 0.0001*u_xx
        U0 = U1 - self.dt*tf.matmul(F, self.IRK_weights.T)  # Apply IRK time stepping to get initial state
        return U0  # Return predicted initial state

    def net_U1(self, x):  # Neural network for predicting boundary state U1
        U1 = self.neural_net(x, self.weights, self.biases)  # Forward pass through neural network
        U1_x = self.fwd_gradients_1(U1, x)  # Compute spatial derivative of U1
        return U1, U1_x  # Return predicted boundary state and its derivative
    
    def callback(self, loss):  # Callback function for optimizer
        print('Loss:', loss)  # Print current loss value
    
    def train(self, nIter):  # Training method for the neural network
        tf_dict = {self.x0_tf: self.x0, self.u0_tf: self.u0, self.x1_tf: self.x1,  # Create feed dictionary for TensorFlow
                   self.dummy_x0_tf: np.ones((self.x0.shape[0], self.q)),  # Dummy variables for forward gradients
                   self.dummy_x1_tf: np.ones((self.x1.shape[0], self.q+1))}  # Dummy variables for forward gradients
        
        start_time = time.time()  # Record start time for timing
        for it in range(nIter):  # Loop through training iterations
            self.sess.run(self.train_op_Adam, tf_dict)  # Run Adam optimizer step
            
            # Print
            if it % 10 == 0:  # Print every 10 iterations
                elapsed = time.time() - start_time  # Calculate elapsed time
                loss_value = self.sess.run(self.loss, tf_dict)  # Compute current loss
                print('It: %d, Loss: %.3e, Time: %.2f' %   # Print iteration, loss, and time
                      (it, loss_value, elapsed))
                start_time = time.time()  # Reset start time
    
        self.optimizer.minimize(self.sess,  # Run L-BFGS-B optimizer
                                feed_dict = tf_dict,  # Provide feed dictionary
                                fetches = [self.loss],  # Fetch loss value
                                loss_callback = self.callback)  # Use callback function
    
    def predict(self, x_star):  # Prediction method for new spatial points
        
        U1_star = self.sess.run(self.U1_pred, {self.x1_tf: x_star})  # Run prediction on new points
                    
        return U1_star  # Return predicted values

    
if __name__ == "__main__":  # Main execution block - runs when script is executed directly
        
    q = 100  # Number of IRK stages for time integration
    layers = [1, 200, 200, 200, 200, q+1]  # Neural network architecture: [input, hidden1, hidden2, hidden3, hidden4, output]
    lb = np.array([-1.0])  # Lower bound of spatial domain
    ub = np.array([1.0])  # Upper bound of spatial domain
    
    N = 200  # Number of training data points
    
    data = scipy.io.loadmat('../Data/AC.mat')  # Load Allen-Cahn equation data from MATLAB file
    
    t = data['tt'].flatten()[:,None] # T x 1 - Extract time points and reshape to column vector
    x = data['x'].flatten()[:,None] # N x 1 - Extract spatial points and reshape to column vector
    Exact = np.real(data['uu']).T # T x N - Extract exact solution and transpose to get T x N format
    
    idx_t0 = 20  # Index for initial time point
    idx_t1 = 180  # Index for final time point
    dt = t[idx_t1] - t[idx_t0]  # Calculate time step size
    
    # Initial data
    noise_u0 = 0.0  # Noise level for initial condition (0 = no noise)
    idx_x = np.random.choice(Exact.shape[1], N, replace=False)  # Randomly select N spatial points for training
    x0 = x[idx_x,:]  # Extract selected spatial points
    u0 = Exact[idx_t0:idx_t0+1,idx_x].T  # Extract initial condition values at selected points
    u0 = u0 + noise_u0*np.std(u0)*np.random.randn(u0.shape[0], u0.shape[1])  # Add noise to initial condition
    
       
    # Boundary data
    x1 = np.vstack((lb,ub))  # Create boundary points array [lower_bound, upper_bound]
    
    # Test data
    x_star = x  # Use all spatial points for testing

    model = PhysicsInformedNN(x0, u0, x1, layers, dt, lb, ub, q)  # Create PINN model instance
    model.train(10000)  # Train the model for 10000 iterations
    
    U1_pred = model.predict(x_star)  # Make predictions on test points

    error = np.linalg.norm(U1_pred[:,-1] - Exact[idx_t1,:], 2)/np.linalg.norm(Exact[idx_t1,:], 2)  # Calculate relative L2 error
    print('Error: %e' % (error))  # Print the error

    ######################################################################
    ############################# Plotting ###############################
    ######################################################################    

    fig, ax = newfig(1.0, 1.2)  # Create new figure with custom size
    ax.axis('off')  # Turn off axis display
    
    ####### Row 0: h(t,x) ##################    
    gs0 = gridspec.GridSpec(1, 2)  # Create grid specification for subplots
    gs0.update(top=1-0.06, bottom=1-1/2 + 0.1, left=0.15, right=0.85, wspace=0)  # Update grid layout
    ax = plt.subplot(gs0[:, :])  # Create subplot spanning full grid
    
    h = ax.imshow(Exact.T, interpolation='nearest', cmap='seismic',   # Display exact solution as heatmap
                  extent=[t.min(), t.max(), x_star.min(), x_star.max()],   # Set extent for time and space axes
                  origin='lower', aspect='auto')  # Set origin to lower and auto aspect ratio
    divider = make_axes_locatable(ax)  # Create divider for colorbar
    cax = divider.append_axes("right", size="5%", pad=0.05)  # Add colorbar axis
    fig.colorbar(h, cax=cax)  # Add colorbar to the plot
        
    line = np.linspace(x.min(), x.max(), 2)[:,None]  # Create vertical line coordinates
    ax.plot(t[idx_t0]*np.ones((2,1)), line, 'w-', linewidth = 1)  # Plot vertical line at initial time
    ax.plot(t[idx_t1]*np.ones((2,1)), line, 'w-', linewidth = 1)  # Plot vertical line at final time
    
    ax.set_xlabel('$t$')  # Set x-axis label
    ax.set_ylabel('$x$')  # Set y-axis label
    leg = ax.legend(frameon=False, loc = 'best')  # Add legend
    ax.set_title('$u(t,x)$', fontsize = 10)  # Set plot title
    
    
    ####### Row 1: h(t,x) slices ##################    
    gs1 = gridspec.GridSpec(1, 2)  # Create grid specification for bottom subplots
    gs1.update(top=1-1/2-0.05, bottom=0.15, left=0.15, right=0.85, wspace=0.5)  # Update grid layout
    
    ax = plt.subplot(gs1[0, 0])  # Create left subplot
    ax.plot(x,Exact[idx_t0,:], 'b-', linewidth = 2)  # Plot exact solution at initial time
    ax.plot(x0, u0, 'rx', linewidth = 2, label = 'Data')  # Plot training data points
    ax.set_xlabel('$x$')  # Set x-axis label
    ax.set_ylabel('$u(t,x)$')  # Set y-axis label
    ax.set_title('$t = %.2f$' % (t[idx_t0]), fontsize = 10)  # Set subplot title
    ax.set_xlim([lb-0.1, ub+0.1])  # Set x-axis limits
    ax.legend(loc='upper center', bbox_to_anchor=(0.8, -0.3), ncol=2, frameon=False)  # Add legend


    ax = plt.subplot(gs1[0, 1])  # Create right subplot
    ax.plot(x,Exact[idx_t1,:], 'b-', linewidth = 2, label = 'Exact')  # Plot exact solution at final time
    ax.plot(x_star, U1_pred[:,-1], 'r--', linewidth = 2, label = 'Prediction')  # Plot PINN prediction
    ax.set_xlabel('$x$')  # Set x-axis label
    ax.set_ylabel('$u(t,x)$')  # Set y-axis label
    ax.set_title('$t = %.2f$' % (t[idx_t1]), fontsize = 10)  # Set subplot title
    ax.set_xlim([lb-0.1, ub+0.1])  # Set x-axis limits
    
    ax.legend(loc='upper center', bbox_to_anchor=(0.1, -0.3), ncol=2, frameon=False)  # Add legend
    
    # savefig('./figures/AC')  # Save figure (commented out)  
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    