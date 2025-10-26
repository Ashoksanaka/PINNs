"""
Physics-Informed Neural Networks for Korteweg-de Vries Equation Identification
@author: Maziar Raissi

This script implements a PINN to identify the parameters of the Korteweg-de Vries (KdV) equation
using discrete time stepping with Implicit Runge-Kutta (IRK) methods. The KdV equation:
u_t + λ₁uu_x + λ₂u_xxx = 0

The network learns to predict the solution at two time points while simultaneously discovering
the unknown parameters λ₁ and λ₂. This approach uses discrete time stepping rather than
continuous time derivatives, making it more robust for certain types of data.
"""

# Add the Utilities directory to Python path for importing custom plotting functions
import sys  # System-specific parameters and functions
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../Utilities'))  # Insert Utilities directory at beginning of Python path

# Core libraries for deep learning and scientific computing
import tensorflow as tf  # Deep learning framework for neural network implementation
import numpy as np       # Numerical computing library for array operations
import matplotlib.pyplot as plt  # Plotting library for visualization
import time              # For timing operations and measuring training duration
import scipy.io          # For loading .mat files containing numerical data
from plotting import newfig, savefig  # Custom plotting functions for publication-quality figures
from mpl_toolkits.mplot3d import Axes3D  # 3D plotting capabilities
import matplotlib.gridspec as gridspec  # For subplot layout and positioning
from mpl_toolkits.axes_grid1 import make_axes_locatable  # For colorbar positioning

# Set random seeds for reproducibility across different runs
np.random.seed(1234)  # Set NumPy random seed to ensure consistent random number generation
tf.random.set_seed(1234)  # Set TensorFlow random seed for reproducible neural network initialization


class PhysicsInformedNN:
    """
    Physics-Informed Neural Network for Korteweg-de Vries equation identification using discrete time stepping.
    
    This class implements a PINN that learns to identify unknown parameters in the KdV equation:
    u_t + λ₁uu_x + λ₂u_xxx = 0
    
    The key innovation is using discrete time stepping with Implicit Runge-Kutta (IRK) methods
    instead of continuous time derivatives, making it more robust for certain types of data.
    """
    
    def __init__(self, x0, u0, x1, u1, layers, dt, lb, ub, q):
        """
        Initialize the Physics-Informed Neural Network for discrete-time KdV identification.
        
        Args:
            x0: Spatial coordinates at time t₀ (N0 x 1)
            u0: Solution values at time t₀ (N0 x 1)
            x1: Spatial coordinates at time t₁ (N1 x 1)
            u1: Solution values at time t₁ (N1 x 1)
            layers: List defining neural network architecture [input_dim, hidden_dims..., output_dim]
            dt: Time step size (t₁ - t₀)
            lb: Lower bounds for spatial domain
            ub: Upper bounds for spatial domain
            q: Number of IRK stages (order of accuracy)
        """
        
        # Store domain bounds for input normalization
        self.lb = lb  # Lower bounds for spatial domain [x_min]
        self.ub = ub  # Upper bounds for spatial domain [x_max]
        
        # Store spatial coordinates at both time points
        self.x0 = x0  # Spatial coordinates at initial time t₀ (N0 x 1)
        self.x1 = x1  # Spatial coordinates at final time t₁ (N1 x 1)
        
        # Store solution data at both time points
        self.u0 = u0  # Solution values at initial time t₀ (N0 x 1)
        self.u1 = u1  # Solution values at final time t₁ (N1 x 1)
        
        # Store network architecture and time stepping parameters
        self.layers = layers  # Neural network architecture [input_dim, hidden_dims..., output_dim]
        self.dt = dt          # Time step size (t₁ - t₀) for discrete time stepping
        self.q = max(q,1)     # Number of IRK stages (minimum 1 for stability)
    
        # Initialize neural network weights and biases using Xavier initialization
        self.weights, self.biases = self.initialize_NN(layers)  # Call initialization method
        
        # Initialize unknown parameters to be learned during training
        # lambda_1: coefficient for nonlinear term uu_x (true value should be 1.0)
        # lambda_2: coefficient for dispersive term u_xxx (true value should be 0.0025)
        self.lambda_1 = tf.Variable([0.0], dtype=tf.float32)   # Linear parameter, initialized to 0
        self.lambda_2 = tf.Variable([-6.0], dtype=tf.float32) # Log parameter (exp(-6) ≈ 0.0025)
        
        # Load Implicit Runge-Kutta weights from external file
        # IRK methods provide high-order accuracy for stiff ODEs and better stability
        tmp = np.float32(np.loadtxt('../../Utilities/IRK_weights/Butcher_IRK%d.txt' % (q), ndmin = 2))  # Load IRK weights as float32
        weights =  np.reshape(tmp[0:q**2+q], (q+1,q))     # Reshape first q²+q elements into (q+1) x q matrix
        self.IRK_alpha = weights[0:-1,:]  # Stage weights for intermediate stages (q x q matrix)
        self.IRK_beta = weights[-1:,:]    # Final stage weights (1 x q matrix)
        self.IRK_times = tmp[q**2+q:]     # Time points for stages (remaining elements)
        
        # Create TensorFlow session with device placement logging for debugging
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,  # Allow CPU fallback if GPU unavailable
                                                     log_device_placement=True))  # Log which device each operation uses
        
        # Define TensorFlow placeholders for input data during training
        self.x0_tf = tf.placeholder(tf.float32, shape=(None, self.x0.shape[1]))  # Spatial coords at t₀
        self.x1_tf = tf.placeholder(tf.float32, shape=(None, self.x1.shape[1]))  # Spatial coords at t₁
        self.u0_tf = tf.placeholder(tf.float32, shape=(None, self.u0.shape[1]))  # Solution values at t₀
        self.u1_tf = tf.placeholder(tf.float32, shape=(None, self.u1.shape[1]))  # Solution values at t₁
        
        # Dummy variables for forward-mode automatic differentiation
        # These are needed for computing higher-order derivatives efficiently in discrete time
        self.dummy_x0_tf = tf.placeholder(tf.float32, shape=(None, self.q))  # For t₀ derivatives (N0 x q)
        self.dummy_x1_tf = tf.placeholder(tf.float32, shape=(None, self.q))  # For t₁ derivatives (N1 x q)
        
        # Build neural network predictions using discrete time stepping with IRK methods
        self.U0_pred = self.net_U0(self.x0_tf)  # Predicted solution at t₀ (N0 x q)
        self.U1_pred = self.net_U1(self.x1_tf)  # Predicted solution at t₁ (N1 x q)
        
        # Define loss function: MSE between predicted and observed solutions
        # This is simpler than continuous-time PINNs since we only fit data points
        # No physics residuals needed - physics is encoded in the discrete time stepping
        # Define loss as the sum of MSE at t₀ and t₁
        self.loss = (tf.reduce_sum(tf.square(self.u0_tf - self.U0_pred)) +    # MSE at t₀
                     tf.reduce_sum(tf.square(self.u1_tf - self.U1_pred)))     # MSE at t₁

        # Set up L-BFGS-B optimizer for fine-tuning (quasi-Newton method)
        # L-BFGS-B is a limited-memory quasi-Newton method that's very effective for PINNs
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,  # Loss function to minimize
                                                                method = 'L-BFGS-B',  # Optimization method
                                                                options = {'maxiter': 50000,  # Maximum iterations
                                                                           'maxfun': 50000,   # Maximum function evaluations
                                                                           'maxcor': 50,      # Maximum corrections stored
                                                                           'maxls': 50,       # Maximum line search steps
                                                                           'ftol' : 1.0 * np.finfo(float).eps})  # Function tolerance
        
        # Set up Adam optimizer for initial training (gradient descent with adaptive learning rates)
        self.optimizer_Adam = tf.train.AdamOptimizer()  # Create Adam optimizer with default parameters
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)  # Create training operation
        
        # Initialize all TensorFlow variables (weights, biases, parameters)
        init = tf.global_variables_initializer()  # Create initialization operation
        self.sess.run(init)  # Run initialization to set initial values
        
    def initialize_NN(self, layers):
        """
        Initialize neural network weights and biases using Xavier initialization.
        
        Args:
            layers: List defining network architecture [input_dim, hidden_dims..., output_dim]
            
        Returns:
            weights: List of weight matrices for each layer
            biases: List of bias vectors for each layer
        """
        weights = []  # List to store weight matrices for each layer
        biases = []   # List to store bias vectors for each layer
        num_layers = len(layers)  # Total number of layers in the network
        
        # Initialize weights and biases for each layer (except output layer)
        for l in range(0,num_layers-1):  # Loop through all layers except the last one
            # Initialize weights using Xavier/Glorot initialization for better gradient flow
            W = self.xavier_init(size=[layers[l], layers[l+1]])  # Create weight matrix
            # Initialize biases to zero (common practice)
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)  # Create bias vector
            weights.append(W)  # Add weight matrix to list
            biases.append(b)   # Add bias vector to list
        return weights, biases  # Return both lists
        
    def xavier_init(self, size):
        """
        Xavier/Glorot initialization for neural network weights.
        
        This initialization method helps prevent vanishing/exploding gradients by
        scaling the initial weights based on the number of input and output neurons.
        The weights are sampled from a truncated normal distribution.
        
        Args:
            size: [input_dimension, output_dimension] - dimensions of weight matrix
            
        Returns:
            tf.Variable: Randomly initialized weight matrix with Xavier scaling
        """
        in_dim = size[0]   # Input dimension (number of neurons in previous layer)
        out_dim = size[1]  # Output dimension (number of neurons in current layer)
        
        # Calculate standard deviation for Xavier initialization
        # Formula: sqrt(2 / (fan_in + fan_out)) where fan_in and fan_out are input/output dimensions
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))  # Compute scaling factor
        
        # Create weight matrix with truncated normal distribution
        # Truncated normal prevents extreme values that could cause training instability
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    
    def neural_net(self, X, weights, biases):
        """
        Forward pass through the neural network with tanh activation.
        
        This method performs the standard neural network forward pass:
        1. Normalize inputs to [-1, 1] range
        2. Apply linear transformation + tanh activation for hidden layers
        3. Apply linear transformation only for output layer
        
        Args:
            X: Input tensor (batch_size, input_dim) - spatial coordinates
            weights: List of weight matrices for each layer
            biases: List of bias vectors for each layer
            
        Returns:
            Y: Output tensor (batch_size, output_dim) - predicted solution
        """
        num_layers = len(weights) + 1  # Total number of layers (weights + output layer)
        
        # Normalize input features to [-1, 1] range for better training stability
        # Formula: 2 * (X - min) / (max - min) - 1
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0  # Normalize to [-1, 1]
        
        # Forward pass through hidden layers with tanh activation
        for l in range(0,num_layers-2):  # Loop through all hidden layers
            W = weights[l]  # Weight matrix for layer l
            b = biases[l]   # Bias vector for layer l
            # Linear transformation: H * W + b, then apply tanh activation
            H = tf.tanh(tf.add(tf.matmul(H, W), b))  # tanh activation for smooth gradients
        
        # Output layer (no activation function - linear output)
        W = weights[-1]  # Final weight matrix
        b = biases[-1]   # Final bias vector
        Y = tf.add(tf.matmul(H, W), b)  # Linear transformation only (no activation)
        return Y  # Return final output
    
    def fwd_gradients_0(self, U, x):
        """
        Forward-mode automatic differentiation for computing spatial derivatives at t₀.
        
        This method uses forward-mode differentiation to efficiently compute higher-order
        spatial derivatives needed for the KdV equation. Forward-mode is more efficient
        than reverse-mode for computing derivatives of scalar functions with respect to
        multiple variables (here: multiple IRK stages).
        
        Args:
            U: Function to differentiate (neural network output)
            x: Variable to differentiate with respect to (spatial coordinates)
            
        Returns:
            Second-order derivative tensor for use in discrete time stepping
        """
        g = tf.gradients(U, x, grad_ys=self.dummy_x0_tf)[0]  # First derivative using dummy variables
        return tf.gradients(g, self.dummy_x0_tf)[0]  # Second derivative (chain rule)
    
    def fwd_gradients_1(self, U, x):
        """
        Forward-mode automatic differentiation for computing spatial derivatives at t₁.
        
        Similar to fwd_gradients_0 but uses dummy variables for t₁ time point.
        This allows efficient computation of derivatives needed for the discrete
        time stepping scheme.
        
        Args:
            U: Function to differentiate (neural network output)
            x: Variable to differentiate with respect to (spatial coordinates)
            
        Returns:
            Second-order derivative tensor for use in discrete time stepping
        """
        g = tf.gradients(U, x, grad_ys=self.dummy_x1_tf)[0]  # First derivative using dummy variables
        return tf.gradients(g, self.dummy_x1_tf)[0]  # Second derivative (chain rule)    
    
    def net_U0(self, x):
        """
        Neural network for predicting solution at initial time t₀ using discrete time stepping.
        
        This method implements the discrete time stepping scheme for the KdV equation:
        u_t + λ₁uu_x + λ₂u_xxx = 0
        
        The discrete scheme uses Implicit Runge-Kutta (IRK) methods to advance the solution
        from intermediate stages to the initial time point.
        
        Args:
            x: Spatial coordinates tensor (N0 x 1)
            
        Returns:
            U0: Predicted solution at t₀ (N0 x q) where q is number of IRK stages
        """
        lambda_1 = self.lambda_1  # Get nonlinear coefficient (learnable parameter)
        lambda_2 = tf.exp(self.lambda_2)  # Get dispersive coefficient (exp of log parameter)
        U = self.neural_net(x, self.weights, self.biases)  # Neural network prediction
        
        # Compute spatial derivatives using forward-mode automatic differentiation
        U_x = self.fwd_gradients_0(U, x)    # First spatial derivative ∂U/∂x
        U_xx = self.fwd_gradients_0(U_x, x)  # Second spatial derivative ∂²U/∂x²
        U_xxx = self.fwd_gradients_0(U_xx, x)  # Third spatial derivative ∂³U/∂x³
        
        # Compute KdV equation right-hand side: F = -λ₁UU_x - λ₂U_xxx
        F = -lambda_1*U*U_x - lambda_2*U_xxx  # Nonlinear + dispersive terms
        
        # Apply discrete time stepping using IRK method
        # U₀ = U - dt * F * α^T where α are IRK stage weights
        U0 = U - self.dt*tf.matmul(F, self.IRK_alpha.T)  # Discrete time step backward
        return U0  # Return predicted solution at t₀
    
    def net_U1(self, x):
        """
        Neural network for predicting solution at final time t₁ using discrete time stepping.
        
        This method implements the discrete time stepping scheme for the KdV equation:
        u_t + λ₁uu_x + λ₂u_xxx = 0
        
        The discrete scheme uses Implicit Runge-Kutta (IRK) methods to advance the solution
        from intermediate stages to the final time point.
        
        Args:
            x: Spatial coordinates tensor (N1 x 1)
            
        Returns:
            U1: Predicted solution at t₁ (N1 x q) where q is number of IRK stages
        """
        lambda_1 = self.lambda_1  # Get nonlinear coefficient (learnable parameter)
        lambda_2 = tf.exp(self.lambda_2)  # Get dispersive coefficient (exp of log parameter)
        U = self.neural_net(x, self.weights, self.biases)  # Neural network prediction
        
        # Compute spatial derivatives using forward-mode automatic differentiation
        U_x = self.fwd_gradients_1(U, x)    # First spatial derivative ∂U/∂x
        U_xx = self.fwd_gradients_1(U_x, x)  # Second spatial derivative ∂²U/∂x²
        U_xxx = self.fwd_gradients_1(U_xx, x)  # Third spatial derivative ∂³U/∂x³
        
        # Compute KdV equation right-hand side: F = -λ₁UU_x - λ₂U_xxx
        F = -lambda_1*U*U_x - lambda_2*U_xxx  # Nonlinear + dispersive terms
        
        # Apply discrete time stepping using IRK method
        # U₁ = U + dt * F * (β - α)^T where β,α are IRK stage weights
        U1 = U + self.dt*tf.matmul(F, (self.IRK_beta - self.IRK_alpha).T)  # Discrete time step forward
        return U1  # Return predicted solution at t₁

    def callback(self, loss):
        """
        Callback function for L-BFGS-B optimizer to print training progress.
        
        This function is called by the L-BFGS-B optimizer during fine-tuning to
        monitor the loss value and provide feedback on convergence.
        
        Args:
            loss: Current loss value from the optimizer
        """
        print('Loss:', loss)  # Print current loss value
    
    def train(self, nIter):
        """
        Train the Physics-Informed Neural Network using two-stage optimization.
        
        Uses a two-stage training approach:
        1. Adam optimizer for initial training (fast convergence)
        2. L-BFGS-B optimizer for fine-tuning (better final accuracy)
        
        Args:
            nIter: Number of Adam iterations before switching to L-BFGS-B
        """
        # Prepare training data dictionary for TensorFlow session
        tf_dict = {self.x0_tf: self.x0, self.u0_tf: self.u0,  # Initial time data
                   self.x1_tf: self.x1, self.u1_tf: self.u1,  # Final time data
                   self.dummy_x0_tf: np.ones((self.x0.shape[0], self.q)),  # Dummy variables for t₀
                   self.dummy_x1_tf: np.ones((self.x1.shape[0], self.q))}  # Dummy variables for t₁
                           
        # Stage 1: Adam optimization for initial training
        start_time = time.time()  # Start timing for performance monitoring
        for it in range(nIter):  # Loop through Adam iterations
            self.sess.run(self.train_op_Adam, tf_dict)  # Run one Adam optimization step
            
            # Print progress every 10 iterations
            if it % 10 == 0:  # Check if it's time to print progress
                elapsed = time.time() - start_time  # Calculate elapsed time
                loss_value = self.sess.run(self.loss, tf_dict)  # Get current loss
                lambda_1_value = self.sess.run(self.lambda_1)  # Get current λ₁
                lambda_2_value = np.exp(self.sess.run(self.lambda_2))  # Get current λ₂ (exp of log)
                print('It: %d, Loss: %.3e, l1: %.3f, l2: %.5f, Time: %.2f' % 
                      (it, loss_value, lambda_1_value, lambda_2_value, elapsed))  # Print progress
                start_time = time.time()  # Reset timer for next interval
    
        # Stage 2: L-BFGS-B optimization for fine-tuning
        self.optimizer.minimize(self.sess,  # Run L-BFGS-B optimization
                                feed_dict = tf_dict,  # Provide training data
                                fetches = [self.loss],  # Monitor loss during optimization
                                loss_callback = self.callback)  # Use callback for progress
    
    def predict(self, x_star):
        """
        Make predictions using the trained neural network.
        
        This method uses the trained model to predict solutions at both time points
        for given spatial coordinates. The predictions include all IRK stages.
        
        Args:
            x_star: Spatial coordinates for prediction (N x 1)
            
        Returns:
            U0_star: Predicted solution at t₀ (N x q)
            U1_star: Predicted solution at t₁ (N x q)
        """
        
        # Predict solution at initial time t₀
        U0_star = self.sess.run(self.U0_pred, {self.x0_tf: x_star,  # Input spatial coordinates
                                               self.dummy_x0_tf: np.ones((x_star.shape[0], self.q))})  # Dummy variables
        
        # Predict solution at final time t₁        
        U1_star = self.sess.run(self.U1_pred, {self.x1_tf: x_star,  # Input spatial coordinates
                                               self.dummy_x1_tf: np.ones((x_star.shape[0], self.q))})  # Dummy variables
                    
        return U0_star, U1_star  # Return both predictions

    
if __name__ == "__main__": 
    """
    Main execution block for Korteweg-de Vries equation parameter identification.
    
    This script demonstrates PINN-based parameter identification for the KdV equation
    using discrete time stepping with Implicit Runge-Kutta methods. The true parameters are:
    - lambda_1 = 1.0 (nonlinear coefficient)
    - lambda_2 = 0.0025 (dispersive coefficient)
    """
        
    q = 50  # Number of IRK stages (higher q = higher accuracy but more computation)
    skip = 120  # Number of time steps to skip between t₀ and t₁ (larger skip = larger dt)

    # Training data configuration
    N0 = 199  # Number of training points at initial time t₀
    N1 = 201  # Number of training points at final time t₁
    layers = [1, 50, 50, 50, 50, q]  # Neural network architecture: [input, hidden..., output]
    
    # Load synthetic KdV solution data
    data = scipy.io.loadmat('../Data/KdV.mat')  # Load .mat file containing exact solution
    
    # Extract time and space coordinates from loaded data
    t_star = data['tt'].flatten()[:,None]  # Time vector (T x 1)
    x_star = data['x'].flatten()[:,None]    # Space vector (N x 1)
    Exact = np.real(data['uu'])            # Exact solution (N x T) - take real part
    
    idx_t = 40  # Index for initial time point (t₀ = t_star[idx_t])
        
    ######################################################################
    ######################## Noiseless Data ###############################
    ######################################################################
    noise = 0.0    # Noise level for clean data (no noise added)
    
    # Sample training data at initial time t₀
    idx_x = np.random.choice(Exact.shape[0], N0, replace=False)  # Random spatial indices
    x0 = x_star[idx_x,:]  # Selected spatial coordinates at t₀
    u0 = Exact[idx_x,idx_t][:,None]  # Solution values at t₀ (add noise if specified)
    u0 = u0 + noise*np.std(u0)*np.random.randn(u0.shape[0], u0.shape[1])  # Add noise (0 for clean data)
        
    # Sample training data at final time t₁
    idx_x = np.random.choice(Exact.shape[0], N1, replace=False)  # Random spatial indices
    x1 = x_star[idx_x,:]  # Selected spatial coordinates at t₁
    u1 = Exact[idx_x,idx_t + skip][:,None]  # Solution values at t₁ (skip time steps ahead)
    u1 = u1 + noise*np.std(u1)*np.random.randn(u1.shape[0], u1.shape[1])  # Add noise (0 for clean data)
    
    # Calculate time step size for discrete time stepping
    dt = np.asscalar(t_star[idx_t+skip] - t_star[idx_t])  # Time difference between t₀ and t₁
        
    # Define domain bounds for input normalization
    lb = x_star.min(0)  # Lower bound for spatial domain
    ub = x_star.max(0)  # Upper bound for spatial domain

    # Initialize and train PINN model on clean data
    model = PhysicsInformedNN(x0, u0, x1, u1, layers, dt, lb, ub, q)  # Create model instance
    model.train(nIter = 50000)  # Train for 50,000 Adam iterations + L-BFGS-B fine-tuning
    
    # Make predictions on full spatial domain
    U0_pred, U1_pred = model.predict(x_star)  # Predict solutions at both time points
        
    # Get identified parameters from trained model
    lambda_1_value = model.sess.run(model.lambda_1)  # Get learned λ₁
    lambda_2_value = np.exp(model.sess.run(model.lambda_2))  # Get learned λ₂ (exp of log)
                
    # Calculate parameter identification errors (relative percentage)
    error_lambda_1 = np.abs(lambda_1_value - 1.0)/1.0 *100  # Error in λ₁ (true value = 1.0)
    error_lambda_2 = np.abs(lambda_2_value - 0.0025)/0.0025 * 100  # Error in λ₂ (true value = 0.0025)
    
    # Print results for clean data
    print('Error lambda_1: %f%%' % (error_lambda_1))  # Print λ₁ identification error
    print('Error lambda_2: %f%%' % (error_lambda_2))  # Print λ₂ identification error
    
    
    ######################################################################
    ########################### Noisy Data ###############################
    ######################################################################
    noise = 0.01  # Add 1% noise to test robustness of parameter identification
        
    # Add Gaussian noise to existing training data to test robustness
    u0 = u0 + noise*np.std(u0)*np.random.randn(u0.shape[0], u0.shape[1])  # Add noise to t₀ data
    u1 = u1 + noise*np.std(u1)*np.random.randn(u1.shape[0], u1.shape[1])  # Add noise to t₁ data
    
    # Initialize and train new PINN model on noisy data
    model = PhysicsInformedNN(x0, u0, x1, u1, layers, dt, lb, ub, q)  # Create new model instance
    model.train(nIter = 50000)  # Train for 50,000 Adam iterations + L-BFGS-B fine-tuning
    
    # Make predictions on full spatial domain (unused variable)
    U_pred = model.predict(x_star)  # This line appears to be redundant
    
    # Make predictions on full spatial domain
    U0_pred, U1_pred = model.predict(x_star)  # Predict solutions at both time points
        
    # Get identified parameters from trained model on noisy data
    lambda_1_value_noisy = model.sess.run(model.lambda_1)  # Get learned λ₁ from noisy data
    lambda_2_value_noisy = np.exp(model.sess.run(model.lambda_2))  # Get learned λ₂ from noisy data
                
    # Calculate parameter identification errors for noisy data (relative percentage)
    error_lambda_1_noisy = np.abs(lambda_1_value_noisy - 1.0)/1.0 *100  # Error in λ₁
    error_lambda_2_noisy = np.abs(lambda_2_value_noisy - 0.0025)/0.0025 * 100  # Error in λ₂
    
    # Print results for noisy data
    print('Error lambda_1: %f%%' % (error_lambda_1_noisy))  # Print λ₁ identification error
    print('Error lambda_2: %f%%' % (error_lambda_2_noisy))  # Print λ₂ identification error
    
    ######################################################################
    ############################# Plotting ###############################
    ######################################################################
    
    # Create main figure for visualization
    fig, ax = newfig(1.0, 1.5)  # Create figure with custom size (width=1.0, height=1.5)
    ax.axis('off')  # Hide axes for clean appearance
    
    # Set up grid layout for solution field plot
    gs0 = gridspec.GridSpec(1, 2)  # Create 1x2 grid for main plot
    gs0.update(top=1-0.06, bottom=1-1/3+0.05, left=0.15, right=0.85, wspace=0)  # Position grid
    ax = plt.subplot(gs0[:, :])  # Create subplot spanning full grid
        
    # Plot exact KdV solution as heatmap
    h = ax.imshow(Exact, interpolation='nearest', cmap='rainbow',  # Use rainbow colormap
                  extent=[t_star.min(),t_star.max(), lb[0], ub[0]],  # Set axis limits
                  origin='lower', aspect='auto')  # Origin at bottom, auto aspect ratio
    
    # Add colorbar to the right of the plot
    divider = make_axes_locatable(ax)  # Create divider for colorbar positioning
    cax = divider.append_axes("right", size="5%", pad=0.05)  # Add colorbar axis
    fig.colorbar(h, cax=cax)  # Add colorbar to the figure
    
    # Add vertical lines to mark training time points
    line = np.linspace(x_star.min(), x_star.max(), 2)[:,None]  # Create vertical line coordinates
    ax.plot(t_star[idx_t]*np.ones((2,1)), line, 'w-', linewidth = 1.0)  # White line at t₀
    ax.plot(t_star[idx_t + skip]*np.ones((2,1)), line, 'w-', linewidth = 1.0)  # White line at t₁
    
    # Set plot labels and title
    ax.set_xlabel('$t$')  # X-axis label (time)
    ax.set_ylabel('$x$')  # Y-axis label (space)
    ax.set_title('$u(t,x)$', fontsize = 10)  # Plot title
    
    # Set up grid layout for training data visualization
    gs1 = gridspec.GridSpec(1, 2)  # Create 1x2 grid for training data plots
    gs1.update(top=1-1/3-0.1, bottom=1-2/3, left=0.15, right=0.85, wspace=0.5)  # Position grid

    # Plot 1: Training data at initial time t₀
    ax = plt.subplot(gs1[0, 0])  # Left subplot
    ax.plot(x_star,Exact[:,idx_t][:,None], 'b', linewidth = 2, label = 'Exact')  # Plot exact solution
    ax.plot(x0, u0, 'rx', linewidth = 2, label = 'Data')  # Plot training data points
    ax.set_xlabel('$x$')  # X-axis label
    ax.set_ylabel('$u(t,x)$')  # Y-axis label
    ax.set_title('$t = %.2f$\n%d trainng data' % (t_star[idx_t], u0.shape[0]), fontsize = 10)  # Title with info
    
    # Plot 2: Training data at final time t₁
    ax = plt.subplot(gs1[0, 1])  # Right subplot
    ax.plot(x_star,Exact[:,idx_t + skip][:,None], 'b', linewidth = 2, label = 'Exact')  # Plot exact solution
    ax.plot(x1, u1, 'rx', linewidth = 2, label = 'Data')  # Plot training data points
    ax.set_xlabel('$x$')  # X-axis label
    ax.set_ylabel('$u(t,x)$')  # Y-axis label
    ax.set_title('$t = %.2f$\n%d trainng data' % (t_star[idx_t+skip], u1.shape[0]), fontsize = 10)  # Title with info
    ax.legend(loc='upper center', bbox_to_anchor=(-0.3, -0.3), ncol=2, frameon=False)  # Add legend
    
    # Set up grid layout for results table
    gs2 = gridspec.GridSpec(1, 2)  # Create 1x2 grid for results table
    gs2.update(top=1-2/3-0.05, bottom=0, left=0.15, right=0.85, wspace=0.0)  # Position grid
    
    # Create results comparison table
    ax = plt.subplot(gs2[0, 0])  # Create subplot for table
    ax.axis('off')  # Hide axes for table
    
    # Build LaTeX table showing identified PDE parameters
    s1 = r'$\begin{tabular}{ |c|c| }  \hline Correct PDE & $u_t + u u_x + 0.0025 u_{xxx} = 0$ \\  \hline Identified PDE (clean data) & '  # Table header
    s2 = r'$u_t + %.3f u u_x + %.7f u_{xxx} = 0$ \\  \hline ' % (lambda_1_value, lambda_2_value)  # Clean data results
    s3 = r'Identified PDE (1\% noise) & '  # Noisy data header
    s4 = r'$u_t + %.3f u u_x + %.7f u_{xxx} = 0$  \\  \hline ' % (lambda_1_value_noisy, lambda_2_value_noisy)  # Noisy data results
    s5 = r'\end{tabular}$'  # Table footer
    s = s1+s2+s3+s4+s5  # Combine all table parts
    ax.text(-0.1,0.2,s)  # Display table at specified position

    # Save the figure (commented out)
    # savefig('./figures/KdV')  # Uncomment to save figure 