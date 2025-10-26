"""
Physics-Informed Neural Networks for Nonlinear Schrödinger Equation
@author: Maziar Raissi

This script implements a PINN to solve the Nonlinear Schrödinger (NLS) equation:
iψ_t + 0.5ψ_xx + |ψ|²ψ = 0

The network learns to predict the real and imaginary parts of the complex wave function ψ(x,t)
while satisfying the NLS equation constraints. The equation is decomposed into real and imaginary
parts: ψ = u + iv, where u and v are the real and imaginary components respectively.
"""

# Add the Utilities directory to Python path for importing custom plotting functions
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../Utilities'))

# Core libraries for deep learning and scientific computing
import tensorflow as tf  # Deep learning framework
import numpy as np       # Numerical computing
import matplotlib.pyplot as plt  # Plotting library
import scipy.io          # For loading .mat files
from scipy.interpolate import griddata  # For interpolating scattered data to regular grids
from pyDOE import lhs   # Latin Hypercube Sampling for collocation points
from plotting import newfig, savefig  # Custom plotting functions
from mpl_toolkits.mplot3d import Axes3D  # 3D plotting
import time              # For timing operations
import matplotlib.gridspec as gridspec  # For subplot layout
from mpl_toolkits.axes_grid1 import make_axes_locatable  # For colorbar positioning

# Set random seeds for reproducibility
np.random.seed(1234)  # NumPy random seed
tf.random.set_seed(1234)  # TensorFlow random seed


class PhysicsInformedNN:
    """
    Physics-Informed Neural Network for Nonlinear Schrödinger equation.
    
    This class implements a neural network that learns to solve the NLS equation by predicting
    the real (u) and imaginary (v) parts of the complex wave function ψ = u + iv while satisfying
    the physics constraints encoded in the NLS equation residuals.
    """
    
    def __init__(self, x0, u0, v0, tb, X_f, layers, lb, ub):
        """
        Initialize the Physics-Informed Neural Network.
        
        Args:
            x0: Initial spatial coordinates (N0 x 1)
            u0: Initial real part of wave function (N0 x 1)
            v0: Initial imaginary part of wave function (N0 x 1)
            tb: Boundary time coordinates (Nb x 1)
            X_f: Collocation points for physics loss (Nf x 2) [x, t]
            layers: List defining neural network architecture [input_dim, hidden_dims..., output_dim]
            lb: Lower bounds [x_min, t_min]
            ub: Upper bounds [x_max, t_max]
        """
        
        # Prepare initial condition data (t=0)
        X0 = np.concatenate((x0, 0*x0), 1) # (x0, 0) - spatial coordinates at t=0
        
        # Prepare boundary condition data at x=lb[0] and x=ub[0]
        X_lb = np.concatenate((0*tb + lb[0], tb), 1) # (lb[0], tb) - left boundary
        X_ub = np.concatenate((0*tb + ub[0], tb), 1) # (ub[0], tb) - right boundary
        
        # Store domain bounds for normalization
        self.lb = lb  # Lower bounds [x_min, t_min]
        self.ub = ub  # Upper bounds [x_max, t_max]
               
        # Extract coordinate arrays for initial conditions
        self.x0 = X0[:,0:1]  # Initial x-coordinates
        self.t0 = X0[:,1:2]  # Initial t-coordinates (all zeros)

        # Extract coordinate arrays for left boundary conditions
        self.x_lb = X_lb[:,0:1]  # Left boundary x-coordinates (all lb[0])
        self.t_lb = X_lb[:,1:2]  # Left boundary t-coordinates

        # Extract coordinate arrays for right boundary conditions
        self.x_ub = X_ub[:,0:1]  # Right boundary x-coordinates (all ub[0])
        self.t_ub = X_ub[:,1:2]  # Right boundary t-coordinates
        
        # Extract collocation point coordinates
        self.x_f = X_f[:,0:1]  # Collocation x-coordinates
        self.t_f = X_f[:,1:2]  # Collocation t-coordinates
        
        # Store initial condition data
        self.u0 = u0  # Initial real part of wave function
        self.v0 = v0  # Initial imaginary part of wave function
        
        # Initialize neural network
        self.layers = layers  # Network architecture
        self.weights, self.biases = self.initialize_NN(layers)
        
        # Define TensorFlow placeholders for all input data
        # Initial condition placeholders
        self.x0_tf = tf.placeholder(tf.float32, shape=[None, self.x0.shape[1]])
        self.t0_tf = tf.placeholder(tf.float32, shape=[None, self.t0.shape[1]])
        self.u0_tf = tf.placeholder(tf.float32, shape=[None, self.u0.shape[1]])
        self.v0_tf = tf.placeholder(tf.float32, shape=[None, self.v0.shape[1]])
        
        # Left boundary condition placeholders
        self.x_lb_tf = tf.placeholder(tf.float32, shape=[None, self.x_lb.shape[1]])
        self.t_lb_tf = tf.placeholder(tf.float32, shape=[None, self.t_lb.shape[1]])
        
        # Right boundary condition placeholders
        self.x_ub_tf = tf.placeholder(tf.float32, shape=[None, self.x_ub.shape[1]])
        self.t_ub_tf = tf.placeholder(tf.float32, shape=[None, self.t_ub.shape[1]])
        
        # Collocation point placeholders
        self.x_f_tf = tf.placeholder(tf.float32, shape=[None, self.x_f.shape[1]])
        self.t_f_tf = tf.placeholder(tf.float32, shape=[None, self.t_f.shape[1]])

        # Build neural network graphs for different data types
        # Initial condition predictions
        self.u0_pred, self.v0_pred, _ , _ = self.net_uv(self.x0_tf, self.t0_tf)
        
        # Left boundary predictions (including spatial derivatives for periodic BCs)
        self.u_lb_pred, self.v_lb_pred, self.u_x_lb_pred, self.v_x_lb_pred = self.net_uv(self.x_lb_tf, self.t_lb_tf)
        
        # Right boundary predictions (including spatial derivatives for periodic BCs)
        self.u_ub_pred, self.v_ub_pred, self.u_x_ub_pred, self.v_x_ub_pred = self.net_uv(self.x_ub_tf, self.t_ub_tf)
        
        # Physics residuals at collocation points
        self.f_u_pred, self.f_v_pred = self.net_f_uv(self.x_f_tf, self.t_f_tf)
        
        # Define total loss function combining all constraints
        # Initial condition loss: MSE between predicted and observed initial values
        # Periodic boundary conditions: u and v must be equal at boundaries
        # Periodic boundary conditions: derivatives must be equal at boundaries  
        # Physics loss: NLS equation residuals (should be zero)
        self.loss = tf.reduce_mean(tf.square(self.u0_tf - self.u0_pred)) + \
                    tf.reduce_mean(tf.square(self.v0_tf - self.v0_pred)) + \
                    tf.reduce_mean(tf.square(self.u_lb_pred - self.u_ub_pred)) + \
                    tf.reduce_mean(tf.square(self.v_lb_pred - self.v_ub_pred)) + \
                    tf.reduce_mean(tf.square(self.u_x_lb_pred - self.u_x_ub_pred)) + \
                    tf.reduce_mean(tf.square(self.v_x_lb_pred - self.v_x_ub_pred)) + \
                    tf.reduce_mean(tf.square(self.f_u_pred)) + \
                    tf.reduce_mean(tf.square(self.f_v_pred))
        
        # Set up L-BFGS-B optimizer for fine-tuning (quasi-Newton method)
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 50000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})
    
        # Set up Adam optimizer for initial training (gradient descent)
        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)
                
        # Create TensorFlow session with device placement logging
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        # Initialize all TensorFlow variables
        init = tf.global_variables_initializer()
        self.sess.run(init)
              
    def initialize_NN(self, layers):
        """
        Initialize neural network weights and biases.
        
        Args:
            layers: List defining network architecture [input_dim, hidden_dims..., output_dim]
            
        Returns:
            weights: List of weight matrices for each layer
            biases: List of bias vectors for each layer
        """
        weights = []  # Store weight matrices
        biases = []   # Store bias vectors
        num_layers = len(layers)  # Total number of layers
        
        # Initialize weights and biases for each layer
        for l in range(0,num_layers-1):
            # Initialize weights using Xavier initialization
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            # Initialize biases to zero
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases
        
    def xavier_init(self, size):
        """
        Xavier/Glorot initialization for neural network weights.
        
        This initialization method helps prevent vanishing/exploding gradients by
        scaling the initial weights based on the number of input and output neurons.
        
        Args:
            size: [input_dimension, output_dimension]
            
        Returns:
            tf.Variable: Randomly initialized weight matrix
        """
        in_dim = size[0]   # Input dimension
        out_dim = size[1]  # Output dimension
        
        # Calculate standard deviation for Xavier initialization
        # Formula: sqrt(2 / (fan_in + fan_out))
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        
        # Create weight matrix with truncated normal distribution
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    
    def neural_net(self, X, weights, biases):
        """
        Forward pass through the neural network.
        
        Args:
            X: Input tensor (batch_size, input_dim)
            weights: List of weight matrices
            biases: List of bias vectors
            
        Returns:
            Y: Output tensor (batch_size, output_dim)
        """
        num_layers = len(weights) + 1  # Total number of layers
        
        # Normalize input features to [-1, 1] range for better training stability
        # Formula: 2 * (X - min) / (max - min) - 1
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        
        # Forward pass through hidden layers with tanh activation
        for l in range(0,num_layers-2):
            W = weights[l]  # Weight matrix for layer l
            b = biases[l]   # Bias vector for layer l
            # Linear transformation: H * W + b, then apply tanh activation
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        
        # Output layer (no activation function)
        W = weights[-1]  # Final weight matrix
        b = biases[-1]   # Final bias vector
        Y = tf.add(tf.matmul(H, W), b)  # Linear transformation only
        return Y
    
    def net_uv(self, x, t):
        """
        Neural network that predicts real and imaginary parts of wave function.
        
        Args:
            x: x-coordinates tensor
            t: t-coordinates tensor
            
        Returns:
            u: Real part of wave function ψ
            v: Imaginary part of wave function ψ
            u_x: Spatial derivative of real part ∂u/∂x
            v_x: Spatial derivative of imaginary part ∂v/∂x
        """
        # Concatenate spatial and temporal coordinates
        X = tf.concat([x,t],1)
        
        # Neural network predicts both u and v components
        uv = self.neural_net(X, self.weights, self.biases)
        u = uv[:,0:1]  # Real part of wave function
        v = uv[:,1:2]  # Imaginary part of wave function
        
        # Compute spatial derivatives using automatic differentiation
        u_x = tf.gradients(u, x)[0]  # ∂u/∂x
        v_x = tf.gradients(v, x)[0]  # ∂v/∂x

        return u, v, u_x, v_x

    def net_f_uv(self, x, t):
        """
        Compute NLS equation residuals.
        
        The Nonlinear Schrödinger equation: iψ_t + 0.5ψ_xx + |ψ|²ψ = 0
        Decomposed into real and imaginary parts:
        - Real part: u_t + 0.5*v_xx + (u² + v²)*v = 0
        - Imaginary part: v_t - 0.5*u_xx - (u² + v²)*u = 0
        
        Args:
            x: x-coordinates tensor
            t: t-coordinates tensor
            
        Returns:
            f_u: Residual of real part equation (should be zero)
            f_v: Residual of imaginary part equation (should be zero)
        """
        # Get wave function components and their spatial derivatives
        u, v, u_x, v_x = self.net_uv(x,t)
        
        # Compute temporal derivatives
        u_t = tf.gradients(u, t)[0]  # ∂u/∂t
        v_t = tf.gradients(v, t)[0]  # ∂v/∂t
        
        # Compute second-order spatial derivatives
        u_xx = tf.gradients(u_x, x)[0]  # ∂²u/∂x²
        v_xx = tf.gradients(v_x, x)[0]  # ∂²v/∂x²
        
        # NLS equation residuals
        # Real part: u_t + 0.5*v_xx + (u² + v²)*v = 0
        f_u = u_t + 0.5*v_xx + (u**2 + v**2)*v
        # Imaginary part: v_t - 0.5*u_xx - (u² + v²)*u = 0
        f_v = v_t - 0.5*u_xx - (u**2 + v**2)*u   
        
        return f_u, f_v
    
    def callback(self, loss):
        """
        Callback function for L-BFGS-B optimizer to print training progress.
        
        Args:
            loss: Current loss value
        """
        print('Loss:', loss)
        
    def train(self, nIter):
        """
        Train the Physics-Informed Neural Network.
        
        Uses a two-stage training approach:
        1. Adam optimizer for initial training (fast convergence)
        2. L-BFGS-B optimizer for fine-tuning (better final accuracy)
        
        Args:
            nIter: Number of Adam iterations before switching to L-BFGS-B
        """
        
        # Prepare training data dictionary for TensorFlow
        tf_dict = {self.x0_tf: self.x0, self.t0_tf: self.t0,
                   self.u0_tf: self.u0, self.v0_tf: self.v0,
                   self.x_lb_tf: self.x_lb, self.t_lb_tf: self.t_lb,
                   self.x_ub_tf: self.x_ub, self.t_ub_tf: self.t_ub,
                   self.x_f_tf: self.x_f, self.t_f_tf: self.t_f}
        
        # Stage 1: Adam optimization
        start_time = time.time()
        for it in range(nIter):
            # Run one Adam optimization step
            self.sess.run(self.train_op_Adam, tf_dict)
            
            # Print progress every 10 iterations
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2f' % 
                      (it, loss_value, elapsed))
                start_time = time.time()
                                                                                                                          
        # Stage 2: L-BFGS-B optimization for fine-tuning
        self.optimizer.minimize(self.sess, 
                                feed_dict = tf_dict,         
                                fetches = [self.loss], 
                                loss_callback = self.callback)        
                                    
    
    def predict(self, X_star):
        """
        Make predictions using the trained neural network.
        
        Args:
            X_star: Input coordinates (N x 2) [x, t]
            
        Returns:
            u_star: Predicted real part of wave function
            v_star: Predicted imaginary part of wave function
            f_u_star: Predicted NLS residual for real part
            f_v_star: Predicted NLS residual for imaginary part
        """
        
        # Predict wave function components
        tf_dict = {self.x0_tf: X_star[:,0:1], self.t0_tf: X_star[:,1:2]}
        
        u_star = self.sess.run(self.u0_pred, tf_dict)  
        v_star = self.sess.run(self.v0_pred, tf_dict)  
        
        # Predict physics residuals
        tf_dict = {self.x_f_tf: X_star[:,0:1], self.t_f_tf: X_star[:,1:2]}
        
        f_u_star = self.sess.run(self.f_u_pred, tf_dict)
        f_v_star = self.sess.run(self.f_v_pred, tf_dict)
               
        return u_star, v_star, f_u_star, f_v_star
    
if __name__ == "__main__": 
    """
    Main execution block for Nonlinear Schrödinger equation solution.
    
    This script demonstrates PINN-based solution of the NLS equation using synthetic data.
    The NLS equation models wave propagation in nonlinear media and exhibits soliton solutions.
    """
     
    # Noise level for robustness testing (set to 0 for clean data)
    noise = 0.0        
    
    # Domain bounds: spatial domain [-5, 5] and temporal domain [0, π/2]
    lb = np.array([-5.0, 0.0])    # Lower bounds [x_min, t_min]
    ub = np.array([5.0, np.pi/2]) # Upper bounds [x_max, t_max]

    # Training data configuration
    N0 = 50      # Number of initial condition points
    N_b = 50     # Number of boundary condition points
    N_f = 20000  # Number of collocation points for physics loss
    
    # Neural network architecture: [input_dim, hidden_layers..., output_dim]
    # Input: (x, t) -> Output: (u, v) where ψ = u + iv
    layers = [2, 100, 100, 100, 100, 2]
        
    # Load synthetic NLS solution data
    data = scipy.io.loadmat('../Data/NLS.mat')
    
    # Extract time and space coordinates
    t = data['tt'].flatten()[:,None]  # Time vector
    x = data['x'].flatten()[:,None]   # Space vector
    Exact = data['uu']                # Complex exact solution
    
    # Decompose complex solution into real and imaginary parts
    Exact_u = np.real(Exact)  # Real part of exact solution
    Exact_v = np.imag(Exact)  # Imaginary part of exact solution
    Exact_h = np.sqrt(Exact_u**2 + Exact_v**2)  # Magnitude |ψ|
    
    # Create coordinate mesh for visualization
    X, T = np.meshgrid(x,t)
    
    # Flatten coordinates and solution for neural network training
    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))  # [x, t] coordinates
    u_star = Exact_u.T.flatten()[:,None]  # Real part flattened
    v_star = Exact_v.T.flatten()[:,None]  # Imaginary part flattened
    h_star = Exact_h.T.flatten()[:,None]  # Magnitude flattened
    
    ########################### Training Data Preparation ###########################
    
    # Sample initial condition data (t=0)
    idx_x = np.random.choice(x.shape[0], N0, replace=False)  # Random spatial points
    x0 = x[idx_x,:]           # Initial x-coordinates
    u0 = Exact_u[idx_x,0:1]  # Initial real part (t=0)
    v0 = Exact_v[idx_x,0:1]  # Initial imaginary part (t=0)
    
    # Sample boundary condition data
    idx_t = np.random.choice(t.shape[0], N_b, replace=False)  # Random time points
    tb = t[idx_t,:]  # Boundary time coordinates
    
    # Generate collocation points using Latin Hypercube Sampling
    # These points are used to enforce the NLS equation physics
    X_f = lb + (ub-lb)*lhs(2, N_f)  # Random points in [lb, ub] domain
            
    # Initialize and train the PINN model
    model = PhysicsInformedNN(x0, u0, v0, tb, X_f, layers, lb, ub)
             
    # Train the model
    start_time = time.time()                
    model.train(50000)  # Train for 50,000 Adam iterations + L-BFGS-B fine-tuning
    elapsed = time.time() - start_time                
    print('Training time: %.4f' % (elapsed))
    
    # Make predictions on the full domain
    u_pred, v_pred, f_u_pred, f_v_pred = model.predict(X_star)
    
    # Compute magnitude of predicted solution
    h_pred = np.sqrt(u_pred**2 + v_pred**2)
            
    # Calculate prediction errors (relative L2 norm)
    error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
    error_v = np.linalg.norm(v_star-v_pred,2)/np.linalg.norm(v_star,2)
    error_h = np.linalg.norm(h_star-h_pred,2)/np.linalg.norm(h_star,2)
    
    # Print results
    print('Error u: %e' % (error_u))
    print('Error v: %e' % (error_v))
    print('Error h: %e' % (error_h))

    
    # Interpolate predictions to regular grid for visualization
    U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')  # Real part
    V_pred = griddata(X_star, v_pred.flatten(), (X, T), method='cubic')  # Imaginary part
    H_pred = griddata(X_star, h_pred.flatten(), (X, T), method='cubic')  # Magnitude

    # Interpolate physics residuals for analysis
    FU_pred = griddata(X_star, f_u_pred.flatten(), (X, T), method='cubic')  # Real part residual
    FV_pred = griddata(X_star, f_v_pred.flatten(), (X, T), method='cubic')  # Imaginary part residual     
    

    
    ######################################################################
    ############################# Plotting ###############################
    ######################################################################    
    
    # Prepare training data coordinates for visualization
    X0 = np.concatenate((x0, 0*x0), 1)  # Initial condition points (x0, 0)
    X_lb = np.concatenate((0*tb + lb[0], tb), 1)  # Left boundary points (lb[0], tb)
    X_ub = np.concatenate((0*tb + ub[0], tb), 1)  # Right boundary points (ub[0], tb)
    X_u_train = np.vstack([X0, X_lb, X_ub])  # All training data points

    # Create main figure
    fig, ax = newfig(1.0, 0.9)  # Custom figure size
    ax.axis('off')  # Hide axes
    
    ####### Row 0: |h(t,x)| - Magnitude of Wave Function ##################    
    gs0 = gridspec.GridSpec(1, 2)  # Grid layout
    gs0.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
    ax = plt.subplot(gs0[:, :])
    
    # Plot magnitude of predicted wave function
    h = ax.imshow(H_pred.T, interpolation='nearest', cmap='YlGnBu', 
                  extent=[lb[1], ub[1], lb[0], ub[0]], 
                  origin='lower', aspect='auto')
    
    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    
    # Plot training data points
    ax.plot(X_u_train[:,1], X_u_train[:,0], 'kx', label = 'Data (%d points)' % (X_u_train.shape[0]), 
            markersize = 4, clip_on = False)
    
    # Add vertical lines at specific time slices for cross-sections
    line = np.linspace(x.min(), x.max(), 2)[:,None]
    ax.plot(t[75]*np.ones((2,1)), line, 'k--', linewidth = 1)   # t = t[75]
    ax.plot(t[100]*np.ones((2,1)), line, 'k--', linewidth = 1)  # t = t[100]
    ax.plot(t[125]*np.ones((2,1)), line, 'k--', linewidth = 1)  # t = t[125]
    
    # Set plot properties
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    leg = ax.legend(frameon=False, loc = 'best')
    ax.set_title('$|h(t,x)|$', fontsize = 10)
    
    ####### Row 1: Cross-Sectional Plots at Different Times ##################    
    gs1 = gridspec.GridSpec(1, 3)  # Grid for three time slices
    gs1.update(top=1-1/3, bottom=0, left=0.1, right=0.9, wspace=0.5)
    
    # Plot 1: Cross-section at t[75]
    ax = plt.subplot(gs1[0, 0])
    ax.plot(x,Exact_h[:,75], 'b-', linewidth = 2, label = 'Exact')       # Exact solution
    ax.plot(x,H_pred[75,:], 'r--', linewidth = 2, label = 'Prediction')  # PINN prediction
    ax.set_xlabel('$x$')
    ax.set_ylabel('$|h(t,x)|$')    
    ax.set_title('$t = %.2f$' % (t[75]), fontsize = 10)
    ax.axis('square')
    ax.set_xlim([-5.1,5.1])
    ax.set_ylim([-0.1,5.1])
    
    # Plot 2: Cross-section at t[100]
    ax = plt.subplot(gs1[0, 1])
    ax.plot(x,Exact_h[:,100], 'b-', linewidth = 2, label = 'Exact')       # Exact solution
    ax.plot(x,H_pred[100,:], 'r--', linewidth = 2, label = 'Prediction')  # PINN prediction
    ax.set_xlabel('$x$')
    ax.set_ylabel('$|h(t,x)|$')
    ax.axis('square')
    ax.set_xlim([-5.1,5.1])
    ax.set_ylim([-0.1,5.1])
    ax.set_title('$t = %.2f$' % (t[100]), fontsize = 10)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.8), ncol=5, frameon=False)
    
    # Plot 3: Cross-section at t[125]
    ax = plt.subplot(gs1[0, 2])
    ax.plot(x,Exact_h[:,125], 'b-', linewidth = 2, label = 'Exact')       # Exact solution
    ax.plot(x,H_pred[125,:], 'r--', linewidth = 2, label = 'Prediction')  # PINN prediction
    ax.set_xlabel('$x$')
    ax.set_ylabel('$|h(t,x)|$')
    ax.axis('square')
    ax.set_xlim([-5.1,5.1])
    ax.set_ylim([-0.1,5.1])    
    ax.set_title('$t = %.2f$' % (t[125]), fontsize = 10)
    
    # Save the figure
    # savefig('./figures/NLS')  
    
