"""
Physics-Informed Neural Networks for Navier-Stokes Equation Identification
@author: Maziar Raissi

This script implements a PINN to identify the parameters of the Navier-Stokes equations
from velocity field data. The network learns to predict velocity components (u,v) and 
pressure (p) while simultaneously discovering the unknown parameters lambda_1 and lambda_2
that govern the convective and viscous terms in the Navier-Stokes equations.
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
import time              # For timing operations
from itertools import product, combinations  # For generating coordinate combinations
from mpl_toolkits.mplot3d import Axes3D  # 3D plotting
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # 3D plotting utilities
from plotting import newfig, savefig  # Custom plotting functions
from mpl_toolkits.axes_grid1 import make_axes_locatable  # For colorbar positioning
import matplotlib.gridspec as gridspec  # For subplot layout

# Set random seeds for reproducibility
np.random.seed(1234)  # NumPy random seed
tf.random.set_seed(1234)  # TensorFlow random seed

class PhysicsInformedNN:
    """
    Physics-Informed Neural Network for Navier-Stokes equation identification.
    
    This class implements a neural network that learns to solve the Navier-Stokes equations
    while simultaneously identifying unknown parameters (lambda_1, lambda_2) from data.
    The network predicts velocity components (u,v) and pressure (p) using the stream function
    formulation of the Navier-Stokes equations.
    
    TensorFlow 2.x implementation with eager execution.
    """
    
    def __init__(self, x, y, t, u, v, layers):
        """
        Initialize the Physics-Informed Neural Network.
        
        Args:
            x: Spatial x-coordinates (N x 1)
            y: Spatial y-coordinates (N x 1) 
            t: Temporal coordinates (N x 1)
            u: x-component of velocity field (N x 1)
            v: y-component of velocity field (N x 1)
            layers: List defining neural network architecture [input_dim, hidden_dims..., output_dim]
        """
        
        # Concatenate spatial and temporal coordinates into a single input matrix
        X = np.concatenate([x, y, t], 1)  # Shape: (N, 3)
        
        # Compute normalization bounds for input features (min-max scaling)
        self.lb = X.min(0)  # Lower bounds for each coordinate
        self.ub = X.max(0)  # Upper bounds for each coordinate
                
        # Store the combined input matrix
        self.X = X
        
        # Extract individual coordinate arrays for easier access
        self.x = X[:,0:1]  # x-coordinates
        self.y = X[:,1:2]  # y-coordinates  
        self.t = X[:,2:3]  # t-coordinates
        
        # Store velocity field data
        self.u = u  # x-component of velocity
        self.v = v  # y-component of velocity
        
        # Store network architecture
        self.layers = layers
        
        # Initialize neural network weights and biases
        self.weights, self.biases = self.initialize_NN(layers)        
        
        # Initialize unknown parameters to be learned
        # lambda_1: coefficient for convective terms (should be 1.0)
        # lambda_2: coefficient for viscous terms (should be 0.01)
        self.lambda_1 = tf.Variable([0.0], dtype=tf.float32)
        self.lambda_2 = tf.Variable([0.0], dtype=tf.float32)
        
        # Set up Adam optimizer for initial training (gradient descent)
        self.optimizer_Adam = tf.optimizers.Adam()
        
        # Store training data as tensors
        self.x_train = tf.constant(self.x, dtype=tf.float32)
        self.y_train = tf.constant(self.y, dtype=tf.float32)
        self.t_train = tf.constant(self.t, dtype=tf.float32)
        self.u_train = tf.constant(self.u, dtype=tf.float32)
        self.v_train = tf.constant(self.v, dtype=tf.float32)

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
        return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    
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
        
    @tf.function
    def net_NS(self, x, y, t):
        """
        Neural network implementation of Navier-Stokes equations using stream function formulation.
        
        The network predicts stream function (psi) and pressure (p), then computes:
        - Velocity components: u = ∂ψ/∂y, v = -∂ψ/∂x
        - Navier-Stokes residuals: f_u, f_v (should be zero)
        
        Args:
            x: x-coordinates tensor
            y: y-coordinates tensor  
            t: time tensor
            
        Returns:
            u: x-component of velocity
            v: y-component of velocity
            p: pressure
            f_u: Navier-Stokes residual for x-momentum equation
            f_v: Navier-Stokes residual for y-momentum equation
        """
        # Get learnable parameters
        lambda_1 = self.lambda_1  # Convective coefficient (should be 1.0)
        lambda_2 = self.lambda_2  # Viscous coefficient (should be 0.01)
        
        # Use GradientTape to compute all derivatives
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([x, y, t])
            
            # Neural network predicts stream function (psi) and pressure (p)
            psi_and_p = self.neural_net(tf.concat([x,y,t], 1), self.weights, self.biases)
            psi = psi_and_p[:,0:1]  # Stream function
            p = psi_and_p[:,1:2]    # Pressure
            
            # Velocity components from stream function
            u = tape.gradient(psi, y)  # u = ∂ψ/∂y
            v = -tape.gradient(psi, x)  # v = -∂ψ/∂x
        
        # Compute derivatives of u
        with tf.GradientTape(persistent=True) as tape_u:
            tape_u.watch([x, y, t])
            psi_and_p = self.neural_net(tf.concat([x,y,t], 1), self.weights, self.biases)
            psi = psi_and_p[:,0:1]
            u = tape_u.gradient(psi, y)
            
        u_t = tape_u.gradient(u, t)  # ∂u/∂t
        u_x = tape_u.gradient(u, x)  # ∂u/∂x
        u_y = tape_u.gradient(u, y)  # ∂u/∂y
        
        # Compute second-order derivatives of u
        with tf.GradientTape(persistent=True) as tape_u2:
            tape_u2.watch([x, y])
            psi_and_p = self.neural_net(tf.concat([x,y,t], 1), self.weights, self.biases)
            psi = psi_and_p[:,0:1]
            u = tape_u2.gradient(psi, y)
            u_x = tape_u2.gradient(u, x)
            
        u_xx = tape_u2.gradient(u_x, x)  # ∂²u/∂x²
        
        with tf.GradientTape(persistent=True) as tape_u2y:
            tape_u2y.watch([x, y])
            psi_and_p = self.neural_net(tf.concat([x,y,t], 1), self.weights, self.biases)
            psi = psi_and_p[:,0:1]
            u = tape_u2y.gradient(psi, y)
            u_y = tape_u2y.gradient(u, y)
            
        u_yy = tape_u2y.gradient(u_y, y)  # ∂²u/∂y²
        
        # Compute derivatives of v
        with tf.GradientTape(persistent=True) as tape_v:
            tape_v.watch([x, y, t])
            psi_and_p = self.neural_net(tf.concat([x,y,t], 1), self.weights, self.biases)
            psi = psi_and_p[:,0:1]
            v = -tape_v.gradient(psi, x)
            
        v_t = tape_v.gradient(v, t)  # ∂v/∂t
        v_x = tape_v.gradient(v, x)  # ∂v/∂x
        v_y = tape_v.gradient(v, y)  # ∂v/∂y
        
        # Compute second-order derivatives of v
        with tf.GradientTape(persistent=True) as tape_v2:
            tape_v2.watch([x, y])
            psi_and_p = self.neural_net(tf.concat([x,y,t], 1), self.weights, self.biases)
            psi = psi_and_p[:,0:1]
            v = -tape_v2.gradient(psi, x)
            v_x = tape_v2.gradient(v, x)
            
        v_xx = tape_v2.gradient(v_x, x)  # ∂²v/∂x²
        
        with tf.GradientTape(persistent=True) as tape_v2y:
            tape_v2y.watch([x, y])
            psi_and_p = self.neural_net(tf.concat([x,y,t], 1), self.weights, self.biases)
            psi = psi_and_p[:,0:1]
            v = -tape_v2y.gradient(psi, x)
            v_y = tape_v2y.gradient(v, y)
            
        v_yy = tape_v2y.gradient(v_y, y)  # ∂²v/∂y²
        
        # Compute pressure gradients
        with tf.GradientTape(persistent=True) as tape_p:
            tape_p.watch([x, y])
            psi_and_p = self.neural_net(tf.concat([x,y,t], 1), self.weights, self.biases)
            p = psi_and_p[:,1:2]
            
        p_x = tape_p.gradient(p, x)  # ∂p/∂x
        p_y = tape_p.gradient(p, y)  # ∂p/∂y

        # Navier-Stokes equations residuals (should be zero)
        # x-momentum equation: ∂u/∂t + λ₁(u∂u/∂x + v∂u/∂y) + ∂p/∂x - λ₂(∂²u/∂x² + ∂²u/∂y²) = 0
        f_u = u_t + lambda_1*(u*u_x + v*u_y) + p_x - lambda_2*(u_xx + u_yy) 
        # y-momentum equation: ∂v/∂t + λ₁(u∂v/∂x + v∂v/∂y) + ∂p/∂y - λ₂(∂²v/∂x² + ∂²v/∂y²) = 0
        f_v = v_t + lambda_1*(u*v_x + v*v_y) + p_y - lambda_2*(v_xx + v_yy)
        
        return u, v, p, f_u, f_v
    
    def callback(self, loss, lambda_1, lambda_2):
        """
        Callback function for L-BFGS-B optimizer to print training progress.
        
        Args:
            loss: Current loss value
            lambda_1: Current value of convective parameter
            lambda_2: Current value of viscous parameter
        """
        print('Loss: %.3e, l1: %.3f, l2: %.5f' % (loss, lambda_1, lambda_2))
      
    @tf.function
    def loss_fn(self, x, y, t, u_true, v_true):
        """
        Compute the total loss function combining data fitting and physics constraints.
        
        Args:
            x, y, t: Input coordinates
            u_true, v_true: True velocity components
            
        Returns:
            Total loss value
        """
        u_pred, v_pred, p_pred, f_u_pred, f_v_pred = self.net_NS(x, y, t)
        
        # Data loss: MSE between predicted and observed velocities
        data_loss = tf.reduce_mean(tf.square(u_true - u_pred)) + tf.reduce_mean(tf.square(v_true - v_pred))
        
        # Physics loss: MSE of Navier-Stokes equation residuals (should be zero)
        physics_loss = tf.reduce_mean(tf.square(f_u_pred)) + tf.reduce_mean(tf.square(f_v_pred))
        
        return data_loss + physics_loss
    
    def train(self, nIter):
        """
        Train the Physics-Informed Neural Network using TensorFlow 2.x.
        
        Uses Adam optimizer for training.
        
        Args:
            nIter: Number of training iterations
        """
        # Flatten all trainable variables into a single list
        trainable_vars = []
        for w in self.weights:
            trainable_vars.append(w)
        for b in self.biases:
            trainable_vars.append(b)
        trainable_vars.extend([self.lambda_1, self.lambda_2])
        
        # Stage 1: Adam optimization
        start_time = time.time()
        for it in range(nIter):
            with tf.GradientTape() as tape:
                loss = self.loss_fn(self.x_train, self.y_train, self.t_train, 
                                   self.u_train, self.v_train)
            
            # Get gradients
            gradients = tape.gradient(loss, trainable_vars)
            
            # Apply gradients
            self.optimizer_Adam.apply_gradients(zip(gradients, trainable_vars))
            
            # Print progress every 10 iterations
            if it % 10 == 0:
                elapsed = time.time() - start_time
                lambda_1_value = self.lambda_1.numpy()[0]
                lambda_2_value = self.lambda_2.numpy()[0]
                print('It: %d, Loss: %.3e, l1: %.3f, l2: %.5f, Time: %.2f' % 
                      (it, loss.numpy(), lambda_1_value, lambda_2_value, elapsed))
                start_time = time.time()
    
    def predict(self, x_star, y_star, t_star):
        """
        Make predictions using the trained neural network.
        
        Args:
            x_star: x-coordinates for prediction (N x 1)
            y_star: y-coordinates for prediction (N x 1)
            t_star: time coordinates for prediction (N x 1)
            
        Returns:
            u_star: Predicted x-component of velocity
            v_star: Predicted y-component of velocity
            p_star: Predicted pressure
        """
        # Convert to tensors
        x_tf = tf.constant(x_star, dtype=tf.float32)
        y_tf = tf.constant(y_star, dtype=tf.float32)
        t_tf = tf.constant(t_star, dtype=tf.float32)
        
        # Run forward pass to get predictions
        u_star, v_star, p_star, _, _ = self.net_NS(x_tf, y_tf, t_tf)
        
        return u_star.numpy(), v_star.numpy(), p_star.numpy()

def plot_solution(X_star, u_star, index):
    """
    Plot a 2D solution field using pcolor.
    
    Args:
        X_star: Spatial coordinates (N x 2) [x, y]
        u_star: Field values to plot (N x 1)
        index: Figure number for the plot
    """
    
    # Get spatial bounds
    lb = X_star.min(0)  # Lower bounds [x_min, y_min]
    ub = X_star.max(0)  # Upper bounds [x_max, y_max]
    
    # Create regular grid for plotting
    nn = 200  # Number of grid points in each direction
    x = np.linspace(lb[0], ub[0], nn)  # x-coordinates
    y = np.linspace(lb[1], ub[1], nn)  # y-coordinates
    X, Y = np.meshgrid(x,y)  # Create 2D grid
    
    # Interpolate scattered data to regular grid
    U_star = griddata(X_star, u_star.flatten(), (X, Y), method='cubic')
    
    # Create and display plot
    plt.figure(index)
    plt.pcolor(X,Y,U_star, cmap = 'jet')  # Pseudo-color plot
    plt.colorbar()  # Add colorbar
    
    
def axisEqual3D(ax):
    """
    Make 3D plot axes equal in scale for better visualization.
    
    Args:
        ax: 3D matplotlib axis object
    """
    # Get current axis limits for x, y, z
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]  # Calculate size of each dimension
    centers = np.mean(extents, axis=1)  # Calculate center of each dimension
    maxsize = max(abs(sz))  # Find maximum dimension size
    r = maxsize/4  # Set radius for equal scaling
    
    # Set equal limits for all dimensions
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)
        
        
if __name__ == "__main__": 
    """
    Main execution block for Navier-Stokes equation identification.
    
    This script demonstrates PINN-based parameter identification for the Navier-Stokes equations
    using flow around a circular cylinder data. The true parameters are:
    - lambda_1 = 1.0 (convective coefficient)
    - lambda_2 = 0.01 (viscous coefficient)
    """
      
    # Training configuration
    N_train = 5000  # Number of training data points
    
    # Neural network architecture: [input_dim, hidden_layers..., output_dim]
    # Input: (x, y, t) -> Output: (psi, p) where psi is stream function
    layers = [3, 20, 20, 20, 20, 20, 20, 20, 20, 2]
    
    # Load experimental/simulation data
    data = scipy.io.loadmat('main/Data/cylinder_nektar_wake.mat')
           
    # Extract data arrays and convert to float32
    U_star = data['U_star'].astype(np.float32) # Velocity field: N x 2 x T (N points, 2 components, T time steps)
    P_star = data['p_star'].astype(np.float32) # Pressure field: N x T
    t_star = data['t'].astype(np.float32) # Time vector: T x 1
    X_star = data['X_star'].astype(np.float32) # Spatial coordinates: N x 2 (x, y)
    
    # Get data dimensions
    N = X_star.shape[0]  # Number of spatial points
    T = t_star.shape[0]   # Number of time steps
    
    # Rearrange data into flattened format for training
    # Create coordinate matrices by repeating spatial coordinates for each time step
    XX = np.tile(X_star[:,0:1], (1,T)) # x-coordinates: N x T
    YY = np.tile(X_star[:,1:2], (1,T)) # y-coordinates: N x T
    TT = np.tile(t_star, (1,N)).T # t-coordinates: N x T
    
    # Extract velocity components
    UU = U_star[:,0,:] # x-component of velocity: N x T
    VV = U_star[:,1,:] # y-component of velocity: N x T
    PP = P_star # Pressure: N x T
    
    # Flatten all arrays for neural network training
    x = XX.flatten()[:,None] # x-coordinates: NT x 1
    y = YY.flatten()[:,None] # y-coordinates: NT x 1
    t = TT.flatten()[:,None] # t-coordinates: NT x 1
    
    u = UU.flatten()[:,None] # x-velocity: NT x 1
    v = VV.flatten()[:,None] # y-velocity: NT x 1
    p = PP.flatten()[:,None] # pressure: NT x 1
    
    ######################################################################
    ######################## Noiseless Data ###############################
    ######################################################################
    # Training Data - Randomly sample training points from the full dataset
    idx = np.random.choice(N*T, N_train, replace=False)  # Random indices without replacement
    x_train = x[idx,:]  # Training x-coordinates
    y_train = y[idx,:]  # Training y-coordinates
    t_train = t[idx,:]  # Training t-coordinates
    u_train = u[idx,:]  # Training u-velocity
    v_train = v[idx,:]  # Training v-velocity

    # Train PINN model on clean data
    model = PhysicsInformedNN(x_train, y_train, t_train, u_train, v_train, layers)
    model.train(200000)  # Train for 200,000 Adam iterations + L-BFGS-B fine-tuning
    
    # Test Data - Use a specific time snapshot for evaluation
    snap = np.array([100])  # Time snapshot index
    x_star = X_star[:,0:1]  # All spatial x-coordinates
    y_star = X_star[:,1:2]  # All spatial y-coordinates
    t_star = TT[:,snap]     # Time coordinates for the snapshot
    
    # Extract true solution at the test snapshot
    u_star = U_star[:,0,snap]  # True u-velocity
    v_star = U_star[:,1,snap]  # True v-velocity
    p_star = P_star[:,snap]    # True pressure
    
    # Make predictions using trained model
    u_pred, v_pred, p_pred = model.predict(x_star, y_star, t_star)
    
    # Get identified parameters
    lambda_1_value = model.lambda_1.numpy()[0]
    lambda_2_value = model.lambda_2.numpy()[0]
    
    # Calculate prediction errors (relative L2 norm)
    error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
    error_v = np.linalg.norm(v_star-v_pred,2)/np.linalg.norm(v_star,2)
    error_p = np.linalg.norm(p_star-p_pred,2)/np.linalg.norm(p_star,2)

    # Calculate parameter identification errors
    error_lambda_1 = np.abs(lambda_1_value - 1.0)*100  # True value is 1.0
    error_lambda_2 = np.abs(lambda_2_value - 0.01)/0.01 * 100  # True value is 0.01
    
    # Print results
    print('Error u: %e' % (error_u))    
    print('Error v: %e' % (error_v))    
    print('Error p: %e' % (error_p))    
    print('Error l1: %.5f%%' % (error_lambda_1))                             
    print('Error l2: %.5f%%' % (error_lambda_2))                  
    
    # Plot Results (commented out - individual field plots)
    # plot_solution(X_star, u_pred, 1)  # Predicted u-velocity
    # plot_solution(X_star, v_pred, 2)  # Predicted v-velocity
    # plot_solution(X_star, p_pred, 3)  # Predicted pressure
    # plot_solution(X_star, p_star, 4)  # True pressure
    # plot_solution(X_star, p_star - p_pred, 5)  # Pressure error
    
    # Prepare data for comprehensive plotting
    # Get spatial bounds for plotting
    lb = X_star.min(0)  # Lower bounds [x_min, y_min]
    ub = X_star.max(0)  # Upper bounds [x_max, y_max]
    
    # Create regular grid for visualization
    nn = 200  # Grid resolution
    x = np.linspace(lb[0], ub[0], nn)  # x-coordinates
    y = np.linspace(lb[1], ub[1], nn)  # y-coordinates
    X, Y = np.meshgrid(x,y)  # 2D coordinate grid
    
    # Interpolate predictions to regular grid for plotting
    UU_star = griddata(X_star, u_pred.flatten(), (X, Y), method='cubic')
    VV_star = griddata(X_star, v_pred.flatten(), (X, Y), method='cubic')
    PP_star = griddata(X_star, p_pred.flatten(), (X, Y), method='cubic')
    P_exact = griddata(X_star, p_star.flatten(), (X, Y), method='cubic')
    
    
    ######################################################################
    ########################### Noisy Data ###############################
    ######################################################################
    # Test robustness by adding noise to training data
    noise = 0.01  # 1% noise level
        
    # Add Gaussian noise to velocity training data
    u_train = u_train + noise*np.std(u_train)*np.random.randn(u_train.shape[0], u_train.shape[1])
    v_train = v_train + noise*np.std(v_train)*np.random.randn(v_train.shape[0], v_train.shape[1])    

    # Train new model on noisy data
    model = PhysicsInformedNN(x_train, y_train, t_train, u_train, v_train, layers)
    model.train(200000)  # Same training procedure
        
    # Get identified parameters from noisy data training
    lambda_1_value_noisy = model.lambda_1.numpy()[0]
    lambda_2_value_noisy = model.lambda_2.numpy()[0]
      
    # Calculate parameter identification errors for noisy case
    error_lambda_1_noisy = np.abs(lambda_1_value_noisy - 1.0)*100
    error_lambda_2_noisy = np.abs(lambda_2_value_noisy - 0.01)/0.01 * 100
        
    # Print noisy data results
    print('Error l1: %.5f%%' % (error_lambda_1_noisy))                             
    print('Error l2: %.5f%%' % (error_lambda_2_noisy))     

             
    
    ######################################################################
    ############################# Plotting ###############################
    ######################################################################    
    # Load vorticity data for visualization
    data_vort = scipy.io.loadmat('main/Data/cylinder_nektar_t0_vorticity.mat')
           
    # Extract vorticity field data and convert to float32
    x_vort = data_vort['x'].astype(np.float32)  # x-coordinates for vorticity
    y_vort = data_vort['y'].astype(np.float32)  # y-coordinates for vorticity
    w_vort = data_vort['w'].astype(np.float32)  # vorticity values
    modes = int(data_vort['modes'])  # Number of spectral modes
    nel = int(data_vort['nel'])      # Number of elements
    
    # Reshape vorticity data to element-wise format
    xx_vort = np.reshape(x_vort, (modes+1,modes+1,nel), order = 'F')
    yy_vort = np.reshape(y_vort, (modes+1,modes+1,nel), order = 'F')
    ww_vort = np.reshape(w_vort, (modes+1,modes+1,nel), order = 'F')
    
    # Define plotting region bounds
    box_lb = np.array([1.0, -2.0])  # Lower bounds [x_min, y_min]
    box_ub = np.array([8.0, 2.0])   # Upper bounds [x_max, y_max]
    
    # Create main figure for vorticity visualization
    fig, ax = plt.subplots(figsize=(12, 10))  # Standard matplotlib figure
    ax.axis('off')  # Hide axes
    
    ####### Row 0: Vorticity Field ##################    
    gs0 = gridspec.GridSpec(1, 2)  # Grid layout for vorticity plot
    gs0.update(top=1-0.06, bottom=1-2/4 + 0.12, left=0.0, right=1.0, wspace=0)
    ax = plt.subplot(gs0[:, :])
    
    # Plot vorticity field for each element
    for i in range(0, nel):
        h = ax.pcolormesh(xx_vort[:,:,i], yy_vort[:,:,i], ww_vort[:,:,i], 
                         cmap='seismic', shading='gouraud', vmin=-3, vmax=3) 
    
    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    
    # Draw bounding box
    ax.plot([box_lb[0],box_lb[0]],[box_lb[1],box_ub[1]],'k',linewidth = 1)
    ax.plot([box_ub[0],box_ub[0]],[box_lb[1],box_ub[1]],'k',linewidth = 1)
    ax.plot([box_lb[0],box_ub[0]],[box_lb[1],box_lb[1]],'k',linewidth = 1)
    ax.plot([box_lb[0],box_ub[0]],[box_ub[1],box_ub[1]],'k',linewidth = 1)
    
    # Set plot properties
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title('Vorticity', fontsize = 10)
    
    
    ####### Row 1: Training Data Visualization ##################
    ########      u(t,x,y) - 3D Visualization     ###################        
    gs1 = gridspec.GridSpec(1, 2)  # Grid for velocity components
    gs1.update(top=1-2/4, bottom=0.0, left=0.01, right=0.99, wspace=0)
    ax = plt.subplot(gs1[:, 0],  projection='3d')  # 3D subplot for u-velocity
    ax.axis('off')  # Hide axes

    # Define 3D bounding box ranges
    r1 = [x_star.min(), x_star.max()]  # x-range
    r2 = [data['t'].min(), data['t'].max()]  # t-range       
    r3 = [y_star.min(), y_star.max()]  # y-range
    
    # Draw 3D bounding box edges
    for s, e in combinations(np.array(list(product(r1,r2,r3))), 2):
        if np.sum(np.abs(s-e)) == r1[1]-r1[0] or np.sum(np.abs(s-e)) == r2[1]-r2[0] or np.sum(np.abs(s-e)) == r3[1]-r3[0]:
            ax.plot3D(*zip(s,e), color="k", linewidth = 0.5)   

    # Plot training data points as scatter
    ax.scatter(x_train, t_train, y_train, s = 0.1)
    
    # Plot predicted u-velocity field as contour
    ax.contourf(X,UU_star,Y, zdir = 'y', offset = t_star.mean(), cmap='rainbow', alpha = 0.8)
              
    # Add axis labels
    ax.text(x_star.mean(), data['t'].min() - 1, y_star.min() - 1, '$x$')
    ax.text(x_star.max()+1, data['t'].mean(), y_star.min() - 1, '$t$')
    ax.text(x_star.min()-1, data['t'].min() - 0.5, y_star.mean(), '$y$')
    ax.text(x_star.min()-3, data['t'].mean(), y_star.max() + 1, '$u(t,x,y)$')    
    
    # Set 3D axis limits
    ax.set_xlim3d(r1)
    ax.set_ylim3d(r2)
    ax.set_zlim3d(r3)
    axisEqual3D(ax)  # Make axes equal scale
    
    ########      v(t,x,y) - 3D Visualization     ###################        
    ax = plt.subplot(gs1[:, 1],  projection='3d')  # 3D subplot for v-velocity
    ax.axis('off')  # Hide axes
    
    # Define 3D bounding box ranges (same as u-velocity)
    r1 = [x_star.min(), x_star.max()]  # x-range
    r2 = [data['t'].min(), data['t'].max()]  # t-range       
    r3 = [y_star.min(), y_star.max()]  # y-range
    
    # Draw 3D bounding box edges
    for s, e in combinations(np.array(list(product(r1,r2,r3))), 2):
        if np.sum(np.abs(s-e)) == r1[1]-r1[0] or np.sum(np.abs(s-e)) == r2[1]-r2[0] or np.sum(np.abs(s-e)) == r3[1]-r3[0]:
            ax.plot3D(*zip(s,e), color="k", linewidth = 0.5)   

    # Plot training data points as scatter
    ax.scatter(x_train, t_train, y_train, s = 0.1)
    
    # Plot predicted v-velocity field as contour
    ax.contourf(X,VV_star,Y, zdir = 'y', offset = t_star.mean(), cmap='rainbow', alpha = 0.8)
              
    # Add axis labels
    ax.text(x_star.mean(), data['t'].min() - 1, y_star.min() - 1, '$x$')
    ax.text(x_star.max()+1, data['t'].mean(), y_star.min() - 1, '$t$')
    ax.text(x_star.min()-1, data['t'].min() - 0.5, y_star.mean(), '$y$')
    ax.text(x_star.min()-3, data['t'].mean(), y_star.max() + 1, '$v(t,x,y)$')    
    
    # Set 3D axis limits
    ax.set_xlim3d(r1)
    ax.set_ylim3d(r2)
    ax.set_zlim3d(r3)
    axisEqual3D(ax)  # Make axes equal scale
    
    # Save the training data visualization figure
    # plt.savefig('./figures/NavierStokes_data.pdf') 

    # Create second figure for pressure comparison
    fig, ax = plt.subplots(figsize=(12, 8))  # Standard matplotlib figure
    ax.axis('off')  # Hide axes
    
    ######## Row 2: Pressure Field Comparison #######################
    ########      Predicted p(t,x,y)     ########### 
    gs2 = gridspec.GridSpec(1, 2)  # Grid for pressure plots
    gs2.update(top=1, bottom=1-1/2, left=0.1, right=0.9, wspace=0.5)
    ax = plt.subplot(gs2[:, 0])  # Left subplot for predicted pressure
    
    # Plot predicted pressure field
    h = ax.imshow(PP_star, interpolation='nearest', cmap='rainbow', 
                  extent=[x_star.min(), x_star.max(), y_star.min(), y_star.max()], 
                  origin='lower', aspect='auto')
    
    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    
    # Set plot properties
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_aspect('equal', 'box')
    ax.set_title('Predicted pressure', fontsize = 10)
    
    ########     Exact p(t,x,y)     ########### 
    ax = plt.subplot(gs2[:, 1])  # Right subplot for exact pressure
    
    # Plot exact pressure field
    h = ax.imshow(P_exact, interpolation='nearest', cmap='rainbow', 
                  extent=[x_star.min(), x_star.max(), y_star.min(), y_star.max()], 
                  origin='lower', aspect='auto')
    
    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)

    # Set plot properties
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_aspect('equal', 'box')
    ax.set_title('Exact pressure', fontsize = 10)
    
    
    ######## Row 3: Results Summary Table #######################
    gs3 = gridspec.GridSpec(1, 2)  # Grid for results table
    gs3.update(top=1-1/2, bottom=0.0, left=0.0, right=1.0, wspace=0)
    ax = plt.subplot(gs3[:, :])  # Full width subplot
    ax.axis('off')  # Hide axes
    
    # Create LaTeX table showing identified PDE parameters
    s = r'$\begin{tabular}{|c|c|}';  # Start table
    s = s + r' \hline'  # Top border
    s = s + r' Correct PDE & $\begin{array}{c}'  # Correct PDE column
    s = s + r' u_t + (u u_x + v u_y) = -p_x + 0.01 (u_{xx} + u_{yy})\\'  # x-momentum equation
    s = s + r' v_t + (u v_x + v v_y) = -p_y + 0.01 (v_{xx} + v_{yy})'  # y-momentum equation
    s = s + r' \end{array}$ \\ '  # End correct PDE
    s = s + r' \hline'  # Horizontal line
    s = s + r' Identified PDE (clean data) & $\begin{array}{c}'  # Clean data results
    s = s + r' u_t + %.3f (u u_x + v u_y) = -p_x + %.5f (u_{xx} + u_{yy})' % (lambda_1_value, lambda_2_value)
    s = s + r' \\'
    s = s + r' v_t + %.3f (u v_x + v v_y) = -p_y + %.5f (v_{xx} + v_{yy})' % (lambda_1_value, lambda_2_value)
    s = s + r' \end{array}$ \\ '  # End clean data
    s = s + r' \hline'  # Horizontal line
    s = s + r' Identified PDE (1\% noise) & $\begin{array}{c}'  # Noisy data results
    s = s + r' u_t + %.3f (u u_x + v u_y) = -p_x + %.5f (u_{xx} + u_{yy})' % (lambda_1_value_noisy, lambda_2_value_noisy)
    s = s + r' \\'
    s = s + r' v_t + %.3f (u v_x + v v_y) = -p_y + %.5f (v_{xx} + v_{yy})' % (lambda_1_value_noisy, lambda_2_value_noisy)
    s = s + r' \end{array}$ \\ '  # End noisy data
    s = s + r' \hline'  # Bottom border
    s = s + r' \end{tabular}$'  # End table
 
    # Display the table
    ax.text(0.015,0.0,s)
    
    # Save the prediction comparison figure
    # plt.savefig('./figures/NavierStokes_prediction.pdf') 

