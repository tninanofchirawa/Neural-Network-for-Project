# Neural-Network-for-Project
This is the code for the Project "Enhancing Neural Network Models Through Binomial Theorem and Combinatorial Geometric Series Integration"

#The Code

#Coded in googlecolab
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scipy.special import factorial
import matplotlib.pyplot as plt

# Function for combinatorial binomial series
def the_series(x, n, r):
    # Computes the combinatorial series:
    factr = factorial(r)
    i_vals = np.arange(n + 1)
    prdt = np.prod(i_vals[:, None] + np.arange(1, r + 1), axis=1)
    series = (prdt / factr) * (x ** i_vals)
    return np.sum(series)


# Load and preprocess stock market data
def pre_data(stk_price):
    """
    Preprocess stock prices to compute percentage changes as observations.
    Args:
        stock_prices (array): Array of daily closing prices.
    Returns:
        observations (array): Percentage changes in stock prices.
    """
    perct_change = np.diff(stk_price) / stk_price[:-1] * 100
    return perct_change.reshape(-1, 1)  # Reshape for model compatibility

    
# Problem Setup: Stock Market Prediction
N = 3  # Number of hidden states (e.g., bull, bear, neutral)
T = 100  # Number of observations (e.g., daily price changes)
d = 1  # Dimension of observation vector (percentage change in price)
n = 3  # Maximum degree of series
r = 2  # Number of terms in combinatorial product


# Simulated stock price data 
np.random.seed(42)  # For reproducibility
stk_price = np.cumsum(np.random.normal(0, 1, T + 1)) + 100  # Simulated prices
obv = pre_data(stk_price)


# Transition probabilities (random initialization)
A = np.random.rand(N, N)
A /= A.sum(axis=1, keepdims=True)  # Normalize rows to sum to 1


# Initial probabilities (random initialization)
pi = np.random.rand(N)
pi /= pi.sum()


# Build a neural network for emission probabilities
mdl = Sequential([
    Dense(16, activation='relu', input_shape=(d,)),
    Dense(N, activation='softmax')  # Output size = number of hidden states ])
mdl.compile(optimizer='adam', loss='categorical_crossentropy')


# Train the neural network (dummy data for demonstration)
trgt = np.eye(N)[np.random.choice(N, T)]  # Random one-hot targets
mdl.fit(obv, trgt, epochs=10, verbose=1)


# Function to compute emission probabilities
def emi_prob(obs):
    nn_output = mdl.predict(obs, verbose=0)  # Neural network predictions
    emis = []
    for state_idx in range(N):
        state_probs = nn_output[:, state_idx]
        # Compute combinatorial series for all time steps
        emis.append([
            the_series(state_probs[t], n, r) for t in range(len(obs))   ])
    return np.array(emis).T  # Shape: (T, N)


# Forward algorithm
def for_algo(obs, verbose=False):
    B = emi_prob(obs)  # Emission probabilities (T x N)
    alpha = np.zeros((len(obs), N))
    # Initialization
    alpha[0, :] = pi * B[0, :]
    if verbose:
        print(f"Initial alpha:\n{alpha[0, :]}")
    # Recursion
    for t in range(1, len(obs)):
        alpha[t, :] = B[t, :] * np.dot(alpha[t - 1, :], A)
    if verbose:
        print(f"Final alpha:\n{alpha}")
    return alpha


# Backward algorithm
def back_algo(obs, verbose=False):
    B = emi_prob(obs)  # Emission probabilities (T x N)
    beta = np.zeros((len(obs), N))
    # Initialization
    beta[-1, :] = 1
    if verbose:
        print(f"Initial beta:\n{beta[-1, :]}")
    # Recursion
    for t in range(len(obs) - 2, -1, -1):
        beta[t, :] = np.dot(A, (B[t + 1, :] * beta[t + 1, :]))
    if verbose:
        print(f"Final beta:\n{beta}")
    return beta


# Compute forward and backward probabilities
alpha = for_algo(obv, verbose=True)
beta = back_algo(obv, verbose=False)

# Likelihood of the observation sequence
like = np.sum(alpha[-1, :])

# Plot results
plt.figure(figsize=(12, 6))

# Plot stock prices
plt.subplot(2, 1, 1)
plt.plot(stk_price, label='Stock Prices', color='blue')
plt.title('Simulated Stock Prices')
plt.xlabel('Time Steps')
plt.ylabel('Price')
plt.legend()

# Plot alpha probabilities for each state
plt.subplot(2, 1, 2)
for state in range(N):
    plt.plot(alpha[:, state], label=f'State {state + 1}')

plt.title('Forward Probabilities (Alpha) for Hidden States')
plt.xlabel('Time Steps')
plt.ylabel('Probability')
plt.legend()
plt.tight_layout()
plt.show()

# Print results
print("Likelihood of observations:", like)
