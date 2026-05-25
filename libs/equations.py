import numpy as np
import torch

from libs import settings

class EuropeanPut:
    def __init__(self):
        # 1-D Analytical Black-Scholes Price: 5.5735
        self.D = 1                      # number of dimensions
        self.S = 10                     # number of time intervals
        self.T = 1.0                    # final time
        self.sigma = 0.2                # volatility
        self.y_init = [1,3]             # initial guess for put price
        self.x_init = 100               # stock price at time 0
        self.r = 0.05                   # risk-free interest rate
        self.K = 100                    # strike price
        self.mu = self.r                # risk-neutral, drift equal rate
        self.dt = self.T/self.S         # time step size
        self.sqrt_dt = np.sqrt(self.dt) # for variance of dW
        self.learning_boundaries = [2500]
        self.learning_values = [5e-3, 5e-3]

    def forward_Xt(self, n_paths):
        """
        Simulate the forward process X_t
        """
        # generate random variables ~ N(0,dt)
        dW = np.random.standard_normal((n_paths, self.D, self.S)).astype(settings.NP_DTYPE) * self.sqrt_dt
        # initialize the forward process X with initial value x
        X = np.zeros((n_paths, self.D, self.S+1), dtype=settings.NP_DTYPE)
        X[:,:,0] = np.ones((n_paths, self.D)) * self.x_init
        # simulate the forward process
        for i in range(self.S):
            X[:,:,i+1] = X[:,:,i]*np.exp((self.mu - (self.sigma**2)/2)*self.dt + self.sigma*dW[:,:,i])
        return X, dW

    def backward_g(self, t, x):
        """
        Payoff of the option, terminal condition of the PDE
        """
        # the discounted price at terminal time
        obj = torch.max(x[:,:,-1], dim=1, keepdim=True)[0]
        return torch.maximum(self.K - obj, torch.zeros_like(obj))

    def backward_f(self, t, x, y, z):
        """
        Generator function of the PDE
        """
        term = torch.sum(z, dim=1, keepdim=True)/self.sigma
        return -self.r*(y - term) - term*self.mu
    
class EuropeanPut100D:
    def __init__(self):
        # 100-D Simulated Numerical Reference Price / Explicit Integral: 0.15035
        self.D = 100                       # number of dimensions
        self.S = 5                         # number of time intervals
        self.T = 1.0                       # final time
        self.sigma = 0.2                   # volatility
        self.y_init = [1,3]                # initial guess for put price
        self.x_init = 100                  # stock price at time 0
        self.r = 0.05                      # risk-free interest rate
        self.K = 150                       # strike price
        self.mu = self.r                   # risk-neutral, drift equal rate
        self.dt = self.T/self.S            # time step size
        self.sqrt_dt = np.sqrt(self.dt)    # for variance of dW
        self.learning_boundaries = [2500]
        self.learning_values = [5e-3, 5e-3]

    def forward_Xt(self, n_paths):
        """
        Simulate the forward process X_t
        """
        # generate random variables ~ N(0,dt)
        dW = np.random.standard_normal((n_paths, self.D, self.S)).astype(settings.NP_DTYPE) * self.sqrt_dt
        # initialize the forward process X with initial value x
        X = np.zeros((n_paths, self.D, self.S+1), dtype=settings.NP_DTYPE)
        X[:,:,0] = np.ones((n_paths, self.D)) * self.x_init
        # simulate the forward process
        for i in range(self.S):
            X[:,:,i+1] = X[:,:,i]*np.exp((self.mu - (self.sigma**2)/2)*self.dt + self.sigma*dW[:,:,i])
        return X, dW
    
    def backward_g(self, t, x):
        """
        Payoff of the option, terminal condition of the PDE
        """
        # x[:,:,-1] to get the value of x at terminal time
        obj = torch.max(x[:,:,-1], dim=1, keepdim=True)[0]
        return torch.maximum(self.K - obj, torch.zeros_like(obj))
    
    def backward_f(self, t, x, y, z):
        """
        Generator function of the PDE
        """
        term = torch.sum(z, dim=1, keepdim=True)/self.sigma
        return -self.r*(y - term) - term*self.mu

class EuropeanPutDiffRate:
    def __init__(self):
        # 1-D Simulated Numerical Reference Price: 5.7740
        self.D = 1                          # number of dimensions
        self.S = 10                         # number of time intervals
        self.T = 1.0                        # final time
        self.sigma = 0.2                    # volatility
        self.y_init = [1,3]                 # initial guess for put price
        self.x_init = 100                   # stock price at time 0
        self.r1 = 0.01                      # lending interest rate
        self.r2 = 0.05                      # borrowing interest rate
        self.K = 100                        # strike price
        self.mu = self.r2                   # risk-neutral, drift equal rate
        self.dt = self.T/self.S             # time step size
        self.sqrt_dt = np.sqrt(self.dt)     # time step size
        self.learning_boundaries = [2500]
        self.learning_values = [5e-3, 5e-3]

    def forward_Xt(self, n_paths):
        """
        Simulate the forward process X_t
        """
        # generate random variables ~ N(0,dt)
        dW = np.random.standard_normal((n_paths, self.D, self.S)).astype(settings.NP_DTYPE) * self.sqrt_dt
        # initialize the forward process X with initial value x
        X = np.zeros((n_paths, self.D, self.S+1), dtype=settings.NP_DTYPE)
        X[:,:,0] = np.ones((n_paths, self.D)) * self.x_init
        # simulate the forward process
        for i in range(self.S):
            X[:,:,i+1] = X[:,:,i]*np.exp((self.mu - (self.sigma**2)/2)*self.dt + self.sigma*dW[:,:,i])
        return X, dW

    def backward_g(self, t, x):
        """
        Payoff of the option, terminal condition of the PDE
        """
        # x[:,:,-1] to get the value of x at terminal time
        obj = torch.max(x[:,:,-1], dim=1, keepdim=True)[0]
        return torch.maximum(self.K - obj, torch.zeros_like(obj))

    def backward_f(self, t, x, y, z):
        """
        Generator function of the PDE
        """
        term = torch.sum(z, dim=1, keepdim=True)/self.sigma
        cash = y - term

        cash_positive = torch.maximum(cash, torch.zeros_like(cash))
        cash_negative = torch.maximum(-cash, torch.zeros_like(cash))


        return -self.r1*cash_positive + self.r2*cash_negative - term*self.mu
    
class EuropeanPutDiffRate100D:
    def __init__(self):
        # 100-D Simulated Numerical Reference Price: 0.4960
        self.D = 100                        # number of dimensions
        self.S = 10                         # number of time intervals
        self.T = 1.0                        # final time
        self.sigma = 0.2                    # volatility
        self.y_init = [1,3]                 # initial guess for put price
        self.x_init = 100                   # stock price at time 0
        self.r1 = 0.01                      # lending interest rate
        self.r2 = 0.05                      # borrowing interest rate
        self.K = 150                        # strike price
        self.mu = self.r2                   # risk-neutral, drift equal rate
        self.dt = self.T/self.S             # time step size
        self.sqrt_dt = np.sqrt(self.dt)     # time step size
        self.learning_boundaries = [2500]
        self.learning_values = [5e-3, 5e-3]

    def forward_Xt(self, n_paths):
        """
        Simulate the forward process X_t
        """
        # generate random variables ~ N(0,dt)
        dW = np.random.standard_normal((n_paths, self.D, self.S)).astype(settings.NP_DTYPE) * self.sqrt_dt
        # initialize the forward process X with initial value x
        X = np.zeros((n_paths, self.D, self.S+1), dtype=settings.NP_DTYPE)
        X[:,:,0] = np.ones((n_paths, self.D)) * self.x_init
        # simulate the forward process
        for i in range(self.S):
            X[:,:,i+1] = X[:,:,i]*np.exp((self.mu - (self.sigma**2)/2)*self.dt + self.sigma*dW[:,:,i])
        return X, dW

    def backward_g(self, t, x):
        """
        Payoff of the option, terminal condition of the PDE
        """
        # x[:,:,-1] to get the value of x at terminal time
        obj = torch.max(x[:,:,-1], dim=1, keepdim=True)[0]
        return torch.maximum(self.K - obj, torch.zeros_like(obj))

    def backward_f(self, t, x, y, z):
        " generator function of the PDE "
        term = torch.sum(z, dim=1, keepdim=True)/self.sigma
        cash = y - term
        cash_positive = torch.maximum(cash, torch.zeros_like(cash))
        cash_negative = torch.maximum(-cash, torch.zeros_like(cash))

        return -self.r1*cash_positive + self.r2*cash_negative - term*self.mu
