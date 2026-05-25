import numpy as np
import torch
import torch.nn as nn
from time import time

from libs import settings

class DeepBSDE(nn.Module):
    def __init__(self, eq, log_name=None):
        # initialize of nn.Module
        super().__init__()
        # initialize other variables
        self.eq = eq
        self.t_space = np.linspace(0, self.eq.T, self.eq.S + 1)
        self.learning_iterations = 5000
        self.initial_batch_size = 256
        self.train_batch_size = 64
        
        # optimizer for gradient descent
        lr = torch.optim.lr_scheduler.StepLR(
                torch.optim.Adam([torch.tensor(0.0)], lr=5e-3, eps=1e-8),
                step_size=2500, gamma=1.0)
        # initialize u(0,X_T)
        self.Y_init = nn.Parameter(torch.tensor(np.random.uniform(eq.y_init[0], eq.y_init[1], size=[1]), dtype=settings.DTYPE))
        # initialize the gradient ∇u(0,X_T)
        self.Z_init = nn.Parameter(torch.tensor(np.random.uniform(-.1, .1, size=(1, eq.D)), dtype=settings.DTYPE))
        # create neural networks for approximating Z_t, for all timesteps
        self.nets = nn.ModuleList([self._create_nn() for _ in range(eq.S - 1)])

    def _create_nn(self):
        """ create a neural network with architecture: Input -> (Dense -> BN -> ReLU)*2 -> Dense """
        return nn.Sequential(
            nn.Linear(self.eq.D, self.eq.D + 10, bias=False),
            nn.BatchNorm1d(self.eq.D + 10),
            nn.ReLU(),
            nn.Linear(self.eq.D + 10, self.eq.D + 10, bias=False),
            nn.BatchNorm1d(self.eq.D + 10),
            nn.ReLU(),
            #nn.Linear(self.eq.D + 10, self.eq.D + 10, bias=False),
            #nn.BatchNorm1d(self.eq.D + 10),
            #nn.ReLU(),
            #nn.Linear(self.eq.D + 10, self.eq.D + 10, bias=False),
            #nn.BatchNorm1d(self.eq.D + 10),
            #nn.ReLU(),
            nn.Linear(self.eq.D + 10, self.eq.D, bias=False)
        )


    def _simulate_Y(self, X, dW):
        """ simulate the backward process Y_N ≈ u(T, X_T) """

        n_samples = X.shape[0]
        # approximate u(0,X_T)
        y = torch.ones([n_samples, 1], dtype=settings.DTYPE) * self.Y_init
        # approximate gradient ∇u(0,X_T)
        z = torch.ones([n_samples, 1], dtype=settings.DTYPE) * self.Z_init

        # Convert numpy arrays to tensors
        X = torch.tensor(X, dtype=settings.DTYPE)
        dW = torch.tensor(dW, dtype=settings.DTYPE)

        # approximate the backward process Y
        for i in range(self.eq.S - 1):
            t = self.t_space[i]
            # Euler-Maruyama approximation of Y at t_{i+1}
            y = y - self.eq.backward_f(t, X[:,:,i], y, z)*self.eq.dt + \
                torch.sum(z * dW[:,:,i], dim=1, keepdim=True)
            # approximate Z at t_{i+1}
            z = self.nets[i](X[:,:,i+1])/self.eq.D

        # compute Y at terminal time T
        y = y - self.eq.backward_f(self.t_space[-1], X[:,:,-1], y, z)*self.eq.dt + \
            torch.sum(z * dW[:,:,-1], dim=1, keepdim=True)

        return y

    def _loss_fn(self, X, dW):
        """ function to minimize, MSE of Y_T and g(X_T) 
        """
        # simulate the backward process with forward process and training data
        y_pred = self._simulate_Y(X, dW)
        # evaluate g(X_T)
        X_tensor = torch.tensor(X, dtype=settings.DTYPE, device=y_pred.device)
        y = self.eq.backward_g(self.eq.T, X_tensor)
        # Mean squared error
        loss = torch.mean(torch.square(y - y_pred))

        return loss

    def train(self):
        """
        train the neural network, i.e. minimize the loss function
        """
        s_time = time()

        # Set up optimizer with piecewise constant decay
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-3, eps=1e-8)

        # simulate the forward process
        X, dW = self.eq.forward_Xt(self.initial_batch_size)
        # stochastic gradient descent
        for i in range(self.learning_iterations+1):
            # Update learning rate based on boundaries
            if i in self.eq.learning_boundaries:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.eq.learning_values[self.eq.learning_boundaries.index(i)]

            loss = self._train_step(X, dW, optimizer)

            if i % 100 == 0:
                print(
                    f"i: {i:4} Y0: {self.Y_init.item():6.3f} "
                    f"Loss: {loss.item():5.3f} "
                    f"Time: {time() - s_time:5.1f}"
                )

                X, dW = self.eq.forward_Xt(self.train_batch_size)

    def _train_step(self, X, dW, optimizer):
        optimizer.zero_grad()
        loss = self._loss_fn(X, dW)
        loss.backward()
        optimizer.step()

        return loss
