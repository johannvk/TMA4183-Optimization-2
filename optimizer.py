import fenics as fe
import numpy as np


class AllenCahnOptimizer():

    def __init__(self, y_d: fe.Expression, y_0: fe.Expression, u_0: fe.Expression, 
                 spatial_function_space: fe.FunctionSpace, alpha=0.1, T=1.0, time_steps=10):
        # Time span of solution:
        self.T = T
        self.time_steps = time_steps

        # Make the function space over time:
        # With as many intervals in the Time-Interval 
        # function space as there are time steps:
        self.time_mesh = fe.IntervalMesh(self.time_steps, 0.0, self.T)
        self.time_V = fe.FunctionSpace(self.time_mesh, 'Lagrange', 3)

        # Function space of the spatial domain Omega.
        self.V = spatial_function_space

        # Desired distribution of the state variable:
        self.y_d = self.set_function(y_d, self.V) 

        # Initial condition of the state variable at time zero:
        self.y_0 = self.set_function(y_0, self.V)

        # Initialze the solution of the state equation at time T:
        self.y_T = fe.Function(self.V)

        # Intial guess for the control function u(t):
        self.u_t = self.set_function(u_0, self.V)

        # Objective for the optimizer:
        self.alpha = alpha
        self.J = 0.5*(self.y_T - self.y_d)**2 + 0.5*self.alpha*(self.u_t)**2

        # Solver parameters:
        pass
    
    def objective(self, y_T: fe.Function=None, u_t: fe.Function=None):
        # Need to be careful not to overwrite any of the functions
        # going in to the objective at a later point:

        if y_T is None and u_t is None:
            # Using the currently store functions 
            # self.y_T and self.u_t
            J = self.J

        elif y_T is not None and u_t is not None:
            # Need to have called 
            J = (y_T - self.y_d)**2 + (self.u_t)**2
        
        else:
            raise ValueError("Objective called with only one of the required arguments 'y_T' and 'u_t'.")

        return fe.assemble(J*fe.dx)

    def set_function(self, v, V):
        return v if isinstance(v, fe.Function) else fe.interpolate(v, V)