import fenics as fe
import numpy as np

from adjoint_equation import AdjointEquationSolver
from state_equation import StateEquationSolver



class AllenCahnOptimizer():

    def __init__(self, y_d: fe.Expression, y_0: fe.UserExpression, u_0: fe.Expression, 
                 spatial_control: fe.Expression, spatial_function_space: fe.FunctionSpace, 
                 eps=1.0e-1, alpha=0.1, T=1.0, time_steps=10):
        # Phase 'strength':
        self.eps = eps

        # Time span of solution:
        self.T = T
        self.time_steps = time_steps

        # Make the function space over time:
        # With as many intervals in the Time-Interval 
        # function space as there are time steps:
        self.time_mesh = fe.IntervalMesh(self.time_steps, 0.0, self.T)
        self.time_V = fe.FunctionSpace(self.time_mesh, 'Lagrange', 2)

        # Function space of the spatial domain Omega.
        self.V = spatial_function_space
        # Set spatial control function:
        self.g = self.set_function(spatial_control, self.V)

        # Desired distribution of the state variable:
        self.y_d = self.set_function(y_d, self.V) 
        # Initial condition of the state variable at time zero:
        self.y_0 = self.set_function(y_0, self.V)
        # Initialze the solution of the state equation at time T:
        self.y_T = fe.Function(self.V)

        # Intial guess for the control function u(t):
        self.u_t = self.set_function(u_0, self.time_V)
        
        # Objective for the optimizer:
        self.alpha = alpha
        self.J_y = 0.5*(self.y_T - self.y_d)**2*fe.dx
        self.J_u = 0.5*self.alpha*(self.u_t)**2*fe.dx

        # State equation solver:
        self.state_equation = StateEquationSolver(
                               spatial_function_space=self.V, inital_condition=self.y_0,
                               spatial_control=self.g, T=self.T, steps=self.time_steps,
                               eps=self.eps, temporal_function_space=self.time_V
                               )

        # Adjoint equation solver:
        self.adjoint_equation = AdjointEquationSolver(
                               spatial_function_space=self.V, y_desired=self.y_d,
                               T=self.T, steps=self.time_steps,
                               eps=self.eps, temporal_function_space=self.time_V
                               )

    @classmethod
    def from_dict(cls, init_dict):
        return cls(**init_dict)
    
    def objective(self, y_T: fe.Function=None, u_t: fe.Function=None, save_steps=False):
        # Need to be careful not to overwrite any of the functions
        # going in to the objective at a later point:

        if y_T is None and u_t is None:
            # Use current value of 'self.u_t', which does not need to be changed.
            # Run forward and find self.y_T:
            y_T = self.state_equation.solve(temporal_control=self.u_t,
                                            save_steps=save_steps, save_to_file=False)
        
        elif y_T is not None and u_t is not None:
            # y_T does not need to be calculated, is given. 
            # Update 'self.u_t':
            self.u_t.assign(u_t)
        
        else:
            raise ValueError("Objective called with only one of the group-optional arguments 'y_T' and 'u_t'.")

        # Assign new function for the end time:
        self.y_T.assign(y_T)

        return fe.assemble(self.J_y) + fe.assemble(self.J_u)

    def set_function(self, v, V):
        return v if isinstance(v, fe.Function) else fe.interpolate(v, V)