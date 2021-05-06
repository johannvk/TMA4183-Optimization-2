import fenics as fe
import numpy as np
import matplotlib.pyplot as plt

import sys
import os
import io

from adjoint_equation import AdjointEquationSolver
from state_equation import StateEquationSolver
from shut_up_fenics import stdout_redirector, normal_print


class AllenCahnGradient(fe.UserExpression):

    def __init__(self, gamma, temporal_control, spatial_control, adjoint_functions, **kwargs):
        
        self.gamma = gamma
        self.u_t = temporal_control
        self.g = spatial_control
        self.p = adjoint_functions

        # Need to connect indicies of adjoint_equations to times:
        self.ts = sorted([(i, t) for (i, (t, p_t)) in adjoint_functions.items()], key= lambda x: x[1])

        super().__init__(**kwargs)

    def eval(self, values, t):
        # Return value of gradient at time 't':
        t = t[0]

        # Linearly interpolate adjoint functions at times 
        # t_0 <= t <= t_1 if needed:
        index_coefficients = self.interpolate_time(t)

        p_t = sum(beta_i*self.p[i][1] for (i, beta_i) in index_coefficients)
        grad_t = self.gamma*self.u_t(t) + fe.assemble(self.g*p_t*fe.dx)

        values[0] = grad_t

    def interpolate_time(self, eval_t):
        beta_lower = None; beta_upper = None

        min_t = self.ts[0][1]; max_t = self.ts[-1][1]
        if (eval_t - min_t) < -5.0e-10:
            raise ValueError(f"Cannot evaluate gradient at time (t = {eval_t:.4f} < t_min = {min_t:.4f}).")
        elif (max_t - eval_t) < -5.0e-10:
            raise ValueError(f"Cannot evaluate gradient at time (t_max = {max_t:.4f} < t = {eval_t:.4f}).")

        t_lower = -np.inf; t_upper = np.inf
        i_lower = -1; i_upper = -1

        # First loop trough and check if any of the adjoint_functions at 
        # evaluated times are 'fe.near()' the evaluation time:
        for (i, t) in self.ts:
            if fe.near(eval_t, t, eps=1.0e-10):
                return [(i, 1.0)]

        # Else:
        # Find first time-interval [t_0, t_1] s.t. eval_t in [t_0, t_1]:
        for (k, (i, t)) in enumerate(self.ts[:-1]):
            if t < eval_t:
                t_lower = t; i_lower = i
                (i_upper, t_upper) = self.ts[k + 1]
                break
        else:
            raise ValueError("Could not establish time-interval eval_t in [t_min, t_max].")

        T_interval = t_upper - t_lower

        beta_lower = (t_upper - eval_t)/T_interval
        beta_upper = (eval_t - t_lower)/T_interval

        index_coefficients = ((i_lower, beta_lower), (i_upper, beta_upper))
        
        return index_coefficients
            
    def set_time_control_and_adjoint(self, time_control, adjoint_steps):
        self.u_t = time_control
        self.p = adjoint_steps

    def value_shape(self):
        return []


class AllenCahnOptimizer():

    def __init__(self, y_d: fe.Expression, y_0: fe.UserExpression, u_0: fe.Expression, 
                 spatial_control: fe.Expression, spatial_function_space: fe.FunctionSpace, 
                 eps=1.0e-1, gamma=0.1, T=1.0, time_steps=10, time_expr_degree=2, 
                 optimizer_params = [10, 0.0001, 10, 1, 1e-4], problem_name="test"):
        # Phase 'strength':
        self.eps = eps

        # Time span of solution:
        self.T = T
        self.time_steps = time_steps
        self.time_expr_degree = time_expr_degree

        # Make the function space over time:
        # With as many intervals in the Time-Interval 
        # function space as there are time steps:
        self.time_mesh = fe.IntervalMesh(self.time_steps, 0.0, self.T)
        self.time_V = fe.FunctionSpace(self.time_mesh, 'Lagrange', time_expr_degree)

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
        self.gamma = gamma
        self.J_y = 0.5*(self.y_T - self.y_d)**2*fe.dx
        self.J_u = 0.5*self.gamma*(self.u_t)**2*fe.dx

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

        # Gradient fe.UserExpression and fe.Function:
        self.gradient_expression: AllenCahnGradient = None
        self.gradient_function: fe.Function = fe.Function(self.time_V)
        
        # Prepare storage of gradient norms and objective values:
        self.gradient_norms = []
        self.objective_values = []
        
        # optimizer parameters
        self.optimizer_params=optimizer_params
        self.alpha = optimizer_params[3]

        self.problem_name = problem_name

    @classmethod
    def from_dict(cls, init_dict):
        return cls(**init_dict)
    
    def solve_state_equation(self, u_t: fe.Function=None, save_steps=False, save_file=False, filename=None):
        if u_t is None:
            u_t = self.u_t
        
        if save_file and filename is None:
            raise ValueError("Filename must be provided in order to save state equation solution to file.")
        elif save_file:
            filename = f"problems/{self.problem_name}/" + filename

        y_T = self.state_equation.solve(temporal_control=u_t, save_steps=save_steps, 
                                        save_to_file=save_file, filename=filename)

        return y_T
    
    def objective(self, y_T: fe.Function=None, u_t: fe.Function=None, save_steps=False):
        # Need to be careful not to overwrite any of the functions
        # going in to the objective at a later point:

        if u_t is None:
            u_t = self.u_t
        else:
            self.u_t.assign(u_t)

        if y_T is None:
            # Calculate new value for y_T
            # Run forward and find self.y_T:
            y_T = self.solve_state_equation(u_t, save_steps=save_steps)

        self.y_T.assign(y_T)
        objective_value = fe.assemble(self.J_y) + fe.assemble(self.J_u)
        return objective_value

    def calculate_gradient(self):
        # Use current control 'self.u_t', and calculate
        # self.p:
        y_T = self.solve_state_equation(self.u_t, save_steps=True)
        self.y_T.assign(y_T)

        state_equation_solution = self.state_equation.saved_steps

        # Find adjoint equation solution:
        self.adjoint_equation.load_y(state_equation_solution)
        self.adjoint_equation.solve(save_steps=True)

        adjoint_equation_solution = self.adjoint_equation.saved_steps
        
        if self.gradient_expression is None:
            self.gradient_expression = AllenCahnGradient(self.gamma, self.u_t, self.g, 
                                                        adjoint_equation_solution, 
                                                        degree=self.time_expr_degree)
        else:
            self.gradient_expression.set_time_control_and_adjoint(self.u_t, adjoint_equation_solution)

        # Interpolate requires a lot less function evaluations, i think:
        # Would be nice if we could cotrol exactly which time-nodes are in
        # self.time_V.
        self.gradient_function.assign(fe.interpolate(self.gradient_expression, self.time_V))

    def armijo_satisfied(self, new_objective, old_objective, gradient_L2_norm, 
                         alpha, c_armijo):        
        return new_objective <= old_objective - self.alpha*c_armijo*gradient_L2_norm

    def line_search(self):
        '''Performs line search in gradient direction, with armijo contions.
        self.u_t is updated with new values'''
        max_iter = self.optimizer_params[2]
        c = self.optimizer_params[4]
        
        # TEST TEST:
        # c = 0.0

        with mute():
            old_evaluation = self.objective(self.y_T) # assumes self.y_T is correct
        print(f'old evaluation: {old_evaluation}')

        old_u_t = self.u_t.copy()
        new_u_t = old_u_t.copy()

        # TODO: Should maybe Copy, so the gradient function itself is not 
        #       tarnished?
        # step = self.gradient_function # .copy()
        step = fe.Function(self.time_V)
        gradient_L2_norm_squared = fe.assemble(self.gradient_function**2*fe.dx)

        # Store gradient norms:
        self.gradient_norms.append(gradient_L2_norm_squared**(0.5))

        print(f"\n||Gradient||^2_(L2): {gradient_L2_norm_squared}\n")

        for _ in range(max_iter):
            step.assign(-self.alpha*self.gradient_function)

            # Need to project the sums onto the function space:
            new_u_t.assign(fe.project(old_u_t + step, self.time_V))

            with mute():
                new_evaluation = self.objective(u_t = new_u_t)

            print(f'new evaluation: {new_evaluation}')
            
            # Made it much more explicit that we are using correct Armijo conditions:
            if self.armijo_satisfied(new_evaluation, old_evaluation, 
                                     gradient_L2_norm_squared, self.alpha, c):
                print(f'Accepted alpha: {self.alpha}')
                # Prevent using small steps
                self.alpha *= 1.5 

                # Save the new Objective value:
                self.objective_values.append(new_evaluation)
                break

            else:
                self.alpha /= 2
        else:
            print(f"Line seach did not satisfy armijo conditions in {max_iter} steps.")
            self.u_t.assign(old_u_t) # Undo Step.
            # TODO: Why return 0? This ends the optimization I guess.
            return 0

        return old_evaluation - new_evaluation

    def optimize(self, silent=True):
        '''Optimize u_t
        silent: decides if output from fenics is printed to terminal'''

        # mute if silent is True
        global mute
        mute = stdout_redirector if silent else normal_print

        # Stopping parameters
        max_iter = self.optimizer_params[0]
        tol = self.optimizer_params[1]

        # Loop gradient calcutation and line search
        for _ in range(max_iter):
            with mute():
                self.calculate_gradient()
            decreased = self.line_search()
            print(f"\nDecreased: {decreased}\n")
            if decreased < tol:
                print(f"\nAbsolute Tolerance {tol:.4f} has been reached.\nEnding Optimization")
                break
        
        return self.u_t

    def set_function(self, v, V):
        return v if isinstance(v, fe.Function) else fe.interpolate(v, V)

    def plot_gradient_norms(self):
        from matplotlib.ticker import MaxNLocator
        
        fig, axis = plt.subplots(1, 1, figsize=(10, 6))
        
        steps = np.arange(len(self.gradient_norms))
        axis.plot(steps, np.log10(np.array(self.gradient_norms)))
        
        axis.set_title("Gradient Norm Convergence")
        axis.set_xlabel("Steps")
        axis.set_ylabel(r"$log_{10}\left(||\nabla f(u)||_{L_2(\Omega)}\right)$")
        
        # Add grid lines:
        axis.grid(which="major", axis="both")

        # Set Integer Tick Marks for the Steps on the X-axis.
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))

        plt.show()
    
    def plot_objective_values(self):
        from matplotlib.ticker import MaxNLocator
        
        fig, axis = plt.subplots(1, 1, figsize=(10, 6))
        
        steps = np.arange(len(self.objective_values))
        axis.plot(steps, np.log10(np.array(self.objective_values)))
        
        axis.set_title("Reduced Objective Convergence")
        axis.set_xlabel("Steps")
        axis.set_ylabel(r"$log_{10}\left(f(u)\right)$")
        
        # Add grid lines:
        axis.grid(which="major", axis="both")

        # Set Integer Tick Marks for the Steps on the X-axis.
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))

        plt.show()
