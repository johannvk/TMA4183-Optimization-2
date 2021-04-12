import fenics as fe
# from fenics import dx

import numpy as np
import random
import matplotlib.pyplot as plt

from typing import Union


def define_unit_square_mesh(ndof=64):
    poly_degree = 2
    mesh = fe.UnitSquareMesh(ndof, ndof)

    V = fe.FunctionSpace(mesh, 'Lagrange', poly_degree)
    return (mesh, V)


def Allen_Cahn_f(y, eps):
    return eps**2*(y**3 - y)


class UnitSquareIslandIC(fe.UserExpression):
    def __init__(self, **kwargs):
        random.seed(2)
        super().__init__(**kwargs)

    def eval(self, values, x):
        # Homogeneous Neumann Conditions must be respected!
        # values[0] = 0.63 + 0.5*(0.5 - random.random())
        if 0.4 <= x[0] <= 0.6 and 0.4 <= x[1] <= 0.6:
            values[0] = 1.0
        else:
            values[0] = 0.0

    def value_shape(self):
        return []


class RandomNoiseIC(fe.UserExpression):
    def __init__(self, **kwargs):
        random.seed(2)
        super().__init__(**kwargs)

    def eval(self, values, x):
        # Homogeneous Neumann Conditions must be respected!
        # Rock the boat between [-1.0, 1.0]:
        value = 0.63 + 0.25*(0.5 - random.random())
        values[0] = max(min(value, 1.0), -1.0)

    def value_shape(self):
        return []


class StateEquationSolver():
    def set_function(self, v, V):
        return v if isinstance(v, fe.Function) else fe.interpolate(v, V)

    def __init__(self, spatial_function_space: fe.FunctionSpace, inital_condition: fe.Expression,
                 spatial_control: fe.Function, T: float, steps=50, eps: float=1.0e-3,
                 visualize_spatial_control=False, temporal_function_space: fe.FunctionSpace=None):

        self.V = spatial_function_space

        # As the Cahn-Hillard PDE is Non-Linear, we need to use 
        # 'Function' instead of 'TrialFunction' for the state y. 
        self.y = fe.Function(self.V)
        self.y_n = fe.Function(self.V)

        # Linear 'Test'-function:
        self.v = fe.TestFunction(self.V)

        self.initial_condition = self.set_function(inital_condition, self.V)
        
        self.spatial_control = self.set_function(spatial_control, self.V)

        # Time parameters:
        self.T = T
        self.time_steps = steps
        self.dt = self.T/self.time_steps

        if temporal_function_space is None:
            # As many intervals in the Time-Interval function space 
            # as there are time steps:
            self.time_mesh = fe.IntervalMesh(self.time_steps, 0.0, self.T)
            self.time_V = fe.FunctionSpace(self.time_mesh, 'Lagrange', 3)
        else:
            self.time_V = temporal_function_space

        # Allen-Cahn parameters:
        self.eps = eps
        self.eps_f_y = Allen_Cahn_f(self.y, self.eps)

        # May avoid constructing the big linear form each time step:
        self.A = ((self.y - self.y_n)/self.dt)*self.v*fe.dx \
                 + fe.inner(fe.grad(self.y), fe.grad(self.v))*fe.dx \
                 + self.eps_f_y*self.v*fe.dx
        
        self.u_t = fe.Constant(-1.0)
        self.l = self.u_t*self.spatial_control*self.v*fe.dx

        self.F = self.A - self.l
        self.dy = fe.TrialFunction(self.V)
        self.Jacobian = fe.derivative(self.F, self.y, self.dy)
        
        # Class parameters:
        self.newton_step_rel_tolerance = 1.0e-6

        # Switch to true to visualize the spatial control:
        if visualize_spatial_control:
            g_plot = fe.plot(self.spatial_control)
            # set  colormap
            g_plot.set_cmap("viridis")
            # add a title to the  plot:
            plt.title("Spatial Control of solution.")
            # add a colorbar:
            plt.colorbar(g_plot)
            plt.show()

    def control_expr(self, t):
        # Return spatial control at time t.
        # Scaled up or down version of the spatial control.
        pass
    
    def solve(self, temporal_control: Union[fe.Function, fe.Expression], save_steps=True,
              save_to_file=False, filename=""):

        if isinstance(temporal_control, fe.Expression):
            u_t = fe.interpolate(temporal_control, self.time_V)
        elif isinstance(temporal_control, fe.Function):
            u_t = temporal_control.copy()
        else:
            raise ValueError("Expexted 'temporal_control': {}\nof type fe.Function or fe.Expression". \
                             format(temporal_control))

        # Perform time-steps.
        saved_steps = {}

        if save_to_file:
            file = fe.File(f"results_state_equation/{filename}.pvd")

        t = 0.0

        # Set up the inital condition for y_n for the first round:
        self.y_n.assign(self.initial_condition)

        for i in range(self.time_steps):

            if save_steps:
                saved_steps[i] = (t, self.y_n.copy())
            
            if save_to_file:
                file << (self.y_n, t)

            t += self.dt

            # TODO: Could integrate and find average between (t, t + delta_t) if we wish.
            control_scale = u_t(t)
            self.time_step_system(u_t=control_scale)
            self.y_n.assign(self.y)

        # Save/Save last solution if wanted:
        i+=1
        if save_steps:
            saved_steps[i] = (t, self.y_n.copy())
            self.saved_steps = saved_steps
            
        if save_to_file:
            file << (self.y_n, t)

        return self.y_n

    def time_step_system(self, u_t):
        # Do not need to specify Homogeneous Neumann BC's:
        # They are the default in FEniCS if no other BC is given.

        self.u_t.assign(u_t)
        fe.solve(self.F == 0, self.y, J=self.Jacobian,
                solver_parameters={"newton_solver":
                                    {
                                        "relative_tolerance":
                                        self.newton_step_rel_tolerance
                                    }
                                   })

    def plot_solution(self):
        print("Plotting final solution:")
        p = fe.plot(self.y_n)

        # set  colormap
        p.set_cmap("viridis")

        # add a title to the  plot:
        plt.title("Allen-Cahn solution")
        # add a colorbar:
        plt.colorbar(p)
        plt.show()



def print_mesh():
    mesh, V = define_unit_square_mesh()
    fe.plot(mesh)
    plt.show()
    print(V)


def island_init_cond():
    
    # Seems to jump to a "zero-solution" very quickly.
    mesh, V = define_unit_square_mesh()

    init_cond = UnitSquareIslandIC(degree=3)

    # spatial_control = fe.Expression("sin(pi*x[0])*sin(pi*x[1])", degree=3)
    spatial_control = fe.Expression("5*(x[0] - 0.5)", degree=2)

    se_solver = StateEquationSolver(spatial_function_space=V, inital_condition=init_cond,
                                    spatial_control=spatial_control, T=0.05, steps=10, eps=0.1)
    
    u_t = fe.Expression("10*sin(2*pi*x[0]/0.1)", degree=3)

    se_solver.solve(u_t, save_steps=False, save_to_file=True, filename="unit_island_ramp_control_IC_1")
    se_solver.plot_solution()
    
    print("Done!")    


def random_init_cond():

    # Seems to jump to a "zero-solution" very quickly.
    mesh, V = define_unit_square_mesh()

    init_cond = RandomNoiseIC(degree=3)
 
    spatial_control = fe.Expression("5*(x[0] - 0.5)", degree=2)
    se_solver = StateEquationSolver(spatial_function_space=V, inital_condition=init_cond,
                                    spatial_control=spatial_control, T=1.0, steps=10, eps=0.1,
                                    visualize_spatial_control=False)

    # Ramp up intensity of control:
    u_t = fe.Expression("5*x[0]", degree=1)

    se_solver.solve(u_t, save_steps=False, save_to_file=True, filename="random_IC_1")
    se_solver.plot_solution()
    print("Done!")

    
def main():

    # random_init_cond()
    island_init_cond()

    # Phase field solutions:
    # Have zero control. Perturbed zero-solution. "Checkerboard" inital condition.
    # Without control, we should be able to see islands of the different phases developing.
    pass


if __name__ == "__main__":
    main()
