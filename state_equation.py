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


def Cahn_Hilliard_f(y, eps):
    return eps**2*(y**3 - y)


def time_step_system(y_n, dt, t, eps=0.01, mesh=None, V=None, ndof=64):
    # V = fe.FunctionSpace(mesh, 'Lagrange', poly_degree)

    # Do not need to specify Homogeneous Neumann BC's:
    # They are the default in FEniCS if no other BC is given.
    
    # As the problem is Non-Linear, we need to use 
    # 'Function' instead of 'TrialFunction'.
    y = fe.Function(V)

    # Our Linear test function:
    v = fe.TestFunction(V)

    # Control function g:
    g = fe.Expression("x[0] - x[1]*x[1] + 5*t", degree=2, t=t)
    
    # Does not work yet:
    # y_n_interp = fe.Function(V).interpolate(y_n)
    # l = dt*g*v*fe.dx + y_n_interp*v*fe.dx
    # a = u*v*fe.dx + dt*fe.inner(fe.grad(u), fe.grad(v))*fe.dx + \
    #     dt*eps**2*(u**3 - u)*v*fe.dx
    # F = a - l

    F_new = ((y - y_n)/dt)*v*fe.dx + fe.inner(fe.grad(y), fe.grad(v))*fe.dx \
            + eps**2*(y**3 - y)*v*fe.dx - g*v*fe.dx
    
    fe.solve(F_new == 0, y, solver_parameters={"newton_solver":{"relative_tolerance":1e-6}})
    return y


class InitialConditions(fe.UserExpression):
    def __init__(self, **kwargs):
        random.seed(2)
        super().__init__(**kwargs)

    def eval(self, values, x):
        values[0] = 0.63 + 0.5*(0.5 - random.random())

    def value_shape(self):
        return []


class StateEquationSolver():

    def __init__(self, mesh: fe.Mesh, function_space: fe.FunctionSpace, inital_condition: fe.Expression,
                 spatial_control: fe.Function, T: float, steps=50, eps: float=1.0e-3):
        self.mesh = mesh
        self.V = function_space

        # As the PDE is Non-Linear, we need to use 
        # 'Function' instead of 'TrialFunction'. 
        self.y = fe.Function(self.V)
        self.y_n = fe.Function(self.V)

        # Linear 'Test'-function:
        self.v = fe.TestFunction(self.V)

        if isinstance(inital_condition, fe.Function):
            self.initial_condition = inital_condition
        else:
            # Assume we got an fe.Expression object:
            self.initial_condition = fe.interpolate(inital_condition, self.V)
        
        if isinstance(spatial_control, fe.Function):
            self.spatial_control = spatial_control.copy()
        else:
            # Assume we got an fe.Expression object:
            self.spatial_control = fe.project(spatial_control, self.V)

        # Switch to true to visualize the spatial control:
        if True:
            g_plot = fe.plot(self.spatial_control)
            # set  colormap
            g_plot.set_cmap("viridis")
            # add a title to the  plot:
            plt.title("Spatial Control of solution.")
            # add a colorbar:
            plt.colorbar(g_plot)
            plt.show()

        self.eps = eps

        # Time parameters:
        self.T = T
        self.time_steps = steps
        self.dt = self.T/self.time_steps
        self.time_mesh = fe.IntervalMesh(self.time_steps, 0.0, self.T)
        self.time_V = fe.FunctionSpace(self.time_mesh, 'Lagrange', 2)

        self.eps_f_y = Cahn_Hilliard_f(self.y, self.eps)

        # May avoid constructing the big linear form each time:
        self.A = ((self.y - self.y_n)/self.dt)*self.v*fe.dx \
                 + fe.inner(fe.grad(self.y), fe.grad(self.v))*fe.dx \
                 + self.eps_f_y*self.v*fe.dx

        # Class parameters:
        self.newton_step_rel_tolerance = 1.0e-6

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
        self.y_n = self.initial_condition.copy()
        for i in range(self.time_steps):

            if save_steps:
                saved_steps[i] = (t, self.y_n.copy())
            
            if save_to_file:
                file << (self.y_n, t)

            t += self.dt

            # TODO: Could integrate and find average between (t, t + delta_t) if we wish.
            control_scale = temporal_control(t)
            self.time_step_system(u_t=control_scale)
            self.y_n.assign(self.y)

        # Save/Save last solution if wanted:
        if save_steps:
            saved_steps[i] = (t, self.y_n.copy())
            
        if save_to_file:
            file << (self.y_n, t)

        return self.y_n

    def time_step_system(self, u_t):
        # Do not need to specify Homogeneous Neumann BC's:
        # They are the default in FEniCS if no other BC is given.
        
        # Spatial control function g:
        g = self.spatial_control
        l = u_t*g*self.v*fe.dx

        F = self.A - l        
        fe.solve(F == 0, self.y, solver_parameters={"newton_solver":{"relative_tolerance":1e-6}})

    def plot_solution(self):
        print("Plotting final solution:")
        p = fe.plot(self.y_n)

        # set  colormap
        p.set_cmap("viridis")
        # p.set_clim(0, 1.0)
        # add a title to the  plot:
        plt.title("Cahn -Hilliard  solution")
        # add a colorbar:
        plt.colorbar(p)
        plt.show()


def make_single_time_step(y_n=None):
    mesh, V = define_unit_square_mesh()

    if y_n is None:
        y_init_expr = fe.Expression("0.2 - x[0] + x[1]", degree=2)
        y_n = fe.interpolate(y_init_expr, V)

    time_step_system(y_n, dt=0.01, t=2, mesh=mesh, V=V)


def print_mesh():
    mesh, V = define_unit_square_mesh()
    fe.plot(mesh)
    plt.show()
    print(V)


def main():

    # Seems to jump to a "zero-solution" very quickly.
    mesh, V = define_unit_square_mesh()
    init_cond = InitialConditions(degree=2)

    spatial_control = fe.Expression("sin(pi*x[0])*sin(pi*x[1])", degree=2)
    se_solver = StateEquationSolver(mesh=mesh, function_space=V, inital_condition=init_cond,
                                    spatial_control=spatial_control, T=1.0, steps=20)
    u_t = fe.Expression("5*x[0]", degree=1)

    se_solver.solve(u_t, save_steps=False, save_to_file=True, filename="test_6_bump_control")
    # se_solver.plot_solution()
    print("What!")


if __name__ == "__main__":
    main()
