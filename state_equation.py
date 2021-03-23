import fenics as fe
# from fenics import dx

import numpy as np
import matplotlib.pyplot as plt


def define_mesh(ndof=64):
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
        # random.seed(2 + MPI.rank(MPI.comm_world))
        super().__init__(**kwargs)
    
    def eval(self, values, x):
        values = 0.63
        
    def value_shape(self):
         return ()


class StateEquationSolver():

    def __init__(self, mesh: fe.Mesh, function_space: fe.FunctionSpace, inital_condition: fe.Function,
                 spatial_control: fe.Function, T: float, eps: float=1.0e-3):
        self.mesh = mesh
        self.V = function_space

        # As the PDE is Non-Linear, we need to use 
        # 'Function' instead of 'TrialFunction'. 
        self.y = fe.Function(self.V)

        # Linear 'Test'-function:
        self.v = fe.TestFunction(self.V)

        if isinstance(inital_condition, fe.Function):
            self.initial_condition = inital_condition
        else:
            # Assume we got an fe.Expression object:
            self.initial_condition = fe.project(inital_condition, self.V)
        
        if isinstance(spatial_control, fe.Function):
            self.spatial_control = spatial_control.copy()
        else:
            # Assume we got an fe.Expression object:
            self.spatial_control = fe.project(spatial_control, self.V)

        self.T = T
        self.eps = eps

        # Class parameters:
        self.newton_step_rel_tolerance = 1.0e-6

        self.time_steps = 50
        self.time_mesh = fe.IntervalMesh(nx=self.time_steps, 0.0, self.T)
        self.time_V = fe.FunctionSpace(time_mesh, 'Lagrange', degree=2)

    def control_expr(self, t):
        # Return spatial control at time t.
        # Scaled up or down version of the spatial control.
        pass
    
    def solve(self, temporal_control: fe.Function, steps=None, save_steps=True):
        if steps is None:
            steps = self.time_steps
        
        if isinstance(temporal_control, fe.Expression):
            u_t = fe.interpolate(temporal_control, self.time_V)
        elif isinstance(temporal_control, fe.Function):
            u_t = temporal_control.copy()
        else:
            raise ValueError("Expexted 'temporal_control': {}\nof type fe.Function or fe.Expression". \
                             format(temporal_control))

        # Perform time-steps.
        delta_t = self.T / steps
        saved_steps = {}

        t = 0.0
        y_n = self.initial_condition.copy()
        for i in range(steps):

            if save_steps:
                saved_steps[i] = (t, y_n)
            
            t += delta_t
            control_scale = temporal_control(t)
            self.time_step_system(y_n, dt=delta_t, t=t, u_t=control_scale)
            y_n.assign(self.y)

        return y_n

    def time_step_system(self, y_n, dt, t, u_t):
        # Do not need to specify Homogeneous Neumann BC's:
        # They are the default in FEniCS if no other BC is given.
        
        # Control function g:
        g = self.spatial_control
        eps_f_y = Cahn_Hilliard_f(self.y, self.eps)

        # May avoid constructing the big linear form each time:
        a = ((self.y - y_n)/dt)*self.v*fe.dx + fe.inner(fe.grad(self.y), fe.grad(self.v))*fe.dx \
            + eps_f_y*v*fe.dx
        l = u_t*g*v*fe.dx

        F = a - l        
        fe.solve(F == 0, self.y, solver_parameters={"newton_solver":{"relative_tolerance":1e-6}})


def make_single_time_step(y_n=None):
    mesh, V = define_mesh()

    if y_n is None:
        y_init_expr = fe.Expression("0.2 - x[0] + x[1]", degree=2)
        y_n = fe.interpolate(y_init_expr, V)

    time_step_system(y_n, dt=0.01, t=2, mesh=mesh, V=V)


def print_mesh():
    mesh, V = define_mesh()
    fe.plot(mesh)
    plt.show()
    print(V)


def main():
    
    y_1 = make_single_time_step()
    y_2 = make_single_time_step(y_1)


if __name__ == "__main__":
    main()
