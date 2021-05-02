import fenics as fe
import numpy as np
import random
import matplotlib.pyplot as plt
import state_equation


def y_desired(degree):
    return fe.Expression("sin(pi*x[0])*sin(pi*x[1])", degree=degree)


class AdjointEquationSolver():

    def __init__(self, spatial_function_space: fe.FunctionSpace, T: float,
                 y_desired: fe.Expression, temporal_function_space: fe.FunctionSpace=None, 
                 steps:int=50, eps: float=1.0):
        # Do not really use the mesh:
        # self.mesh = mesh
        self.V = spatial_function_space
        self.eps = eps
        self.y_desired = y_desired

        # Time parameters:
        self.T = T
        self.time_steps = steps
        self.dt = self.T/self.time_steps

        if temporal_function_space is None:
            self.time_mesh = fe.IntervalMesh(self.time_steps, 0.0, self.T)
            self.time_V = fe.FunctionSpace(self.time_mesh, 'Lagrange', 2)
        else:
            self.time_V = temporal_function_space


        ### Define Variational Problem ###

        # Unknown next Trial step Argument:
        self.p = fe.TrialFunction(self.V)
        # Function to store solution in:
        self.p_sol = fe.Function(self.V)
        # Know step:
        self.p_n = fe.Function(self.V)
        
        # Driving state-equation solution:
        self.y_n = fe.Function(self.V)
        # Storage of the State-Equation solution:
        self.y = None 

        # Linear 'Test'-function:
        self.h = fe.TestFunction(self.V)
        
        # Class parameters: 
        # Only solve linear problem, do not need Newton's method.
        # self.newton_step_rel_tolerance = 1.0e-6
        self.one = fe.Constant(1)
        self.F = ((self.p_n - self.p)/self.dt)*self.h*fe.dx \
                   - fe.inner(fe.grad(self.p), fe.grad(self.h))*fe.dx \
                   - self.eps**2*(3*self.y_n**2 - self.one)*self.p*self.h*fe.dx

        # Sort out the left- and right-hand sides:
        self.a = fe.lhs(self.F)
        self.L = fe.rhs(self.F)

        # Storage for the saved adjoint equation steps:
        self.saved_steps = None

    def load_y(self, y):
        self.y = y
        
        # Desired solution Cannot be 
        # hard-coded this way:
        # y_d = y_desired(degree=2)
        y_d = self.y_desired        

        if not isinstance(y_d, fe.Function):
            # Assume we got an fe.Expression object:
            y_d = fe.project(y_d, self.V)

        yN = y_d.copy() #fe.Function(self.V)
        yN.assign(self.y[self.time_steps][1])

        # Assign initial condition: 
        # TODO: Do this in the 'solve' step, if we wish to 
        #       call '.solve()' several times. 
        # self.p_n = fe.Function(self.V)
        self.p_n.assign(yN.copy() - y_d.copy())

    def solve(self, save_steps=True, save_to_file=False, filename='', verbose=False):
        # Need to call 'load_y()' before solving.

        saved_steps = {}

        if save_to_file:
            file = fe.File(f"{filename}.pvd")

        t = self.T
        y = fe.Function(self.V)

        for i in range(self.time_steps):
            if save_steps:
                saved_steps[i] = (t, self.p_n.copy())
            if save_to_file:
                file << (self.p_n, t)
            
            y.assign(self.y[self.time_steps-i-1][1])
            
            self.time_step_system()
            self.p_n.assign(self.p_sol)

            t -= self.dt

        # Save/Save last solution if wanted:
        if save_steps:
            saved_steps[i] = (t, self.p_n.copy())
            self.saved_steps = saved_steps

        if save_to_file:
            file << (self.p_n, t)

        return self.p_n

    def time_step_system(self):
        # Do not need to specify Homogeneous Neumann BC's:
        # They are the default in FEniCS if no other BC is given.
        # The equation is linear in 'p':
        solver_parameters = {"linear_solver": "lu"}
        fe.solve(self.a == self.L, self.p_sol, 
                 solver_parameters=solver_parameters)

    def plot_solution(self):
        print('Plotting solution for t=0')
        p = fe.plot(self.p_n)

        # set  colormap
        p.set_cmap("viridis")
        # p.set_clim(0, 1.0)
        # add a title to the  plot:
        plt.title("Cahn - Hilliard Adjoint solution")
        # add a colorbar:
        plt.colorbar(p)
        plt.show()


def island_init_cond(ndof=64, T=0.05, steps=10, eps=0.1):
    
    # --- We first solve the state equation
    mesh, V = state_equation.define_unit_square_mesh(ndof = ndof)
    init_cond = state_equation.UnitSquareIslandIC(degree=3)

    #spatial_control = fe.Expression("sin(pi*x[0])*sin(pi*x[1])", degree=3)
    spatial_control = fe.Expression("5*(x[0] - 0.5)", degree=2)
    
    se_solver = state_equation.StateEquationSolver(spatial_function_space=V, inital_condition=init_cond,
                                    spatial_control=spatial_control, T=T, steps=steps, eps=eps)

    u_t = fe.Expression("10*sin(2*pi*x[0]/0.1)", degree=3)
    se_solver.solve(u_t, save_steps=True, save_to_file=True, filename="unit_island_ramp_control_IC_1")
    
    se_solver.plot_solution()

    # --- Now for the adjoint part:
    ae_solver = AdjointEquationSolver(spatial_function_space=V, y_desired=y_desired(4),
                                      T=T, steps=steps, eps=eps)
    ae_solver.load_y(se_solver.saved_steps)

    ae_solver.solve(save_steps=False, save_to_file=False, filename="unit_island_ramp_control_IC_1")
    
    # For visualizing final p
    ae_solver.plot_solution()
    print("Done")


def random_init_cond(ndof=64,T=1.0,steps=10,eps=0.1):

    # --- We first solve the state equation
    mesh, V = state_equation.define_unit_square_mesh(ndof = ndof)
    init_cond = state_equation.RandomNoiseIC(degree=3)
    spatial_control = fe.Expression("5*(x[0] - 0.5)", degree=2)
    se_solver = state_equation.StateEquationSolver(spatial_function_space=V, inital_condition=init_cond,
                                    spatial_control=spatial_control, T=T, steps=steps, eps=eps,
                                    visualize_spatial_control=False)
    u_t = fe.Expression("5*x[0]", degree=1)
    se_solver.solve(u_t, save_steps=True, save_to_file=False, filename="random_IC_1")
    se_solver.plot_solution()

    # --- Now for the adjoint part:
    ae_solver = AdjointEquationSolver(spatial_function_space=V, y_desired=y_desired(4), 
                                      T=T, steps=steps, eps=eps)
    ae_solver.load_y(se_solver.saved_steps)
    #ae_solver.plot_solution() # For visualizing final p
    ae_solver.solve(save_steps=False, save_to_file=False, filename="random_IC_1")
    ae_solver.plot_solution()
    print("Done")


def adjoint_main():
    ndof=64
    T = 1.0
    steps = 10
    eps = 0.1

    # random_init_cond(ndof,T,steps,eps)
    island_init_cond(ndof,T,steps,eps)
    print('we are done now')


if __name__ == "__main__":
    adjoint_main()
