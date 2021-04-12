import fenics as fe
import numpy as np
import random
import matplotlib.pyplot as plt
import state_equation


def y_desired(degree):
    return fe.Expression("sin(pi*x[0])*sin(pi*x[1])", degree=degree)

class AdjointEquationSolver():
    def __init__(self, mesh: fe.Mesh, function_space: fe.FunctionSpace, T: float, steps:int=50, eps: float=1.0):
        self.mesh = mesh
        self.V = function_space
        self.p = fe.Function(self.V)

        # Linear 'Test'-function:
        self.h = fe.TestFunction(self.V)
        
        self.eps = eps

        # Time parameters:
        self.T = T
        self.time_steps = steps
        self.dt = self.T/self.time_steps
        self.time_mesh = fe.IntervalMesh(self.time_steps, 0.0, self.T)
        self.time_V = fe.FunctionSpace(self.time_mesh, 'Lagrange', 2)

        # Class parameters:
        self.newton_step_rel_tolerance = 1.0e-6

    def load_y(self, y):
        self.y = y
        
        y_d = y_desired(degree=2)
        if not isinstance(y_d, fe.Function):
            # Assume we got an fe.Expression object:
            y_d= fe.project(y_d, self.V)

        yN=y_d.copy() #fe.Function(self.V)
        yN.assign(self.y[self.time_steps][1])

        self.p_n = fe.Function(self.V)
        self.p_n.assign(yN.copy() - y_d.copy())

    def solve(self, save_steps=True, save_to_file=False, filename=''):

        saved_steps = {}

        if save_to_file:
            file = fe.File(f"results_adjoint_equation/{filename}.pvd")

        t = self.T
        y=fe.Function(self.V)
        for i in range(self.time_steps):
            if save_steps:
                saved_steps[i] = (t, self.p_n.copy())
            if save_to_file:
                file << (self.p_n, t)
            
            y.assign(self.y[self.time_steps-i][1])
            self.A = ((self.p_n - self.p)/self.dt)*self.h*fe.dx \
                     - fe.inner(fe.grad(self.p), fe.grad(self.h))*fe.dx \
                     - self.eps**2*(3*y**2-1)*self.p*self.h*fe.dx
            self.time_step_system()
            self.p_n.assign(self.p)
            t -= self.dt

        # Save/Save last solution if wanted:
        if save_steps:
            saved_steps[i] = (t, self.p_n.copy())
            
        if save_to_file:
            file << (self.p_n, t)

        return self.p_n

    def time_step_system(self):
        # Do not need to specify Homogeneous Neumann BC's:
        # They are the default in FEniCS if no other BC is given.
        fe.solve(self.A == 0, self.p, solver_parameters={"newton_solver":{"relative_tolerance":1e-6}})

    def plot_solution(self):
        print('Plotting solution for t=0')
        p = fe.plot(self.p_n)

        # set  colormap
        p.set_cmap("viridis")
        # p.set_clim(0, 1.0)
        # add a title to the  plot:
        plt.title("Cahn -Hilliard adjoint solution")
        # add a colorbar:
        plt.colorbar(p)
        plt.show()




def island_init_cond(ndof=64,T=0.05,steps=10,eps=0.1):
    
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
    ae_solver = AdjointEquationSolver(mesh=mesh, function_space=V, T=T, steps=steps, eps=eps)
    ae_solver.load_y(se_solver.saved_steps)
    #ae_solver.plot_solution() # For visualizing final p
    ae_solver.solve(save_steps=False, save_to_file=False, filename="unit_island_ramp_control_IC_1")
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
    ae_solver = AdjointEquationSolver(mesh=mesh, function_space=V, T=T, steps=steps, eps=eps)
    ae_solver.load_y(se_solver.saved_steps)
    #ae_solver.plot_solution() # For visualizing final p
    ae_solver.solve(save_steps=False, save_to_file=False, filename="random_IC_1")
    ae_solver.plot_solution()
    print("Done")

def main():
    ndof=64
    T = 1.0
    steps = 10
    eps = 0.1

    random_init_cond(ndof,T,steps,eps)
    island_init_cond()
    print('we are done now')

if __name__ == "__main__":
    main()
