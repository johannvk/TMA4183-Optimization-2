import fenics as fe
import matplotlib.pyplot as plt
from problem_definitions import define_unit_square_mesh


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
