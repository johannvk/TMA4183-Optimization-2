import fenics as fe
import numpy as np
import matplotlib.pyplot as plt

from optimizer import AllenCahnOptimizer

from problem_definitions import UnitSquareIslandIC, define_unit_square_mesh


def make_example_optimizer():
    
    eps = 0.1
    gamma = 0.5

    spatial_control = fe.Expression("5*(x[0] - 0.5)", degree=2)
    u_0 = fe.Expression("1 + 0.1*x[0]", degree=2)

    y_d = fe.Expression("-5*(x[0] - 0.5)", degree=2)

    UnitSquare_mesh, UnitSquare_V = define_unit_square_mesh(ndof=32)

    init_dict = {
        "spatial_function_space": UnitSquare_V,
        "y_d": y_d,
        "y_0": UnitSquareIslandIC(),
        "u_0": u_0,
        "spatial_control": spatial_control,
        "eps": eps,
        "gamma": gamma,
    }

    allen_cahn_optimizer = AllenCahnOptimizer.from_dict(init_dict)
    u_t = allen_cahn_optimizer.optimize(silent=False)

    fe.plot(allen_cahn_optimizer.set_function(u_0, allen_cahn_optimizer.time_V), label='initial')
    plt.legend(title='Original Temporal control')
    plt.show()

    fe.plot(u_t, label='optimal')
    plt.legend(title='Optimal Temporal control')
    plt.show()
    
    '''
    J0 = allen_cahn_optimizer.objective()
    print("J0:", J0)
    allen_cahn_optimizer.calculate_gradient()

    J0 = allen_cahn_optimizer.objective()
    print("J0:", J0)

    old_u_t = allen_cahn_optimizer.u_t
    gradient = allen_cahn_optimizer.gradient_function
    
    allen_cahn_optimizer.u_t.assign(old_u_t - 0.5*gradient)

    J1 = allen_cahn_optimizer.objective(allen_cahn_optimizer.y_T)
    print("J1:", J1)

    fe.plot(allen_cahn_optimizer.gradient_function)
    plt.title("Reduced cost-functional gradient!")
    plt.xlabel("Time t")
    plt.ylabel("Grad f(u)")
    plt.show()
    
    pass
    '''
    pass


def main():
    make_example_optimizer()
    pass

if __name__ == "__main__":
    main()
