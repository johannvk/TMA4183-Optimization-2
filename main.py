import fenics as fe
import numpy as np
import matplotlib.pyplot as plt

from optimizer import AllenCahnOptimizer

from problem_definitions import UnitSquareIslandIC, define_unit_square_mesh

def make_example_optimizer():
    
    eps = 0.1
    alpha = 0.5

    spatial_control = fe.Expression("5*(x[0] - 0.5)", degree=2)
    u_0 = fe.Expression("5*x[0]", degree=2)

    y_d = fe.Expression("-5*(x[0] - 0.5)", degree=2)

    UnitSquare_mesh, UnitSquare_V = define_unit_square_mesh(ndof=32)

    init_dict = {
        "spatial_function_space": UnitSquare_V,
        "y_d": y_d,
        "y_0": UnitSquareIslandIC(),
        "u_0": u_0,
        "spatial_control": spatial_control,
        "eps": eps,
        "alpha": alpha,
    }

    allen_cahn_optimizer = AllenCahnOptimizer.from_dict(init_dict)

    J0 = allen_cahn_optimizer.objective()
    print("J0:", J0)
    pass


def main():
    make_example_optimizer()
    pass


if __name__ == "__main__":
    main()
