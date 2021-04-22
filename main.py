import fenics as fe
import numpy as np
import matplotlib.pyplot as plt

from optimizer import AllenCahnOptimizer

from problem_definitions import UnitSquareIslandIC, define_unit_square_mesh, RandomUnitSquareQuadrantIC


def mock_problem():
    
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

    allen_cahn_optimizer.plot_gradient_norms()

    fe.plot(allen_cahn_optimizer.set_function(u_0, allen_cahn_optimizer.time_V), label='initial')
    plt.legend(title='Original Temporal control')
    plt.show()

    fe.plot(u_t, label='optimal')
    plt.legend(title='Optimal Temporal control')
    plt.show()
    pass


def constructed_problem_1(ndof=32):
    # Rebound, near the optimal end-state without much us of 'u'.
    save_files = False
    testnumber = 1 

    eps = 0.1
    gamma = 0.5

    bump_spatial_control = fe.Expression("sin(pi*x[0])*sin(pi*x[1]) - 0.5", degree=2)

    # Construct the desired control:
    # u_d = fe.Expression("sin(2*pi*x[0])", degree=2)
    u_d = fe.Expression("-1*(x[0] + 1)*(x[0] - 1)", degree=2)

    temp_y_d = fe.Expression("1", degree=2)

    UnitSquare_mesh, UnitSquare_V = define_unit_square_mesh(ndof=ndof)

    init_dict = {
        "spatial_function_space": UnitSquare_V,
        "y_d": temp_y_d,
        "y_0": RandomUnitSquareQuadrantIC(),
        "u_0": u_d,
        "spatial_control": bump_spatial_control,
        "eps": eps,
        "gamma": gamma,
        "time_steps": 20,
        "T": 2.0,
        "problem_name": "rebound"
    }

    allen_cahn_optimizer = AllenCahnOptimizer.from_dict(init_dict)
    allen_cahn_optimizer.state_equation.visualize_spatial_control()

    # Run once to find the end state:
    actual_y_d = allen_cahn_optimizer.solve_state_equation(save_file=save_files, filename=f"Test{testnumber}_desired_state") 

    # Plot the constructed desired end state:
    g_plot = fe.plot(actual_y_d)
    # set  colormap
    g_plot.set_cmap("viridis")
    # add a title to the  plot:
    plt.title("Constructed Desired End State")
    # add a colorbar:
    plt.colorbar(g_plot)
    plt.show()

    # Set the desired end state to the one found from solving the state-equation
    # with the desired control 'u_d':
    """
    allen_cahn_optimizer.y_d = actual_y_d

    # Set a new starting control function:
    allen_cahn_optimizer.u_t = allen_cahn_optimizer.\
                               set_function(u_0, allen_cahn_optimizer.time_V)
    """
    # Define a new inital temporal control:
    u_0 = fe.Expression("1.0", degree=2)

    # Make a new optimizer:
    new_init_dict = {
        "spatial_function_space": UnitSquare_V,
        "y_d": actual_y_d,
        "y_0": RandomUnitSquareQuadrantIC(),
        "u_0": u_0,
        "spatial_control": bump_spatial_control,
        "eps": eps,
        "gamma": gamma,
        "time_steps": 20,
        "T": 2.0,
        "problem_name": "rebound"
    }
    new_allen_cahn_optimizer = AllenCahnOptimizer.from_dict(new_init_dict)

    # Display Purposes:
    # Save first state equation from initial temporal control:
    new_allen_cahn_optimizer.solve_state_equation(save_file=save_files, filename=f"Test{testnumber}_original_control")

    # Perform the optimization:
    u_t = allen_cahn_optimizer.optimize(silent=False)

    # Save the last state equation from optimal temporal control:
    allen_cahn_optimizer.solve_state_equation(u_t=u_t, save_file=save_files, filename=f"Test{testnumber}_optimal_control")

    allen_cahn_optimizer.plot_gradient_norms()
    allen_cahn_optimizer.plot_objective_values()
    
    fe.plot(allen_cahn_optimizer.set_function(u_d, allen_cahn_optimizer.time_V), 
            label='State Generating Control')
    plt.title("State Generating Control")
    plt.xlabel("Time t, [0, T = 2]")
    plt.ylabel("Temporal Control u(t)")
    plt.legend(title='State Equation-Generating Temporal control')
    plt.show()

    fe.plot(allen_cahn_optimizer.set_function(u_0, allen_cahn_optimizer.time_V), 
            label='Initial')
    plt.title("Intial Guess Control")
    plt.xlabel("Time t, [0, T = 2]")
    plt.ylabel("Temporal Control u(t)")
    plt.legend(title='Initial Guess Temporal control')
    plt.show()

    fe.plot(u_t, label='Optimal Control')
    plt.title("Final, Optimal, Control")
    plt.xlabel("Time t, [0, T = 2]")
    plt.ylabel("Temporal Control u(t)")
    plt.legend(title='Optimal Temporal control')
    plt.show()
   
    
    print("WHAT!")


def constructed_problem_2(ndof=32):
    # Rebound, near the optimal end-state without much us of 'u'.
    testnumber = 2
    save_files = False
    # Would like to lower the tolerance/End on gradient norm instead of 
    # absolute decrease in objective value.

    # Big Eps:
    eps = 0.5
    
    # Small Gamma:
    gamma = 0.01

    bump_spatial_control = fe.Expression("sin(pi*x[0])*sin(pi*x[1]) - 0.5", degree=2)

    # Construct the desired control:
    # u_d = fe.Expression("sin(2*pi*x[0])", degree=2)
    u_d = fe.Expression("-1*(x[0] + 1)*(x[0] - 1)", degree=2)

    temp_y_d = fe.Expression("1", degree=2)

    UnitSquare_mesh, UnitSquare_V = define_unit_square_mesh(ndof=ndof)

    init_dict = {
        "spatial_function_space": UnitSquare_V,
        "y_d": temp_y_d,
        "y_0": RandomUnitSquareQuadrantIC(),
        "u_0": u_d,
        "spatial_control": bump_spatial_control,
        "eps": eps,
        "gamma": gamma,
        "time_steps": 20,
        "T": 2.0,
        "problem_name": "low_gamma"
    }

    allen_cahn_optimizer = AllenCahnOptimizer.from_dict(init_dict)
    allen_cahn_optimizer.state_equation.visualize_spatial_control()

    # Run once to find the end state:
    actual_y_d = allen_cahn_optimizer.solve_state_equation(save_file=save_files, filename=f"Test{testnumber}_desired_state") 

    # Plot the constructed desired end state:
    g_plot = fe.plot(actual_y_d)
    # set  colormap
    g_plot.set_cmap("viridis")
    # add a title to the  plot:
    plt.title("Constructed Desired End State")
    # add a colorbar:
    plt.colorbar(g_plot)
    plt.show()

    # Set the desired end state to the one found from solving the state-equation
    # with the desired control 'u_d':
    """
    allen_cahn_optimizer.y_d = actual_y_d

    # Set a new starting control function:
    allen_cahn_optimizer.u_t = allen_cahn_optimizer.\
                               set_function(u_0, allen_cahn_optimizer.time_V)
    """
    # Define a new inital temporal control:
    u_0 = fe.Expression("1.0", degree=2)

    # Make a new optimizer:
    new_init_dict = {
        "spatial_function_space": UnitSquare_V,
        "y_d": actual_y_d,
        "y_0": RandomUnitSquareQuadrantIC(),
        "u_0": u_0,
        "spatial_control": bump_spatial_control,
        "eps": eps,
        "gamma": gamma,
        "time_steps": 20,
        "T": 2.0,
        "problem_name": "low_gamma"
    }
    new_allen_cahn_optimizer = AllenCahnOptimizer.from_dict(new_init_dict)

    # Display Purposes:
    # Save first state equation from initial temporal control:
    new_allen_cahn_optimizer.solve_state_equation(save_file=save_files, filename=f"Test{testnumber}_original_control")

    # Perform the optimization:
    u_t = allen_cahn_optimizer.optimize(silent=False)

    # Save the last state equation from optimal temporal control:
    allen_cahn_optimizer.solve_state_equation(u_t=u_t, save_file=save_files, filename=f"Test{testnumber}_optimal_control")

    allen_cahn_optimizer.plot_gradient_norms()
    allen_cahn_optimizer.plot_objective_values()

    fe.plot(allen_cahn_optimizer.set_function(u_d, allen_cahn_optimizer.time_V), 
            label='State Generating Control')
    plt.title("State Generating Control")
    plt.xlabel("Time t, [0, T = 2]")
    plt.ylabel("Temporal Control u(t)")
    plt.legend(title='State Equation-Generating Temporal control')
    plt.show()

    fe.plot(allen_cahn_optimizer.set_function(u_0, allen_cahn_optimizer.time_V), 
            label='Initial')
    plt.title("Intial Guess Control")
    plt.xlabel("Time t, [0, T = 2]")
    plt.ylabel("Temporal Control u(t)")
    plt.legend(title='Initial Guess Temporal control')
    plt.show()

    fe.plot(u_t, label='Optimal Control')
    plt.title("Final, Optimal, Control")
    plt.xlabel("Time t, [0, T = 2]")
    plt.ylabel("Temporal Control u(t)")
    plt.legend(title='Optimal Temporal control')
    plt.show()
    
    print("WHAT!")


def main():
    fe.set_log_active(False)
    # mock_problem()

    # constructed_problem_1()

    constructed_problem_2()

    
    pass

if __name__ == "__main__":
    main()
