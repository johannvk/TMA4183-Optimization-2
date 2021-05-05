import fenics as fe

import matplotlib.pyplot as plt

from optimizer import AllenCahnOptimizer
import initial_conditions as IC

# Storage of problem definitions:
# 
# {
#  'mesh': fe.Mesh,  
#  'initial_conditions': fe.UserExpression, 
#  'y_desired': fe.Expression,
#  'spatial_control': fe.Expression, 
#  'y_initial': fe.Expression,
# }



def mock_problem():
    
    eps = 0.1
    gamma = 0.5

    spatial_control = fe.Expression("5*(x[0] - 0.5)", degree=2)
    u_0 = fe.Expression("1 + 0.1*x[0]", degree=2)

    y_d = fe.Expression("-5*(x[0] - 0.5)", degree=2)

    _, UnitSquare_V = IC.define_unit_square_mesh(ndof=32)

    init_dict = {
        "spatial_function_space": UnitSquare_V,
        "y_d": y_d,
        "y_0": IC.UnitSquareIslandIC(),
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
    save_files = True
    testnumber = 2

    eps = 0.1
    gamma = 0.5

    bump_spatial_control = fe.Expression("sin(pi*x[0])*sin(pi*x[1]) - 0.5", degree=2)

    # Construct the desired control:
    # u_d = fe.Expression("sin(2*pi*x[0])", degree=2)
    u_d = fe.Expression("-1*(x[0] + 1)*(x[0] - 1)", degree=2)

    temp_y_d = fe.Expression("1", degree=2)

    _, UnitSquare_V = IC.define_unit_square_mesh(ndof=ndof)

    T = 4.0
    time_steps = int(T/0.1)

    init_dict = {
        "spatial_function_space": UnitSquare_V,
        "y_d": temp_y_d,
        "y_0": IC.RandomUnitSquareQuadrantIC(degree=3),
        # "y_0": IC.UnitSquareQuadrantIC(degree=3),
        "u_0": u_d,
        "spatial_control": bump_spatial_control,
        "eps": eps,
        "gamma": gamma,
        "time_steps": time_steps,
        "T": T,
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
    # Define a new inital temporal control:
    u_0 = fe.Expression("1.0", degree=2)

    # Make a new optimizer:
    new_init_dict = {
        "spatial_function_space": UnitSquare_V,
        "y_d": actual_y_d,
        "y_0": IC.RandomUnitSquareQuadrantIC(),
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
    u_t = new_allen_cahn_optimizer.optimize(silent=False)

    # Save the last state equation from optimal temporal control:
    new_allen_cahn_optimizer.solve_state_equation(u_t=u_t, save_file=save_files, filename=f"Test{testnumber}_optimal_control")

    new_allen_cahn_optimizer.plot_gradient_norms()
    new_allen_cahn_optimizer.plot_objective_values()
    
    fe.plot(new_allen_cahn_optimizer.set_function(u_d, new_allen_cahn_optimizer.time_V), 
            label='State Generating Control')
    plt.title("State Generating Control")
    plt.xlabel("Time t, [0, T = 2]")
    plt.ylabel("Temporal Control u(t)")
    plt.legend(title='State Equation-Generating Temporal control')
    plt.show()

    fe.plot(new_allen_cahn_optimizer.set_function(u_0, new_allen_cahn_optimizer.time_V), 
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


def constructed_problem_2(ndof=32, testnumber=5, save_files=False):
    # Rebound, near the optimal end-state without much us of 'u'.

    # Would like to lower the tolerance/End on gradient norm instead of 
    # absolute decrease in objective value.

    # Big Eps:
    eps = 0.1
    
    # Small Gamma:
    gamma = 1.0e-3

    bump_spatial_control = fe.Expression("sin(pi*x[0])*sin(pi*x[1]) - 0.5", degree=2)

    # Construct the desired control:
    u_d = fe.Expression("-1*(x[0] + 1)*(x[0] - 1)", degree=2)

    temp_y_d = fe.Expression("1", degree=2)

    T = 4.0
    time_steps = int(T/0.1)

    _, UnitSquare_V = IC.define_unit_square_mesh(ndof=ndof)

    init_dict = {
        "spatial_function_space": UnitSquare_V,
        "y_d": temp_y_d,
        "y_0": IC.RandomUnitSquareQuadrantIC(),
        "u_0": u_d,
        "spatial_control": bump_spatial_control,
        "eps": eps,
        "gamma": gamma,
        "time_steps": time_steps,
        "T": T,
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

    # Define a new inital temporal control:
    u_0 = fe.Expression("1.0", degree=2)

    # Make a new optimizer:
    new_init_dict = {
        "spatial_function_space": UnitSquare_V,
        "y_d": actual_y_d,
        "y_0": IC.RandomUnitSquareQuadrantIC(),
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
    u_t = new_allen_cahn_optimizer.optimize(silent=False)

    # Save the last state equation from optimal temporal control:
    new_allen_cahn_optimizer.solve_state_equation(u_t=u_t, save_file=save_files, filename=f"Test{testnumber}_optimal_control")

    new_allen_cahn_optimizer.plot_gradient_norms()
    new_allen_cahn_optimizer.plot_objective_values()

    fe.plot(new_allen_cahn_optimizer.set_function(u_d, new_allen_cahn_optimizer.time_V), 
            label='State Generating Control')
    plt.title("State Generating Control")
    plt.xlabel("Time t, [0, T = 2]")
    plt.ylabel("Temporal Control u(t)")
    plt.legend(title='State Equation-Generating Temporal control')
    plt.show()

    fe.plot(new_allen_cahn_optimizer.set_function(u_0, new_allen_cahn_optimizer.time_V), 
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


def constructed_problem_3(ndof=32, testnumber=1, save_files = False):
    construct_end_state = False
    # Rebound, near the optimal end-state without much us of 'u'.

    # Would like to lower the tolerance/End on gradient norm instead of 
    # absolute decrease in objective value.

    # Big Eps:
    eps = 1.0e0
    
    # Small Gamma:
    gamma = 1.0e-2

    # The same as y_0:
    # square_spatial_control = IC.UnitSquareIslandIC(min_x=0.2, min_y=0.3, degree=3)
    d_min_x = d_min_y = 0.2
    orig_min_x = orig_min_y = 0.3
    # Almost the same as y_d, but can drive downward outside the desired "high" region:
    square_spatial_control = IC.UnitSquareIslandIC(min_x=d_min_x , min_y=d_min_y, high_level=1.0, 
                                                   low_level=-1.0, degree=3)

    # Want to Contract the initial square condition:
    y_0 = IC.UnitSquareIslandIC(min_x=orig_min_x, min_y=orig_min_y, degree=3)
    # y_0 = IC.UnitSquareQuadrantIC(degree=3)

    y_d = IC.UnitSquareIslandIC(min_x=d_min_x, min_y=d_min_y, high_level=1.0, 
                                low_level=-1.0, degree=3)

    T = 4.0
    time_steps = int(T/0.1)

    u_d = fe.Expression("x[0]*x[0] + 1", T=T, degree=3)

    _, UnitSquare_V = IC.define_unit_square_mesh(ndof=ndof)

    init_dict = {
        "spatial_function_space": UnitSquare_V,
        "y_d": y_d,
        "y_0": y_0,
        "u_0": u_d,
        "spatial_control": square_spatial_control,
        "eps": eps,
        "gamma": gamma,
        "time_steps": time_steps,
        "T": T,
        "problem_name": "square_y_d"
    }

    allen_cahn_optimizer = AllenCahnOptimizer.from_dict(init_dict)
    allen_cahn_optimizer.state_equation.visualize_spatial_control()

    if construct_end_state:
        actual_y_d = allen_cahn_optimizer.solve_state_equation(save_file=False)

        # Reset the desired state, and the initial control:
        allen_cahn_optimizer.y_d = actual_y_d

        # Plot the constructed desired end state:
        g_plot = fe.plot(fe.project(actual_y_d, allen_cahn_optimizer.V))
        # set  colormap
        g_plot.set_cmap("viridis")
        # add a title to the  plot:
        plt.title("Naiive Desired y_d End State")
        # add a colorbar:
        plt.colorbar(g_plot)
        plt.show()
    
    u_0 = fe.project(fe.Expression("x[0] < T/2.0 ? 1.0 : 0.0", T=T, degree=2), 
                     allen_cahn_optimizer.time_V)
    allen_cahn_optimizer.u_t = u_0.copy()

    # Run once to find the end state for the initial condition u_0:
    initial_y_T = allen_cahn_optimizer.solve_state_equation(u_t=u_0, 
                   save_file=save_files, filename=f"Test{testnumber}_initial_control") 

    # Perform the optimization:
    u_t = allen_cahn_optimizer.optimize(silent=False)

    # Save the last state equation from optimal temporal control:
    opt_y_T = allen_cahn_optimizer.solve_state_equation(u_t=u_t, 
                save_file=save_files, filename=f"Test{testnumber}_optimal_control")

    # Display Purposes:
    allen_cahn_optimizer.plot_gradient_norms()
    allen_cahn_optimizer.plot_objective_values()

    if construct_end_state:
        plot_u_d = fe.project(u_d, allen_cahn_optimizer.time_V)
        fe.plot(plot_u_d, label='u_d(t)')
        plt.title("Constructing Desired End State Control")
        plt.xlabel("Time t, [0, T = 2]")
        plt.ylabel("Temporal Control u_d(t)")
        plt.legend(title='Constructing Temporal control')
        plt.show()

    fe.plot(u_0, label='Initial')
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
    
    # Plot the constructed desired end state:
    g_plot = fe.plot(allen_cahn_optimizer.y_d)
    # set  colormap
    g_plot.set_cmap("viridis")
    # add a title to the  plot:
    plt.title("Desired y_d End State")
    # add a colorbar:
    plt.colorbar(g_plot)
    plt.show()

    # Plot the end state from inital control:
    g_plot = fe.plot(initial_y_T)
    # set  colormap
    g_plot.set_cmap("viridis")
    # add a title to the  plot:
    plt.title("Initial y_T End State")
    # add a colorbar:
    plt.colorbar(g_plot)
    plt.show()

    # Plot the optimized end state:
    g_plot = fe.plot(opt_y_T)
    # set  colormap
    g_plot.set_cmap("viridis")
    # add a title to the  plot:
    plt.title("Optimal y_opt_T End State")
    # add a colorbar:
    plt.colorbar(g_plot)
    plt.show()

    print("DONE!")



def constructed_problem_4(ndof=32, testnumber=323, save_files = False):
    construct_end_state = False
    # Rebound, near the optimal end-state without much us of 'u'.

    # Would like to lower the tolerance/End on gradient norm instead of 
    # absolute decrease in objective value.

    # Big Eps:
    eps = 25
    
    # Small Gamma:
    gamma = 0.001

    # The same as y_0:
    # square_spatial_control = IC.UnitSquareIslandIC(min_x=0.2, min_y=0.3, degree=3)
    d_min_x = d_min_y = 0.2
    orig_min_x = orig_min_y = 0.3
    # Almost the same as y_d, but can drive downward outside the desired "high" region:
    square_spatial_control = IC.CheckerIC(degree=3)

    # Want to Contract the initial square condition:
    y_0 = IC.RandomNoiseIC(degree=3)
    # y_0 = IC.UnitSquareQuadrantIC(degree=3)

    y_d = IC.CheckerIC(degree=3)

    T = 0.01
    time_steps = 25

    u_d = fe.Expression("1", T=T, degree=3)

    _, UnitSquare_V = IC.define_unit_square_mesh(ndof=ndof)

    init_dict = {
        "spatial_function_space": UnitSquare_V,
        "y_d": y_d,
        "y_0": y_0,
        "u_0": u_d,
        "spatial_control": square_spatial_control,
        "eps": eps,
        "gamma": gamma,
        "time_steps": time_steps,
        "T": T,
        "problem_name": "square_y_d",
        "optimizer_params" : [20, 0.0001, 20, 0.1, 1e-10]
        }

    allen_cahn_optimizer = AllenCahnOptimizer.from_dict(init_dict)
    allen_cahn_optimizer.state_equation.visualize_spatial_control()

    if construct_end_state:
        actual_y_d = allen_cahn_optimizer.solve_state_equation(save_file=False)

        # Reset the desired state, and the initial control:
        allen_cahn_optimizer.y_d = actual_y_d

        # Plot the constructed desired end state:
        g_plot = fe.plot(fe.project(actual_y_d, allen_cahn_optimizer.V))
        # set  colormap
        g_plot.set_cmap("viridis")
        # add a title to the  plot:
        plt.title("Naiive Desired y_d End State")
        # add a colorbar:
        plt.colorbar(g_plot)
        plt.show()
    
    #u_0 = fe.project(fe.Expression("x[0] < T/2.0 ? 1.0 : 0.0", T=T, degree=2), 
    u_0 = fe.project(fe.Expression("-10", T=T, degree=2), 
                     allen_cahn_optimizer.time_V)
    allen_cahn_optimizer.u_t = u_0.copy()

    # Run once to find the end state for the initial condition u_0:
    initial_y_T = allen_cahn_optimizer.solve_state_equation(u_t=u_0, 
                   save_file=save_files, filename=f"Test{testnumber}_initial_control") 

    # Plot the end state from inital control:
    g_plot = fe.plot(initial_y_T)
    # set  colormap
    g_plot.set_cmap("viridis")
    # add a title to the  plot:
    plt.title("Initial y_T End State")
    # add a colorbar:
    plt.colorbar(g_plot)
    plt.savefig(f'figs/testnb_{testnumber}_yT_init.pdf')
    plt.show()

    # Perform the optimization:
    u_t = allen_cahn_optimizer.optimize(silent=False)

    # Save the last state equation from optimal temporal control:
    opt_y_T = allen_cahn_optimizer.solve_state_equation(u_t=u_t, 
                save_file=save_files, filename=f"Test{testnumber}_optimal_control")

    # Display Purposes:
    allen_cahn_optimizer.plot_gradient_norms()
    allen_cahn_optimizer.plot_objective_values()

    if construct_end_state:
        plot_u_d = fe.project(u_d, allen_cahn_optimizer.time_V)
        fe.plot(plot_u_d, label='u_d(t)')
        plt.title("Constructing Desired End State Control")
        plt.xlabel("Time t, [0, T = 2]")
        plt.ylabel("Temporal Control u_d(t)")
        plt.legend(title='Constructing Temporal control')
        plt.savefig(f'figs/testnb_{testnumber}_u_construct.pdf')
        plt.show()

    fe.plot(u_0, label='Initial')
    plt.title("Intial Guess Control")
    plt.xlabel("Time t, [0, T = 2]")
    plt.ylabel("Temporal Control u(t)")
    plt.legend(title='Initial Guess Temporal control')
    plt.savefig(f'figs/testnb_{testnumber}_u0.pdf')
    plt.show()

    fe.plot(u_t, label='Optimal Control')
    plt.title("Final, Optimal, Control")
    plt.xlabel("Time t, [0, T = 2]")
    plt.ylabel("Temporal Control u(t)")
    plt.legend(title='Optimal Temporal control')
    plt.savefig(f'figs/testnb_{testnumber}_u_optimal.pdf')
    plt.show()
    
    # Plot the constructed desired end state:
    g_plot = fe.plot(allen_cahn_optimizer.y_d)
    # set  colormap
    g_plot.set_cmap("viridis")
    # add a title to the  plot:
    plt.title("Desired y_d End State")
    # add a colorbar:
    plt.colorbar(g_plot)
    plt.savefig(f'figs/testnb_{testnumber}_yd.pdf')
    plt.show()

    # Plot the optimized end state:
    g_plot = fe.plot(opt_y_T)
    # set  colormap
    g_plot.set_cmap("viridis")
    # add a title to the  plot:
    plt.title("Optimal y_opt_T End State")
    # add a colorbar:
    plt.colorbar(g_plot)
    plt.savefig(f'figs/testnb_{testnumber}_yT_optimal.pdf')
    plt.show()

    print("DONE!")

