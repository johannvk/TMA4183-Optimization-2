import fenics as fe
import numpy as np
import matplotlib.pyplot as plt

import problem_definitions as prob


def main():
    fe.set_log_active(False)

    # prob.constructed_problem_1()

    # prob.constructed_problem_2()

    # prob.constructed_problem_3(testnumber=8, save_files=True)

    prob.constructed_problem_4() 
    

if __name__ == "__main__":
    main()
