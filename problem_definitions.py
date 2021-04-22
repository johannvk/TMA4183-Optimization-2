import fenics as fe
import random

# Storage of problem definitions:
# 
# {
#  'mesh': fe.Mesh,  
#  'initial_conditions': fe.UserExpression, 
#  'y_desired': fe.Expression,
#  'spatial_control': fe.Expression, 
#  'y_initial': fe.Expression,
# }

def define_unit_square_mesh(ndof=64):
    poly_degree = 2
    mesh = fe.UnitSquareMesh(ndof, ndof)

    V = fe.FunctionSpace(mesh, 'Lagrange', poly_degree)
    return (mesh, V)


class UnitSquareIslandIC(fe.UserExpression):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def eval(self, values, x):
        # Homogeneous Neumann Conditions must be respected!
        if 0.4 <= x[0] <= 0.6 and 0.4 <= x[1] <= 0.6:
            values[0] = 1.0
        else:
            values[0] = 0.0

    def value_shape(self):
        return []


class RandomNoiseIC(fe.UserExpression):
    def __init__(self, **kwargs):
        random.seed(2)
        super().__init__(**kwargs)

    def eval(self, values, x):
        # Homogeneous Neumann Conditions must be respected!
        # Rock the boat between [-1.0, 1.0]:
        value = 0.63 + 0.25*(0.5 - random.random())
        values[0] = max(min(value, 1.0), -1.0)

    def value_shape(self):
        return []


class RandomUnitSquareQuadrantIC(fe.UserExpression):

    def __init__(self, **kwargs):
        random.seed(2)
        super().__init__(**kwargs)

    def eval(self, values, pos):
        # Four quadrants:
        #  Lower left:  ([0.0, 0.0] to [0.5, 0.5])
        #  Lower right: ([0.5, 0.0] to [1.0, 0.5])
        #  Upper left:  ([0.0, 0.5] to [0.5, 1.0])
        #  Upper right: ([0.5, 0.5] to [1.0, 1.0])
        low_level = 0.4
        high_level = 0.6

        x, y = pos

        in_lower_left = (0.0 <= x <= 0.5) and (0.0 <= y <= 0.5)
        in_lower_right = (0.5 <= x <= 1.0) and (0.0 <= y <= 0.5)

        in_upper_left = (0.0 <= x <= 0.5) and (0.5 <= y <= 1.0)
        in_upper_right = (0.5 <= x <= 1.0) and (0.5 <= y <= 1.0)

        nudge = 0.25*(0.5 - random.random())

        if in_lower_left or in_upper_right:
            values[0] =  low_level + nudge
        else:
            values[0] =  high_level + nudge

    def value_shape(self):
        return []