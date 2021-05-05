import fenics as fe
import random

def define_unit_square_mesh(ndof=64):
    poly_degree = 2
    mesh = fe.UnitSquareMesh(ndof, ndof)

    V = fe.FunctionSpace(mesh, 'Lagrange', poly_degree)
    return (mesh, V)


class UnitSquareIslandIC(fe.UserExpression):
    def __init__(self, min_x=0.4, min_y=0.4, 
                 high_level=0.99, low_level=-0.01, **kwargs):
        self.min_x = min_x
        self.min_y = min_y

        self.max_x = 1.0 - min_x
        self.max_y = 1.0 - min_y

        self.high_level = high_level
        self.low_level = low_level

        super().__init__(**kwargs)

    def eval(self, values, x):
        # Homogeneous Neumann Conditions must be respected!
        if self.min_x <= x[0] <= self.max_x and \
           self.min_y <= x[1] <= self.max_y:
            values[0] = self.high_level
        else:
            values[0] = self.low_level

    def value_shape(self):
        return []

class CheckerIC(fe.UserExpression):
    def __init__(high_level=0.99, low_level=-0.99, **kwargs):
        #self.high_level = high_level
        #self.low_level = low_level

        super().__init__(**kwargs)

    def eval(self, values, x):
        b = 2
        if ((b*x[0])//1)%2 == ((b*x[1])//1)%2:
            values[0] = -1#self.high_level
        else:
            values[0] = 1#self.low_level
        # Homogeneous Neumann Conditions must be respected!
        if x[0]<0.1 or x[0]>0.9 or x[1]<0.1 or x[1]>0.9:
            values[0] = 1

    def value_shape(self):
        return []


class RandomNoiseIC(fe.UserExpression):
    def __init__(self, **kwargs):
        random.seed(2)
        super().__init__(**kwargs)

    def eval(self, values, x):
        # Homogeneous Neumann Conditions must be respected!
        # Rock the boat between [-1.0, 1.0]:
        #value = 0.63 + 0.25*(0.5 - random.random())
        value = 2*(0.5 - random.random())
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


class UnitSquareQuadrantIC(fe.UserExpression):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def eval(self, values, pos):
        # Four quadrants:
        #  Lower left:  ([0.0, 0.0] to [0.5, 0.5])
        #  Lower right: ([0.5, 0.0] to [1.0, 0.5])
        #  Upper left:  ([0.0, 0.5] to [0.5, 1.0])
        #  Upper right: ([0.5, 0.5] to [1.0, 1.0])
        low_level = 0.4
        high_level = 0.61

        x, y = pos

        in_lower_left = (0.0 <= x <= 0.5) and (0.0 <= y <= 0.5)
        in_lower_right = (0.5 <= x <= 1.0) and (0.0 <= y <= 0.5)

        in_upper_left = (0.0 <= x <= 0.5) and (0.5 <= y <= 1.0)
        in_upper_right = (0.5 <= x <= 1.0) and (0.5 <= y <= 1.0)

        if in_lower_left or in_upper_right:
            values[0] =  low_level
        else:
            values[0] =  high_level

    def value_shape(self):
        return []
