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
