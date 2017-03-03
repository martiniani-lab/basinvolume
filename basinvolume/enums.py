from enum import Enum, unique  # Package enum34


@unique
class Minimizer(Enum):
    FIRE = 1
    CG = 2
    LBFGS = 3
