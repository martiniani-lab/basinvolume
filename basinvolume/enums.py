from enum import Enum, unique  # Package enum34


@unique
class Minimizer(Enum):
    FIRE = 1
    CG = 2
    LBFGS = 3
    CVODE = 4
    MXD = 5  # Mixed Descent


@unique
class Interaction(Enum):
    HS_WCA = 1
    INVERSE_POWER_STILLINGER = 2
    INVERSE_POWER = 3
    NEGATIVE_COS = 4
    INVERSE_POWER_HS = 5
