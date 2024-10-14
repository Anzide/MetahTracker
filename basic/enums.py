from enum import Enum


class OptimizationType(Enum):
    """
    The type of extrema you wish to find on the landscape - maximum or minimum.
    """
    MAXIMISATION = 'max'
    MINIMISATION = 'min'


class MoveType(Enum):
    """
    True Exploration: local optimum: target better than base, fitness: target better than base
    False Exploration: local optimum: target worse than base, fitness: target better than base
    True Rejection: local optimum: target worse than base, fitness: target worse than base
    False Rejection: local optimum: target better than base, fitness: target worse than base
    Exploitation: target and base are in the same attraction basin
    """
    TRUE_EXPL = 'true_exploration'
    FALSE_EXPL = 'false_exploration'
    TRUE_REJ = 'true_rejection'
    FALSE_REJ = 'false_rejection'
    EXPLOITATION = 'exploitation'

exploration_color_map = {
    MoveType.TRUE_EXPL: 'green',
    MoveType.FALSE_EXPL: 'white',
    MoveType.TRUE_REJ: 'black',
    MoveType.FALSE_REJ: 'red',
    MoveType.EXPLOITATION: 'grey'
}


class OptimumComparison(Enum):
    """
    Comparison the two local optimum to target and base.
    BETTER: Target is better than base.
    WORSE: Target is worse than base.
    SAME: Target and base are in the same attraction basin.
    """
    BETTER = 'better'
    WORSE = 'worse'
    SAME = 'same'
