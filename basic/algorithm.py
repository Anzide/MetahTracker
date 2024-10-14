from abc import abstractmethod, ABC

from basic.landscape import DimSeparatedLandscape


class Algorithm(ABC):
    """
    Not a very useful class, but it is here to show that the algorithm should have a landscape to work on.
    """
    def __init__(self, landscape: DimSeparatedLandscape):
        self.landscape = landscape

    @abstractmethod
    def run(self):
        pass
