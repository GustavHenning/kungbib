from abc import ABC, abstractmethod

class Encoder(ABC):
    
    """
    Should return an n-dimensional array of vectors based on the content of the text
    """
    @abstractmethod
    def encode(self, text):
        pass

