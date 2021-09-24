from .encoder import Encoder
import torch
from sentence_transformers import SentenceTransformer

class BERT(Encoder):
    def __init__(self, dimensions, model_name='all-MiniLM-L6-v2'):
        self.dimensions = dimensions
        self.model = SentenceTransformer(model_name, device="cpu") # cpu because we are already using the cuda device :(
    """
    Returns an (1,n)-dimensional array of vectorzied text where n is the dimensions passed to the constructor
    """
    def encode(self, text):
        return self.model.encode(text)[0:self.dimensions] # TODO original vector is 384 dimensions, how to compress into n?