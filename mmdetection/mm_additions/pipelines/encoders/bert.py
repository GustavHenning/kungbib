from .encoder import Encoder
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import json, glob, re
from tqdm import tqdm

class BERT(Encoder):
    def __init__(self, dimensions, texts_location="/data/gustav/datalab_data/poly-dn-2010-2020-729/text", model_name='multi-qa-MiniLM-L6-cos-v1'): # TODO other models?
        self.dimensions = dimensions
        self.original_model_name = model_name
        model_name = model_name.replace('-PCA', '')
        print("Model name is {}".format(model_name))
        self.model = SentenceTransformer(model_name, device="cpu") # cpu because we are already using the cuda device :(

        if self.original_model_name.endswith('-PCA'):
            self.is_PCA = True
            if self.text_to_pca is None:
                self.build_PCA(texts_location)
        else:
            self.is_PCA = False
        
    """
    Returns an (1,n)-dimensional array of vectorized text where n is the dimensions passed to the constructor
    """
    def encode(self, text):
        if self.is_PCA:
            return self.text_to_pca[self.pattern.sub('', text)]
        else:
            return self.model.encode(text)[0:self.dimensions] # TODO original vector is 384 dimensions, how to compress into n?

    def build_PCA(self, texts_location):
        print("Building PCA")
        self.pattern = re.compile('[^\w\d ]+')
        sentences = self.build_corpus(texts_location)
        print("Encoding embeddings")
        embeddings = [self.model.encode(sentence) for sentence in tqdm(sentences)]
        df = pd.DataFrame(embeddings)
        X = df.values
        print("Scaling embeddings")
        scaler = StandardScaler()
        scaler.fit(X)
        X_scaled = scaler.transform(X)
        print("Creating PCA")
        pca_95 = PCA(n_components=0.95, random_state=2021)
        pca_95.fit(X_scaled)
        X_pca_95 = pca_95.transform(X_scaled)
        print("shapes:")
        print(np.shape(sentences))
        print(X_pca_95.shape) # 366 dimensions TODO create a copy of all the configs for 366 dimensions
        self.text_to_pca = dict(zip(sentences, X_pca_95))


    def build_corpus(self, texts_location):
        sentences = []
        files = glob.glob(texts_location + "/*")
        print("building corpus")
        for file in tqdm(files): 
            with open(file, encoding='utf-8') as fh:
                data = json.load(fh)
                for d in data["content"]:
                    res = self.pattern.sub('', d["text"])
                    sentences.append(res)
        return sentences
