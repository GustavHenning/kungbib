from .encoder import Encoder
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import json, glob, re, sys
from tqdm import tqdm

class BERT(Encoder):
    def __init__(self, dimensions, texts_location="/data/gustav/datalab_data/model/text", model_name='multi-qa-MiniLM-L6-cos-v1'): # TODO other models?
        self.dimensions = dimensions
        self.vector_location="/data/gustav/datalab_data/model/vectors"
        self.texts_location = texts_location
        self.original_model_name = model_name
        self.model_name = model_name
        model_name = model_name.replace('-PCA', '')
        print("Model name is {}".format(model_name))
        self.model = SentenceTransformer(model_name, device="cpu") # cpu because we are already using the cuda device :(
        self.cache = self.build_or_open_cache(model_name)
        if len(self.cache) == 0:
            self.all_text()

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
        if text in self.cache:
            return np.array(self.cache[text])
        if self.is_PCA:
            return self.text_to_pca[self.pattern.sub('', text)]
        else:
            vector = self.model.encode(text)[0:self.dimensions]
            self.cache[text] = vector.tolist()
            return vector # TODO original vector is 384 dimensions, how to compress into n?

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

    def build_or_open_cache(self, model_name):
        model_cache_path = self.vector_location + "/" + model_name
        from pathlib import Path
        Path(model_cache_path).mkdir(parents=True, exist_ok=True)        
        cache = Path(model_cache_path + "/cache.json")
        if not cache.is_file():
            return {}
        with open(cache, encoding='utf-8') as fh:
            try: 
                data = json.load(fh)
                return data
            except ValueError:
                print("json was malformatted...")
                return {}

    def dump_cache(self):
        model_cache_path = self.vector_location + "/" + self.model_name
        print("cache dumped with {} keys".format(len(self.cache)))
        with open(model_cache_path + "/cache.json", 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=4)
        sys.exit()

    def all_text(self):
        files = glob.glob(self.texts_location + "/*")
        print("building corpus")
        for file in tqdm(files): 
            with open(file, encoding='utf-8') as fh:
                data = json.load(fh)
                for d in data["content"]:
                    self.cache[d['text']] = self.model.encode(d['text'])[0:self.dimensions].tolist()
        self.dump_cache()
