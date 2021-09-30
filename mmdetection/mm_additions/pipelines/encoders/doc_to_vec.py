from .encoder import Encoder

from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec as D2V, TaggedDocument
import json
import glob
import re, string
from tqdm import tqdm

class Doc2Vec(Encoder):
    def __init__(self, dimensions, texts_location="/data/gustav/datalab_data/poly-dn-2010-2020-720/text", model_name=""):
        print("initializing Doc2Vec")
        self.pattern = re.compile('[^\w\d ]+')
        sentences = self.build_corpus(texts_location)
        documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(sentences)]
        self.model = D2V(documents, vector_size=dimensions, window=2, min_count=1, workers=4)

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

    def encode(self, text):
        res = self.pattern.sub('', text)
        return self.model.infer_vector(res.split(" "))