import yavs

import os
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

fname = "db.yavs"
dimension = 384
if not os.path.exists(fname):
    yavs.create(fname, dimension)

text = "My first record"
embedding = model.encode(text)
yavs.append(fname, embedding, text)

text = "Another piece of data"
embedding = model.encode(text)
yavs.append(fname, embedding, text)

