import codecs
import re
import os
from typing import Callable
import pandas as pd
from pathlib import Path


def load_pres(fname):
    """Load data (on presidents)"""

    alltxts = []
    alllabs = []
    s = codecs.open(fname, 'r', 'utf-8')  # pour régler le codage
    while True:
        txt = s.readline()
        if (len(txt)) < 5:
            break
        #
        lab = re.sub(r"<[0-9]*:[0-9]*:(.)>.*", "\\1", txt)
        txt = re.sub(r"<[0-9]*:[0-9]*:.>(.*)", "\\1", txt)
        if lab.count('M') > 0:
            alllabs.append(-1)
        else:
            alllabs.append(1)
        alltxts.append(txt)
    return alltxts, alllabs


def load_movies(path2data):  # 1 classe par répertoire
    alltxts = []  # init vide
    labs = []
    cpt = 0
    for cl in os.listdir(path2data):  # parcours des fichiers d'un répertoire
        for f in os.listdir(path2data+cl):
            txt = open(path2data+cl+'/'+f).read()
            alltxts.append(txt)
            labs.append(cpt)
        cpt += 1  # chg répertoire = cht classe

    return alltxts, labs


def clean_dataset(load_func: Callable, data_path: str, out_path: str, preprocessor):
    texts, classes = load_func(data_path)
    text_preprocessor = preprocessor.fit(texts)
    texts_cleaned = text_preprocessor.transform(texts)

    df = pd.DataFrame({"text": texts_cleaned, "label": classes})
    df.to_parquet(Path(out_path).with_suffix(".parquet"), index=False)


def load_clean_data(path):
    df = pd.read_parquet(path)
    return df["text"].tolist(), df["label"].to_numpy()
