import re
import pickle
import numpy as np
import pandas as pd
import razdel
import pymorphy3
from functools import lru_cache
from rank_bm25 import BM25Okapi

wre = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)
morph = pymorphy3.MorphAnalyzer()

stop = set("""и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по
только ее мне было вот от меня еще нет о из ему теперь когда даже ну вдруг ли если уже или ни
быть был него до вас нибудь опять уж вам ведь там потом себя ничего ей может они тут где есть надо
ней для мы тебя их чем была сам чтоб без будто чего раз тоже себе под будет ж тогда кто этот того
потому этого какой совсем ним здесь этом один почти мой тем чтобы нее сейчас были куда зачем всех
никогда можно при наконец два об другой хоть после над больше тот через эти нас про всего них какая
много разве три эту моя впрочем хорошо свою этой перед иногда лучше чуть том нельзя такой им более
всегда конечно всю между это""".split())


@lru_cache(maxsize=300_000)
def norm(t):
    return morph.parse(t)[0].normal_form


def lemmatize(text):
    out = []
    for t in razdel.tokenize(str(text)):
        w = t.text.lower()
        if not wre.fullmatch(w) or w in stop or len(w) < 2:
            continue
        out.append(norm(w))
    return out


class HybridRetriever:
    def __init__(self, dense_model="intfloat/multilingual-e5-small", e5_prefix=True, device=None):
        self.dm = dense_model
        self.pref = e5_prefix
        self.device = device
        self.ch = None
        self.bm = None
        self.dense = None
        self.de = None

    def build(self, ch, emb_path=None):
        from sentence_transformers import SentenceTransformer
        self.ch = ch.reset_index(drop=True)
        txt = self.ch["text"].tolist()
        self.bm = BM25Okapi([lemmatize(t) for t in txt])
        self.dense = SentenceTransformer(self.dm, device=self.device)
        pas = [("passage: " + t) if self.pref else t for t in txt]
        self.de = self.dense.encode(pas, batch_size=128, show_progress_bar=True,
                                    normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
        if emb_path:
            np.save(emb_path, self.de)
        return self

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"ch": self.ch, "bm": self.bm, "de": self.de,
                         "dm": self.dm, "pref": self.pref}, f)

    @classmethod
    def load(cls, path, device=None):
        from sentence_transformers import SentenceTransformer
        with open(path, "rb") as f:
            d = pickle.load(f)
        r = cls(dense_model=d["dm"], e5_prefix=d["pref"], device=device)
        r.ch, r.bm, r.de = d["ch"], d["bm"], d["de"]
        r.dense = SentenceTransformer(r.dm, device=device)
        return r

    def bm25(self, q):
        return np.asarray(self.bm.get_scores(lemmatize(q)), dtype=np.float32)

    def dense_scores(self, qs):
        x = [("query: " + s) if self.pref else s for s in qs]
        qe = self.dense.encode(x, batch_size=128, normalize_embeddings=True,
                               convert_to_numpy=True).astype(np.float32)
        return qe @ self.de.T

    @staticmethod
    def rrf(lists, k=60):
        sc = {}
        for ranks in lists:
            for pos, i in enumerate(ranks):
                sc[i] = sc.get(i, 0.0) + 1.0 / (k + pos + 1)
        return sorted(sc, key=sc.get, reverse=True)

    def search_batch(self, qs, topk=50, rrf_k=60, pool=200):
        ds = self.dense_scores(qs)
        res = []
        for i, q in enumerate(qs):
            d = ds[i]
            b = self.bm25(q)
            dr = np.argsort(-d)[:pool]
            br = np.argsort(-b)[:pool]
            fused = self.rrf([dr, br], k=rrf_k)[:topk]
            res.append([(int(j), float(d[j]), float(b[j])) for j in fused])
        return res


if __name__ == "__main__":
    import os, time
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    ch = pd.read_parquet("chunks.parquet")
    t = time.time()
    r = HybridRetriever().build(ch, emb_path="doc_emb.npy")
    r.save("retriever.pkl")
    print("indexed in %.0fs" % (time.time() - t))
