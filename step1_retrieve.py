import os
import time
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import numpy as np
import pandas as pd

TOPN = 50


def rrf(lists, k=60):
    sc = {}
    for ranks in lists:
        for pos, i in enumerate(ranks):
            sc[i] = sc.get(i, 0.0) + 1.0 / (k + pos + 1)
    return sorted(sc, key=sc.get, reverse=True)


def main():
    from preprocess import build_chunks
    from retrieval import lemmatize
    from rank_bm25 import BM25Okapi
    from FlagEmbedding import BGEM3FlagModel

    t0 = time.time()
    ch = build_chunks("websites.csv", "chunks.parquet").reset_index(drop=True)
    q = pd.read_csv("questions.csv")
    qs = q["query"].astype(str).tolist()

    m = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, devices=["cuda:0"])
    de = np.asarray(m.encode(ch["text"].tolist(), batch_size=64, max_length=512)["dense_vecs"],
                    dtype=np.float32)

    bm = BM25Okapi([lemmatize(t) for t in ch["text"].tolist()])

    qe = np.asarray(m.encode(qs, batch_size=64, max_length=128)["dense_vecs"], dtype=np.float32)
    sims = qe @ de.T

    rows = []
    for i, qq in enumerate(qs):
        d = sims[i]
        b = np.asarray(bm.get_scores(lemmatize(qq)))
        fused = rrf([np.argsort(-d)[:200], np.argsort(-b)[:200]])[:TOPN]
        for rank, j in enumerate(fused):
            c = ch.iloc[j]
            rows.append({"q_id": q["q_id"].iloc[i], "query": qq, "rank": rank,
                         "chunk_idx": int(j), "title": c["title"], "raw": c["raw"],
                         "dense": float(d[j])})
    pd.DataFrame(rows).to_parquet("candidates.parquet", index=False)
    print("done %d cand in %.0fs" % (len(rows), time.time() - t0))


if __name__ == "__main__":
    main()
