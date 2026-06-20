import os
import time
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import numpy as np
import pandas as pd

TOPN = 30


def main():
    from FlagEmbedding import FlagReranker
    t0 = time.time()
    d = pd.read_parquet("candidates.parquet")
    d = d[d["rank"] < TOPN].copy()

    rr = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True, devices=["cuda:0"])
    pairs = [[q, (t + ". " + r) if isinstance(t, str) and t else r]
             for q, t, r in zip(d["query"], d["title"], d["raw"])]
    sc = rr.compute_score(pairs, batch_size=256, normalize=True)
    d["ce_score"] = np.asarray(sc, dtype=np.float32)
    d["ce_rank"] = d.groupby("q_id")["ce_score"].rank(ascending=False, method="first") - 1
    d = d.sort_values(["q_id", "ce_rank"]).reset_index(drop=True)
    d.to_parquet("reranked.parquet", index=False)
    print("done in %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
