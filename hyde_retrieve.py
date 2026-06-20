import os
import time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import pandas as pd


def na(s):
    return str(s).strip().startswith("Нет ответа")


def main():
    from FlagEmbedding import BGEM3FlagModel, FlagReranker
    t0 = time.time()
    ch = pd.read_parquet("chunks.parquet").reset_index(drop=True)
    q = pd.read_csv("questions.csv")
    # ответы первого прохода (step3) используем как гипотетический документ для поиска
    fp = pd.read_csv("submission.csv").set_index("q_id")
    qm = dict(zip(q["q_id"], q["query"]))

    hyde = []
    for qid in q["q_id"]:
        a = str(fp.loc[qid, "answer_new"]) if qid in fp.index else ""
        a = "" if na(a) else a
        hyde.append((qm[qid] + " " + a).strip())

    m = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, devices=["cuda:0"])
    doc = np.asarray(m.encode(ch["text"].tolist(), batch_size=96, max_length=512)["dense_vecs"],
                     dtype=np.float32)
    qe = np.asarray(m.encode(hyde, batch_size=96, max_length=300)["dense_vecs"], dtype=np.float32)
    top = np.argsort(-(qe @ doc.T), axis=1)[:, :60]

    rr = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True, devices=["cuda:0"])
    rows = []
    for i, qid in enumerate(q["q_id"]):
        cand = top[i]
        pairs = [[qm[qid], ch.iloc[j]["text"]] for j in cand]
        sc = np.asarray(rr.compute_score(pairs, batch_size=128, normalize=True), dtype=np.float32)
        for rank, o in enumerate(np.argsort(-sc)[:8]):
            j = int(cand[o])
            rows.append({"q_id": qid, "ce_rank": rank, "ce_score": float(sc[o]),
                         "raw": ch.iloc[j]["raw"]})
    pd.DataFrame(rows).to_parquet("reranked2.parquet", index=False)
    print("done in %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
