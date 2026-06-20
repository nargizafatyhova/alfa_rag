import math
import numpy as np
import torch
from collections import Counter
from transformers import AutoTokenizer, AutoModel


def lmul(la, lr):
    # штраф за длину из условия задачи
    if lr <= 0:
        return 1.0 if la == 0 else 0.0
    r = la / lr
    if r <= 1.5:
        return 1.0
    if r >= 3.0:
        return 0.0
    return -2.0 / 3.0 * r + 2.0


class BertScorer:
    def __init__(self, model_name="cointegrated/rubert-tiny2", device=None, max_len=256, bs=64):
        self.dev = device or ("cuda" if torch.cuda.is_available() else
                              ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.m = AutoModel.from_pretrained(model_name).to(self.dev).eval()
        self.max_len = max_len
        self.bs = bs

    def ntok(self, s):
        return len(self.tok.encode(str(s), add_special_tokens=False, truncation=True,
                                   max_length=self.max_len * 4))

    @torch.no_grad()
    def embed(self, texts):
        out = []
        for i in range(0, len(texts), self.bs):
            batch = [str(t) if str(t).strip() else "." for t in texts[i:i + self.bs]]
            enc = self.tok(batch, return_tensors="pt", padding=True, truncation=True,
                           max_length=self.max_len).to(self.dev)
            hs = torch.nn.functional.normalize(self.m(**enc).last_hidden_state, dim=-1)
            mask = enc["attention_mask"].bool()
            ids = enc["input_ids"]
            for b in range(hs.size(0)):
                mm = mask[b].clone()
                for sid in self.tok.all_special_ids:
                    mm &= ids[b] != sid
                out.append(hs[b][mm].cpu())
        return out

    def idf(self, refs):
        n = len(refs)
        df = Counter()
        ids = []
        for r in refs:
            e = self.tok.encode(str(r), add_special_tokens=False, truncation=True, max_length=self.max_len)
            ids.append(e)
            for t in set(e):
                df[t] += 1
        return {t: math.log((n + 1) / (c + 1)) + 1 for t, c in df.items()}, ids

    @torch.no_grad()
    def recall(self, refs, cands, use_idf=False):
        re_, ce_ = self.embed(refs), self.embed(cands)
        idf, rid = (self.idf(refs) if use_idf else (None, None))
        sc = np.zeros(len(refs), dtype=np.float64)
        for i, (a, b) in enumerate(zip(re_, ce_)):
            if a.shape[0] == 0:
                sc[i] = 1.0 if b.shape[0] == 0 else 0.0
                continue
            if b.shape[0] == 0:
                sc[i] = 0.0
                continue
            ms = (a @ b.T).max(dim=1).values.numpy()  # для каждого токена эталона лучший токен ответа
            if use_idf:
                w = np.array([idf.get(t, 1.0) for t in rid[i][:len(ms)]], dtype=np.float64)
                if len(w) < len(ms):
                    w = np.pad(w, (0, len(ms) - len(w)), constant_values=1.0)
                sc[i] = float((ms * w).sum() / w.sum())
            else:
                sc[i] = float(ms.mean())
        return sc

    def recall_L(self, refs, cands, use_idf=False):
        r = self.recall(refs, cands, use_idf=use_idf)
        lr = np.array([self.ntok(x) for x in refs], dtype=np.float64)
        la = np.array([self.ntok(x) for x in cands], dtype=np.float64)
        L = np.array([lmul(a, b) for a, b in zip(la, lr)])
        return r * L, r, L, lr, la


if __name__ == "__main__":
    sc = BertScorer()
    refs = ["Номер счёта состоит из 20 цифр.", "Реквизиты можно посмотреть в приложении."]
    cands = ["Счёт содержит двадцать знаков.", "Нет ответа."]
    rl, r, L, lr, la = sc.recall_L(refs, cands)
    for i in range(len(refs)):
        print("recall=%.3f L=%.2f RL=%.3f" % (r[i], L[i], rl[i]))
