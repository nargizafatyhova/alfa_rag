import os
import re
import time
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import pandas as pd
import razdel

NA = "Нет ответа."
GATE = float(os.environ.get("GATE", "0.05"))
CTX = int(os.environ.get("CTX_CHUNKS", "5"))
MAXNEW = int(os.environ.get("MAX_NEW", "320"))
MODEL = os.environ.get("LLM", "Qwen/Qwen2.5-14B-Instruct")
TP = int(os.environ.get("TP", "2"))
MAXLEN = int(os.environ.get("MAX_LEN", "8192"))
BUDGET = MAXLEN - MAXNEW - 48

sys_msg = ("Ты — ассистент поддержки Альфа-Банка. Отвечай на вопрос пользователя строго по "
           "приведённым фрагментам базы знаний.\n"
           "- Используй только факты из фрагментов, ничего не выдумывай.\n"
           "- Дай полный связный ответ, покрывающий все относящиеся к вопросу детали из фрагментов "
           "(условия, шаги, цифры, способы), но без воды и повторов.\n"
           "- Нейтрально-деловой стиль, как в справке банка.\n"
           "- Если во фрагментах нет информации для ответа — ответь ровно: «Нет ответа.»")
usr = "Фрагменты базы знаний:\n{ctx}\n\nВопрос: {q}\n\nОтвет:"


def nw(s):
    return len(re.findall(r"\w+", s))


def tgt_len(q):
    return int(min(max(45 + 4 * min(nw(q), 20), 25), 140))


def calib(a, tw):
    a = a.strip()
    if a.lower().startswith("нет ответа"):
        return NA
    cap = int(tw * 1.45)
    out, n = [], 0
    for s in (x.text.strip() for x in razdel.sentenize(a) if x.text.strip()):
        w = nw(s)
        if out and n + w > cap:
            break
        out.append(s)
        n += w
    return " ".join(out) if out else a


def main():
    t0 = time.time()
    rk = pd.read_parquet("reranked.parquet")
    q = pd.read_csv("questions.csv")
    g = rk.sort_values(["q_id", "ce_rank"]).groupby("q_id")
    ctx, mce = {}, {}
    for qid, gg in g:
        h = gg.head(CTX)
        ctx[qid] = list(zip(h["title"], h["raw"]))
        mce[qid] = float(gg["ce_score"].max())

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    llm = LLM(model=MODEL, tensor_parallel_size=TP, dtype="float16", max_model_len=MAXLEN,
              gpu_memory_utilization=0.92, trust_remote_code=True, enforce_eager=True)
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=MAXNEW, repetition_penalty=1.05)

    qm = dict(zip(q["q_id"], q["query"]))

    # контекст режем по бюджету токенов, чтобы промпт точно влез в max_model_len
    def prompt(qid):
        cc = ctx[qid]
        for n in range(len(cc), 0, -1):
            blocks = "\n\n".join(f"[Фрагмент {k+1}] {(t+'. ' if t else '')}{r}"
                                 for k, (t, r) in enumerate(cc[:n]))
            msgs = [{"role": "system", "content": sys_msg},
                    {"role": "user", "content": usr.format(ctx=blocks, q=qm[qid])}]
            p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            if len(tok(p)["input_ids"]) <= BUDGET:
                return p
        t, r = cc[0]
        r = tok.decode(tok(r)["input_ids"][:BUDGET - 256])
        msgs = [{"role": "system", "content": sys_msg},
                {"role": "user", "content": usr.format(ctx=f"[Фрагмент 1] {r}", q=qm[qid])}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    ids, prompts = [], []
    for qid in q["q_id"]:
        if mce.get(qid, -1) < GATE:
            continue
        prompts.append(prompt(qid))
        ids.append(qid)

    outs = llm.generate(prompts, sp)
    ans = {qid: o.outputs[0].text.strip() for qid, o in zip(ids, outs)}

    rows = []
    for qid in q["q_id"]:
        if qid in ans:
            a = calib(ans[qid], tgt_len(qm[qid])) or NA
        else:
            a = NA
        rows.append({"q_id": qid, "answer_new": a})
    sub = pd.DataFrame(rows)
    sub.to_csv("submission.csv", index=False)
    na = sub["answer_new"].str.startswith("Нет ответа").sum()
    print("done %.0fs, NA=%d (%.1f%%)" % (time.time() - t0, na, 100 * na / len(sub)))


if __name__ == "__main__":
    main()
