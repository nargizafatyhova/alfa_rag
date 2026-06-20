import os
import time
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import pandas as pd

sys_msg = ("Ты — эксперт поддержки Альфа-Банка. Дай максимально полный и подробный ответ на вопрос, "
           "опираясь на фрагменты базы знаний и свои знания о продуктах Альфа-Банка. Перечисли все "
           "относящиеся к теме детали: условия, способы, шаги, сроки, суммы, проценты, комиссии, "
           "ограничения, названия продуктов и сервисов. Пиши плотно, конкретными фактами, без "
           "вводных фраз и воды. Не пиши «нет информации».")
usr = "Фрагменты:\n{ctx}\n\nВопрос: {q}\n\nПодробный ответ:"


def main():
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    name = "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4"
    q = pd.read_csv("questions.csv")
    qm = dict(zip(q["q_id"], q["query"]))
    rk = pd.read_parquet("reranked2.parquet").sort_values(["q_id", "ce_rank"])
    ctx = rk[rk.ce_rank < 6].groupby("q_id")["raw"].apply(lambda s: list(s.astype(str))).to_dict()

    tok = AutoTokenizer.from_pretrained(name)
    llm = LLM(model=name, tensor_parallel_size=2, dtype="float16", quantization="gptq",
              max_model_len=8192, gpu_memory_utilization=0.92, enforce_eager=True, trust_remote_code=True)
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=380, repetition_penalty=1.05)

    ids = list(q["q_id"])
    prompts = []
    for i in ids:
        blocks = "\n\n".join(f"[{k+1}] {c}" for k, c in enumerate(ctx.get(i, [])[:6]))
        msgs = [{"role": "system", "content": sys_msg},
                {"role": "user", "content": usr.format(ctx=blocks, q=qm[i])}]
        prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

    t0 = time.time()
    outs = llm.generate(prompts, sp)
    ans = [o.outputs[0].text.strip() for o in outs]
    pd.DataFrame({"q_id": ids, "answer_new": ans}).to_csv("answers_32b.csv", index=False)
    print("done in %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
