import re
import pandas as pd

wre = re.compile(r"\w+", re.UNICODE)

junk = {
    "сервисы", "курсы валют", "больше валют", "получайте больше с альфа-банком",
    "рассчитайте выгоду", "заявка онлайн", "подробнее", "оформить", "узнать больше",
    "наверх", "меню", "войти", "регистрация", "поиск",
}


def ntok(s):
    return len(wre.findall(s or ""))


def cl(line):
    line = line.replace("\x0c", " ").strip()
    return re.sub(r"[ \t]+", " ", line)


def skip(line):
    low = line.lower().strip()
    if not low or low in junk:
        return True
    return bool(re.fullmatch(r"[\d\s.,]+\s*(кб|мб|байт)?", low))


def cut(line, mx):
    if ntok(line) <= mx:
        return [line]
    parts = re.split(r"(?<=[.!?])\s+", line)
    out, buf, n = [], [], 0
    for p in parts:
        k = ntok(p)
        if buf and n + k > mx:
            out.append(" ".join(buf))
            buf, n = [], 0
        buf.append(p)
        n += k
    if buf:
        out.append(" ".join(buf))
    return out


def split_doc(text, tgt=320, ov=64, mx=420):
    lines = []
    for raw in str(text).split("\n"):
        line = cl(raw)
        if skip(line):
            continue
        lines.extend(cut(line, mx))

    res, buf, n = [], [], 0
    for line in lines:
        k = ntok(line)
        if buf and n + k > tgt:
            res.append("\n".join(buf))
            # хвост предыдущего чанка тащим в начало следующего, чтобы был overlap
            tail, tn = [], 0
            for prev in reversed(buf):
                pn = ntok(prev)
                if tn + pn > ov:
                    break
                tail.insert(0, prev)
                tn += pn
            buf, n = tail[:], tn
        buf.append(line)
        n += k
    if buf:
        res.append("\n".join(buf))
    return [c for c in res if ntok(c) >= 4]


def build_chunks(src, out, tgt=320, ov=64, mx=420):
    df = pd.read_csv(src).drop_duplicates(subset=["text"]).reset_index(drop=True)
    rows = []
    for r in df.itertuples(index=False):
        title = "" if pd.isna(r.title) else str(r.title).strip()
        for j, c in enumerate(split_doc(r.text, tgt, ov, mx)):
            rows.append({
                "web_id": r.web_id,
                "url": r.url,
                "title": title,
                "chunk_id": f"{r.web_id}_{j}",
                "text": (title + "\n" + c) if title else c,
                "raw": c,
                "n_tok": ntok(c),
            })
    ch = pd.DataFrame(rows)
    ch.to_parquet(out, index=False)
    return ch


if __name__ == "__main__":
    import sys
    a = sys.argv[1] if len(sys.argv) > 1 else "../websites.csv"
    b = sys.argv[2] if len(sys.argv) > 2 else "chunks.parquet"
    ch = build_chunks(a, b)
    print(len(ch), "chunks from", ch.web_id.nunique(), "docs")
