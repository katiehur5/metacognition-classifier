#!/usr/bin/env python3
"""
Decode the think-aloud writing logs produced by qualtrics_logger.js (format v1 and v2).

Input : a Qualtrics CSV export (Data & Analysis -> Export & Import -> Export data -> CSV).
        Do NOT open/re-save the export in Excel first.
Output: <out>/events.csv                        one row per raw event, with replayed text
        <out>/timeline_<ResponseId>_<prompt>.md  rendered, LLM-friendly timeline
        a validation report on stdout (one line per response x prompt)

Usage:
  python3 src/decode_kl.py data/kl/<input>.csv --text-col p1=Q1 --text-col p2=Q2 --text-col p3=Q3 --text-col p4=Q4 --text-col p5=Q5
  python3 src/decode_kl.py data/kl/<input>.csv --pause-ms 3000

Standard library only (Python 3.8+).
"""
import argparse
import csv
import os
import re
import sys

# CONSTANTS -----------------------------------------------------------------------------
CODES = {
    "t": "insertText", "r": "insertReplacementText", "n": "insertLineBreak", "N": "insertParagraph",
    "p": "insertFromPaste", "d": "insertFromDrop", "c": "insertCompositionText",
    "b": "deleteContentBackward", "f": "deleteContentForward",
    "wb": "deleteWordBackward", "wf": "deleteWordForward",
    "lb": "deleteSoftLineBackward", "lf": "deleteSoftLineForward",
    "hb": "deleteHardLineBackward", "hf": "deleteHardLineForward",
    "x": "deleteByCut", "g": "deleteByDrag", "z": "historyUndo", "y": "historyRedo",
    "u": "UNEXPLAINED",
}
EDIT_TYPES = {"t", "k", "i", "u"}
LABELS = {"p": "PASTE", "z": "UNDO", "y": "REDO", "x": "CUT", "u": "UNEXPLAINED CHANGE",
          "r": "BROWSER REPLACEMENT", "c": "IME INPUT"}
_UNESC = re.compile(r"%u([0-9A-F]{4})|%([0-9A-F]{2})|_")
_SENT_END = re.compile(r"[.!?]+[\"')\]]*\s+")


# HELPERS ----------------------------------------------------------------------------- 
# unescape
def unesc(s):
    def rep(m):
        if m.group(1):
            return chr(int(m.group(1), 16))
        if m.group(2):
            return chr(int(m.group(2), 16))
        return " "
    # chr() of UTF-16 surrogate halves -> re-pair them into real characters
    return _UNESC.sub(rep, s).encode("utf-16", "surrogatepass").decode("utf-16", "surrogatepass")


def u16(s):
    """String -> list of UTF-16 code units (JavaScript string indexing)."""
    b = s.encode("utf-16-le", "surrogatepass")
    return [b[i] | (b[i + 1] << 8) for i in range(0, len(b), 2)]


def from_u16(units):
    return b"".join(x.to_bytes(2, "little") for x in units).decode("utf-16-le", "surrogatepass")


def u16len(s):
    return len(s.encode("utf-16-le", "surrogatepass")) // 2


# PARSING + REPLAY -----------------------------------------------------------------------------
def parse_log(body, tolerant=False):
    """Return (header, events). With tolerant=True, stop at the first malformed record."""
    recs = body.split("|")
    if not recs[0].startswith("h"):
        raise ValueError("log does not start with a header record")
    hf = recs[0][1:].split(",")
    header = {"version": int(hf[0]), "epoch0": int(hf[1]), "prompt": unesc(hf[2]),
              "initial": unesc(hf[3]) if len(hf) > 3 else "", "ua": unesc(hf[4]) if len(hf) > 4 else ""}
    events, t = [], 0
    for r in recs[1:]:
        if not r:
            continue
        try:
            typ, f = r[0], r[1:].split(",")
            t_new = t + int(f[0])
            ev = {"t": t_new, "type": typ}
            if typ == "t":
                ev.update(ins=unesc(f[1]), code="t")
            elif typ == "k":
                ev.update(ins="", code="b")
            elif typ in ("i", "u"):
                ev.update(pos=int(f[1]), ndel=int(f[2]), ins=unesc(f[3]),
                          code=(f[4] if typ == "i" and len(f) > 4 else "u"))
            elif typ in ("s", "c"):
                ev.update(start=int(f[1]), end=int(f[2]))
            elif typ == "b":
                ev.update(epoch=int(f[1]))
            elif typ == "P":
                ev.update(n=int(f[1]) if len(f) > 1 and f[1] else 0)
            elif typ == "e":
                ev.update(final_len=int(f[1]))
        except (ValueError, IndexError):
            if tolerant:
                break
            raise
        t = t_new
        events.append(ev)
    return header, events


def replay(header, events):
    """Apply edits in order. Annotates edit events with pos, ndel, deleted text and the text after."""
    doc = u16(header["initial"])
    next_pos = -1
    for ev in events:
        typ = ev["type"]
        if typ in EDIT_TYPES:
            if typ == "t":
                ev["pos"], ev["ndel"] = next_pos, 0
            elif typ == "k":
                ev["pos"], ev["ndel"] = next_pos - 1, 1
            ev["text_before"] = from_u16(doc)
            ins = u16(ev["ins"])
            ev["deleted"] = from_u16(doc[ev["pos"]:ev["pos"] + ev["ndel"]])
            doc[ev["pos"]:ev["pos"] + ev["ndel"]] = ins
            ev["text_after"] = from_u16(doc)
            ev["len_after"] = len(doc)
            next_pos = ev["pos"] + len(ins)
        elif typ in ("s", "c"):
            ev["selected"] = from_u16(doc[ev["start"]:ev["end"]])
            ev["text_now"] = from_u16(doc)
    return from_u16(doc)


# RENDERING -----------------------------------------------------------------------------
def sentence_index(text, pos):
    n = 1
    for m in _SENT_END.finditer(text):
        if m.end() <= pos:
            n += 1
        else:
            break
    return n


def fmt_t(ms):
    tenths = int(round(ms / 100.0))
    return "%02d:%02d.%d" % (tenths // 600, (tenths // 10) % 60, tenths % 10)


def q(s, limit=None):
    s = s.replace("\n", "↵")
    if limit and len(s) > limit:
        s = s[: limit // 2] + "…" + s[-limit // 2:]
    return '"' + s + '"'


def where(text, pos):
    """Human/LLM-readable location: 'end' or 'S2, after "…dows were "'."""
    if pos >= len(text.rstrip()):
        return "end"
    before = text[max(0, pos - 18):pos].replace("\n", "↵")
    return 'S%d, after "%s%s"' % (sentence_index(text, pos), "…" if pos > 18 else "", before)


def render(header, events, pause_ms):
    rows = []          # (t, kind, text)
    burst = None
    last_cut = None
    last_activity = None   # time of last edit or Begin (pauses are gaps between these)

    if header["initial"]:
        rows.append((0, "init", "INITIAL TEXT (prefilled): %s" % q(header["initial"])))

    def flush():
        nonlocal burst
        if not burst:
            return
        parts = []
        for op in burst["ops"]:
            loc = op["where"]
            if op["kind"] == "ins":
                s = "%s: +%s" % (loc, q(op["ins"]))
            elif op["kind"] == "del":
                s = "%s: −%s" % (loc, q(op["del"]))
            else:
                s = "%s: −%s +%s" % (loc, q(op["del"]), q(op["ins"]))
            parts.append((op["label"] + " " if op.get("label") else "") + s)
        dur = (burst["t1"] - burst["t0"]) / 1000.0
        note = " · %d typo char(s) fixed in-burst" % burst["typo"] if burst["typo"] else ""
        plain = all(o["kind"] == "ins" and not o.get("label") for o in burst["ops"])
        rows.append((burst["t0"], "edit", "%s (%.1f s) %s%s" % ("BURST" if plain else "EDIT", dur, " ; ".join(parts), note)))
        burst = None

    for ev in events:
        t, typ = ev["t"], ev["type"]
        is_activity = typ in EDIT_TYPES or typ == "b"
        if is_activity and last_activity is not None and t - last_activity >= pause_ms:
            flush()
            rows.append((last_activity, "pause", "⏸ %.1f s" % ((t - last_activity) / 1000.0)))

        if typ in EDIT_TYPES:
            pos, deleted, ins, code = ev["pos"], ev["deleted"], ev["ins"], ev["code"]
            text_before = ev["text_before"]
            label = LABELS.get(code)
            if code == "t" and len(ins) > 1:
                label = "MULTI-CHAR INSERT"
            if code == "x":
                last_cut = deleted
            if code == "p" and last_cut is not None and ins.strip() == last_cut.strip():
                label = "PASTE (= move of cut text)"
            if label or (burst and burst["ops"] and burst["ops"][-1].get("label")):
                flush()
            if burst is None:
                burst = {"t0": t, "t1": t, "ops": [], "typo": 0}
            burst["t1"] = t
            ops = burst["ops"]
            op = ops[-1] if ops else None
            merged = False
            if not label and op and not op.get("label"):
                if deleted == "" and op["kind"] in ("ins", "repl") and pos == op["pos"] + u16len(op["ins"]):
                    op["ins"] += ins                                   # contiguous typing / typing over a selection
                    merged = True
                elif (ins == "" and op["kind"] == "ins" and len(deleted) <= 2 and not re.search(r"\s", deleted)
                      and len(op["ins"]) > len(deleted) and op["ins"].endswith(deleted)
                      and pos == op["pos"] + u16len(op["ins"]) - u16len(deleted)):
                    op["ins"] = op["ins"][: len(op["ins"]) - len(deleted)]  # tiny typo fix at the leading edge
                    burst["typo"] += len(deleted)
                    merged = True
                elif ins == "" and op["kind"] == "del" and pos + u16len(deleted) == op["pos"]:
                    op["del"] = deleted + op["del"]                    # consecutive backspaces
                    op["pos"] = pos
                    merged = True
                elif ins == "" and op["kind"] == "del" and pos == op["pos"]:
                    op["del"] += deleted                               # consecutive forward deletes
                    merged = True
            if not merged:
                kind = "ins" if deleted == "" else ("del" if ins == "" else "repl")
                ops.append({"kind": kind, "pos": pos, "ins": ins, "del": deleted, "label": label,
                            "where": where(text_before, pos)})
            last_activity = t
            continue

        flush()
        if typ == "b":
            rows.append((t, "sync", "SYNC: Begin clicked (epoch %d)" % ev["epoch"]))
            last_activity = t
        elif typ == "s":
            caret = ev["start"] == ev["end"]
            if rows and rows[-1][1] in ("cursor", "select") and t - rows[-1][0] < 1000:
                pk, pt = rows[-1][1], rows[-1][2]
                pstart, pend = rows[-1][3], rows[-1][4]
                same_anchor = ev["start"] in (pstart, pend) or ev["end"] in (pstart, pend)
                if (pk == "cursor" and (caret or same_anchor)) or (pk == "select" and not caret and same_anchor):
                    rows.pop()                                         # collapse one gesture into its final state
            txt = ev["text_now"]
            if caret:
                rows.append((t, "cursor", "CURSOR → %s" % where(txt, ev["start"]), ev["start"], ev["end"]))
            else:
                loc = where(txt, ev["start"])
                if loc == "end":
                    loc = "S%d" % sentence_index(txt, ev["start"])
                rows.append((t, "select", "SELECT %s: %s" % (loc, q(ev["selected"], 300)), ev["start"], ev["end"]))
        elif typ == "c":
            rows.append((t, "copy", "COPY: %s" % q(ev["selected"], 300)))
        elif typ == "P":
            rows.append((t, "block", "BLOCKED PASTE (%s from outside the box)" % ("%d chars" % ev["n"] if ev["n"] else "text")))
        elif typ == "D":
            rows.append((t, "block", "BLOCKED DROP"))
        elif typ == "e":
            rows.append((t, "submit", "SUBMIT"))
    flush()
    rows.sort(key=lambda r: r[0])
    return [(r[0], r[2]) for r in rows]


# HANDLE EXPORT ----------------------------------------------------------------------------- 
def read_qualtrics_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        return [], []
    cols, data = rows[0], rows[1:]
    if len(data) >= 2 and data[1] and data[1][0].startswith("{\"ImportId\""):
        data = data[2:]                      # drop Qualtrics' question-text and ImportId rows
    return cols, [dict(zip(cols, r)) for r in data]


def find_prompts(cols, prefix):
    pat = re.compile(r"(__js_)?%s_([A-Za-z0-9]+)_n" % re.escape(prefix))
    out = {}
    for c in cols:
        m = pat.fullmatch(c)
        if m:
            out.setdefault(m.group(2), []).append((m.group(1) or "") + prefix)
    return out   # prompt -> candidate field prefixes (with/without __js_)


def assemble(row, prefixes, prompt):
    for pre in prefixes:                     # prefer whichever declaration actually has data
        meta = row.get("%s_%s_n" % (pre, prompt), "").strip()
        if meta:
            break
    else:
        return None, None
    parts = meta.split(",")
    if len(parts) >= 6 and parts[5]:
        return None, "logger error: %s" % parts[5]
    n, total, overflow, selfok = int(parts[0]), int(parts[1]), parts[2], parts[3]
    body = "".join(row.get("%s_%s_%d" % (pre, prompt, k), "") for k in range(n))
    problems = []
    if len(body) != total:
        problems.append("log length %d != expected %d (missing chunk fields? declare _0.._%d)" % (len(body), total, n - 1))
    if overflow == "1":
        problems.append("OVERFLOW: log hit CHUNK_SIZE*MAX_CHUNKS and the end was cut (raise MAX_CHUNKS)")
    if selfok != "1":
        problems.append("in-browser self-check failed")
    return body, "; ".join(problems)


def decode_prompt(row, prefixes, prompt, text_col=None):
    """Decode one response x prompt. Returns None if no log was saved, else a dict with
    body, problem (assemble problem or None), header, events (replayed), final, checks.
    Raises on a log that cannot be decoded."""
    body, problem = assemble(row, prefixes, prompt)
    if body is None:
        return {"body": None, "problem": problem}
    header, events = parse_log(body, tolerant="OVERFLOW" in (problem or ""))
    final = replay(header, events)
    checks = [problem] if problem else []
    end = [e for e in events if e["type"] == "e"]
    if not end:
        checks.append("no SUBMIT record (participant never clicked Next?)")
    elif u16len(final) != end[-1]["final_len"]:
        checks.append("replayed length %d != final length %d" % (u16len(final), end[-1]["final_len"]))
    if text_col:
        saved = row.get(text_col, "")
        if saved.replace("\r\n", "\n").strip() != final.replace("\r\n", "\n").strip():
            checks.append("replayed text != saved answer in %s" % text_col)
    if not any(e["type"] == "b" for e in events):
        checks.append("no Begin click (no sync marker)")
    return {"body": body, "problem": problem, "header": header, "events": events,
            "final": final, "checks": checks}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "output"))
    ap.add_argument("--prefix", default="kl", help="FIELD_PREFIX used in the logger (default kl)")
    ap.add_argument("--pause-ms", type=int, default=2000, help="pause/burst threshold for rendering (default 2000)")
    ap.add_argument("--text-col", action="append", default=[],
                    help="prompt=COLUMN of the saved essay answer, for the final-text check (repeatable)")
    ap.add_argument("--id-col", default="ResponseId")
    args = ap.parse_args()

    cols, rows = read_qualtrics_csv(args.csv)
    prompts = find_prompts(cols, args.prefix)
    if not prompts:
        sys.exit("No %s_<prompt>_n columns found. Are the embedded data fields declared in Survey Flow?" % args.prefix)
    text_cols = dict(tc.split("=", 1) for tc in args.text_col)
    os.makedirs(args.out, exist_ok=True)

    ev_path = os.path.join(args.out, "events.csv")
    n_ok = n_bad = 0
    with open(ev_path, "w", newline="", encoding="utf-8", errors="backslashreplace") as fh:
        w = csv.writer(fh)
        w.writerow(["response_id", "prompt", "t_ms", "epoch_ms", "event", "input_type", "pos", "n_del",
                    "deleted", "inserted", "sel_start", "sel_end", "selected", "doc_len_after"])
        for row in rows:
            rid = row.get(args.id_col, "?")
            for prompt, prefixes in sorted(prompts.items()):
                try:
                    d = decode_prompt(row, prefixes, prompt, text_cols.get(prompt))
                    body = d["body"]
                    if body is None:
                        print("MISS %s %s  %s" % (rid, prompt, d["problem"] or "no log saved (page JS failed or fields not declared?)"))
                        n_bad += 1
                        continue
                    header, events, final, checks = d["header"], d["events"], d["final"], d["checks"]

                    for ev in events:
                        w.writerow([rid, prompt, ev["t"], header["epoch0"] + ev["t"], ev["type"],
                                    CODES.get(ev.get("code", ""), ev.get("code", "")) if ev["type"] in EDIT_TYPES else "",
                                    ev.get("pos", ""), ev.get("ndel", ""), ev.get("deleted", ""), ev.get("ins", ""),
                                    ev.get("start", ""), ev.get("end", ""), ev.get("selected", ""), ev.get("len_after", "")])

                    tl = render(header, events, args.pause_ms)
                    with open(os.path.join(args.out, "timeline_%s_%s.md" % (rid, prompt)), "w",
                              encoding="utf-8", errors="backslashreplace") as th:
                        th.write("# %s · %s\n\n" % (rid, prompt))
                        th.write("Times are from page load (page-load epoch %d ms). Pause/burst threshold %d ms. "
                                 "S# = sentence number in the text at that moment.\n\n" % (header["epoch0"], args.pause_ms))
                        th.write("| Time | Interface activity |\n|---|---|\n")
                        for t, desc in tl:
                            th.write("| %s | %s |\n" % (fmt_t(t), desc.replace("|", "\\|")))
                        th.write("\n**Final text:** %s\n" % final.replace("\n", " ↵ "))
                except Exception as e:  # noqa: BLE001 - one bad log must not kill the run
                    print("FAIL %s %s  could not decode: %r" % (rid, prompt, e))
                    n_bad += 1
                    continue
                status = "OK  " if not checks else "FAIL"
                n_ok += not checks
                n_bad += bool(checks)
                print("%s %s %s  events=%d  log=%d chars  final=%d chars  %s"
                      % (status, rid, prompt, len(events), len(body), len(final), "; ".join(checks)))
    print("\n%d OK, %d problem(s).  Events: %s" % (n_ok, n_bad, ev_path))


if __name__ == "__main__":
    main()
