"""Turn data/output/aligned/slices.jsonl into readable Markdown.

Writes one file per participant to data/output/formatted/ (does not touch the aligned files).

Run:  python3 src/format_md.py [--slices PATH] [--out DIR]
"""
import argparse
import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SLICES = os.path.join(HERE, "..", "data", "output", "aligned", "slices.jsonl")
DEFAULT_OUT = os.path.join(HERE, "..", "data", "output", "formatted")


def load(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def cell(s):
    return s.replace("|", "\\|").replace("\n", " ")


def prompt_table(rs):
    """One Time | Interface activity | Think-aloud table, events de-duplicated across slices."""
    rows = {}
    for r in rs:
        for e in r["events"]:
            rows[(e["rel_ms"], 0, e["desc"])] = (e["rel"], e["desc"], "")
        rows[(r["rel_start_ms"], 1, r["text"])] = (r["rel_start"], "", '"%s"' % r["text"])
    out = ["| Time | Interface activity | Think-aloud |", "|---|---|---|"]
    for k in sorted(rows):
        t, act, talk = rows[k]
        out.append("| %s | %s | %s |" % (t, cell(act), cell(talk)))
    return out + [""]


def format_participant(rid, recs):
    first = recs[0]
    lines = ["# %s" % rid, "",
             "Transcript: `%s` · offset %.1f s · activity shown within −%.0f s / +%.0f s of speech"
             % (first["transcript"], first["offset_ms"] / 1000,
                first["window_before_ms"] / 1000, first["window_after_ms"] / 1000), ""]

    by_prompt = defaultdict(list)
    for r in recs:
        by_prompt[(r["display_pos"], r["prompt_id"])].append(r)

    for (pos, pid), rs in sorted(by_prompt.items()):
        lines += ["## %s · position %d of %d%s" % (pid, pos, rs[0]["n_prompts"], " · SKIPPED" if rs[0]["skipped"] else ""), ""]
        lines += prompt_table(rs)
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slices", default=DEFAULT_SLICES)
    ap.add_argument("--out", default=DEFAULT_OUT)
    a = ap.parse_args()

    by_rid = defaultdict(list)
    for r in load(a.slices):
        by_rid[r["response_id"]].append(r)

    os.makedirs(a.out, exist_ok=True)
    for rid, recs in sorted(by_rid.items()):
        path = os.path.join(a.out, rid + ".md")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(format_participant(rid, recs))
        print("wrote", os.path.relpath(path))


if __name__ == "__main__":
    main()
