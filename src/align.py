#!/usr/bin/env python3
"""
Align Zoom think-aloud transcripts (WebVTT) with the keystroke logs decoded by decode_kl.py.

Input : the Qualtrics CSV export (same one decode_kl.py reads) and a folder of Zoom transcripts
        named <ResponseId>_GMT<YYYYMMDD>-<HHMMSS>_....vtt (recording start in UTC).
Output: <out>/merged_<rid>_pos<k>_<prompt>.md   interface + think-aloud timeline per prompt
        <out>/outside_<rid>.md                  speech before / between / after prompts
        <out>/slices.jsonl                      one record per in-prompt utterance (for LLM prompting)
        <out>/report.md                         validation report (summary also on stdout)

Usage:
  python3 src/align.py data/kl/<export>.csv --transcripts data/transcripts --researcher "Name"

Standard library only (Python 3.9+, for zoneinfo).
"""
import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decode_kl as kl  # noqa: E402

_NAME = re.compile(r"^(.+?)_GMT(\d{8})-(\d{6})")
_TS = re.compile(r"(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{3})")
_SPEAKER = re.compile(r"^([^:\n]{1,60}):\s+(.*)$", re.S)
SERVER_FMT = "%Y-%m-%d %H:%M:%S"
DATE_COLS = ("StartDate", "EndDate", "RecordedDate")


# TRANSCRIPTS -----------------------------------------------------------------------------
def parse_transcript_name(fname):
    """'R_x_GMT20261118-122651_Recording.transcript.vtt' -> ('R_x', recording start in UTC ms), or None."""
    m = _NAME.match(os.path.basename(fname))
    if not m:
        return None
    try:
        dt = datetime.strptime(m.group(2) + m.group(3), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    return m.group(1), int(dt.timestamp() * 1000)


def _ts_ms(s):
    m = _TS.fullmatch(s.strip())
    if not m:
        raise ValueError("bad VTT timestamp %r" % s)
    h, mi, se, ms = (int(x) if x else 0 for x in m.groups())
    return ((h * 60 + mi) * 60 + se) * 1000 + ms


def parse_vtt(path):
    """Return cues: {idx, start_ms, end_ms, speaker, text}. Times are offsets from recording start."""
    with open(path, encoding="utf-8-sig") as fh:
        blocks = re.split(r"\n\s*\n", fh.read().replace("\r\n", "\n"))
    cues = []
    for b in blocks:
        lines = b.strip().split("\n")
        for i, line in enumerate(lines):
            if "-->" in line:
                a, z = line.split("-->", 1)
                text = " ".join(l.strip() for l in lines[i + 1:] if l.strip())
                m = _SPEAKER.match(text)
                speaker, text = (m.group(1).strip(), m.group(2).strip()) if m else (None, text)
                cues.append({"idx": len(cues) + 1, "start_ms": _ts_ms(a), "end_ms": _ts_ms(z.split()[0]),
                             "speaker": speaker, "text": text})
                break
    return cues


def segment(cues):
    """Speech units used downstream. For now one unit = one VTT cue.
    HOOK: finer segmentation (e.g. Whisper word timestamps grouped into thought units) goes here;
    it must return the same dict shape, and nothing downstream needs to change."""
    return cues


# EXPORT -----------------------------------------------------------------------------
def export_timezones(path):
    """Read Qualtrics' ImportId row -> {column: tz name} for the date columns."""
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = [r for _, r in zip(range(3), csv.reader(fh))]
    out = {}
    if len(rows) == 3:
        for col, cell in zip(rows[0], rows[2]):
            if col in DATE_COLS:
                try:
                    out[col] = json.loads(cell).get("timeZone")
                except ValueError:
                    pass
    return out


def parse_server_time(s, tz):
    """'2026-11-18 05:26:40' in tz -> UTC ms. Raises ValueError with a fix-it message otherwise."""
    try:
        dt = datetime.strptime(s.strip(), SERVER_FMT)
    except ValueError:
        raise ValueError("date %r is not YYYY-MM-DD HH:MM:SS; re-export from Qualtrics and don't re-save it "
                         "in Excel" % s)
    if not tz:
        raise ValueError("no time zone found in the export's ImportId row")
    return int(dt.replace(tzinfo=ZoneInfo(tz)).timestamp() * 1000)


def load_participants(csv_path, prefix="kl", text_cols=None, id_col="ResponseId"):
    """rid -> {row, prompts: {pid: decoded dict}, decoder: [status lines]}."""
    cols, rows = kl.read_qualtrics_csv(csv_path)
    prompts = kl.find_prompts(cols, prefix)
    if not prompts:
        raise SystemExit("No %s_<prompt>_n columns found in %s" % (prefix, csv_path))
    text_cols = text_cols or {}
    out = {}
    for row in rows:
        rid = row.get(id_col, "?")
        p = out.setdefault(rid, {"row": row, "prompts": {}, "decoder": []})
        for pid, prefixes in sorted(prompts.items()):
            try:
                d = kl.decode_prompt(row, prefixes, pid, text_cols.get(pid))
            except Exception as e:  # noqa: BLE001 - one bad log must not kill the run
                p["decoder"].append("FAIL %s %s  could not decode: %r" % (rid, pid, e))
                continue
            if d["body"] is None:
                p["decoder"].append("MISS %s %s  %s" % (rid, pid, d["problem"] or "no log saved"))
                continue
            if d["checks"]:
                p["decoder"].append("FAIL %s %s  %s" % (rid, pid, "; ".join(d["checks"])))
            p["prompts"][pid] = d
    return out


# ALIGNMENT -----------------------------------------------------------------------------
def prompt_windows(p):
    """Ordered list of windows {pid, pos, start, end, begin, skipped, no_submit, d, warn}."""
    wins = []
    for pid, d in p["prompts"].items():
        e0, evs = d["header"]["epoch0"], d["events"]
        begins = [ev["epoch"] for ev in evs if ev["type"] == "b"]
        submits = [e0 + ev["t"] for ev in evs if ev["type"] == "e"]
        wins.append({"pid": pid, "d": d, "start": e0, "begin": begins[0] if begins else None,
                     "end": submits[-1] if submits else None, "skipped": not begins,
                     "last": e0 + (evs[-1]["t"] if evs else 0)})
    wins.sort(key=lambda w: (w["begin"] if w["begin"] is not None else w["start"]))
    warn = None
    if [w["pid"] for w in wins] != [w["pid"] for w in sorted(wins, key=lambda w: w["start"])]:
        warn = "order by Begin differs from order by page load"
    for i, w in enumerate(wins):
        w["pos"] = i + 1
        w["no_submit"] = w["end"] is None
        if w["no_submit"]:
            w["end"] = wins[i + 1]["start"] if i + 1 < len(wins) else w["last"]
    return wins, warn


def clock_check(p, tzs, tol_ms):
    """Compare browser epochs with Qualtrics' server-side StartDate/EndDate."""
    row, wins = p["row"], p["windows"]
    res = {"recorded": row.get("RecordedDate", "")}
    if not wins:
        return dict(res, status="ERROR", msg="no decoded prompts")
    try:
        sd = parse_server_time(row.get("StartDate", ""), tzs.get("StartDate"))
        ed = parse_server_time(row.get("EndDate", ""), tzs.get("EndDate"))
    except ValueError as e:
        return dict(res, status="ERROR", msg=str(e))
    gs = min(w["start"] for w in wins) - sd
    ge = max(w["end"] for w in wins) - ed
    ok = abs(gs) <= tol_ms and abs(ge) <= tol_ms
    return dict(res, status="PASS" if ok else "WARN", gap_start_ms=gs, gap_end_ms=ge,
                msg="first page load − StartDate = %+.2f s; last submit − EndDate = %+.2f s (tol %.1f s)"
                    % (gs / 1000, ge / 1000, tol_ms / 1000))


def _anchor_epochs(p):
    """Begin epochs, or page-load epochs when the participant clicked no Begin at all."""
    b = [w["begin"] for w in p["windows"] if w["begin"] is not None]
    return b or [w["start"] for w in p["windows"]]


def _fits(p, span):
    eps = _anchor_epochs(p)
    return bool(eps) and all(span[0] <= x <= span[1] for x in eps)


def match_transcripts(paths, participants, offset_ms):
    """-> (matches {rid: transcript dict}, issues dict). Never guesses: a bad match is skipped."""
    issues = {"unparseable": [], "duplicates": {}, "unmatched": [], "span_fail": [], "no_transcript": []}
    by_prefix = {}
    for path in sorted(paths):
        parsed = parse_transcript_name(path)
        if not parsed:
            issues["unparseable"].append(os.path.basename(path))
            continue
        by_prefix.setdefault(parsed[0], []).append((path, parsed[1]))
    matches = {}
    for prefix, files in by_prefix.items():
        if len(files) > 1:
            issues["duplicates"][prefix] = [os.path.basename(f) for f, _ in files]
            continue
        path, rec0 = files[0]
        cues = parse_vtt(path)
        last_end = max((c["end_ms"] for c in cues), default=0)
        span = (rec0 + offset_ms, rec0 + last_end + offset_ms)
        hints = [rid for rid, p in participants.items() if rid != prefix and _fits(p, span)]
        t = {"file": os.path.basename(path), "rec0": rec0, "span": span, "cues": cues, "hints": hints}
        if prefix not in participants:
            issues["unmatched"].append(t)
        elif not _fits(participants[prefix], span):
            issues["span_fail"].append(dict(t, rid=prefix))
        else:
            matches[prefix] = t
    matched_or_tried = set(matches) | {t["rid"] for t in issues["span_fail"]} | set(issues["duplicates"])
    issues["no_transcript"] = [rid for rid in participants if rid not in matched_or_tried]
    return matches, issues


def assign_cues(units, wins):
    """Assign each unit by its START time. -> (per-pid lists, outside list). Adds 'crosses' flags."""
    per = {w["pid"]: [] for w in wins}
    outside = []
    for u in units:
        s, e = u["abs_start"], u["abs_end"]
        for i, w in enumerate(wins):
            if w["start"] <= s <= w["end"]:
                u["crosses"] = e > w["end"]
                per[w["pid"]].append(u)
                break
        else:
            if not wins or s < wins[0]["start"]:
                u["where"] = "before first prompt"
                nxt = wins[0] if wins else None
            elif s > wins[-1]["end"]:
                u["where"], nxt = "after last prompt", None
            else:
                k = max(i for i, w in enumerate(wins) if w["end"] < s)
                a, b = wins[k], wins[k + 1]
                u["where"] = "between pos %d (%s) and pos %d (%s)" % (a["pos"], a["pid"], b["pos"], b["pid"])
                nxt = b
            u["crosses"] = bool(nxt) and e > nxt["start"]
            outside.append(u)
    return per, outside


def doc_at(d, epoch_ms):
    """Document text as it stood at epoch_ms, rebuilt from the log (never the final text)."""
    e0, text = d["header"]["epoch0"], d["header"]["initial"]
    for ev in d["events"]:
        if e0 + ev["t"] > epoch_ms:
            break
        if ev["type"] in kl.EDIT_TYPES:
            text = ev["text_after"]
    return text


# OUTPUT -----------------------------------------------------------------------------
def utc(ms):
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _cell(s):
    return s.replace("|", "\\|")


def write_merged(path, rid, w, n, t, units, rows, offset_ms, pause_ms):
    e0 = w["start"]
    merged = [(r[0], 0, r[1], "") for r in rows] + [(u["abs_start"] - e0, 1, "", kl.q(u["text"])) for u in units]
    merged.sort(key=lambda r: (r[0], r[1]))
    with open(path, "w", encoding="utf-8", errors="backslashreplace") as fh:
        fh.write("# %s · %s · display position %d of %d%s\n\n"
                 % (rid, w["pid"], w["pos"], n, " · SKIPPED (no Begin)" if w["skipped"] else ""))
        fh.write("Transcript: %s · offset %.1f s · times from page load (epoch %d ms) · pause threshold %d ms · "
                 "speech times are ±1 s (recording start in the filename has 1-second resolution)\n\n"
                 % (t["file"], offset_ms / 1000, e0, pause_ms))
        fh.write("| Time | Interface activity | Think-aloud |\n|---|---|---|\n")
        for rel, _, act, talk in merged:
            fh.write("| %s | %s | %s |\n" % (kl.fmt_t(rel), _cell(act), _cell(talk)))


def write_outside(path, rid, t, outside):
    with open(path, "w", encoding="utf-8", errors="backslashreplace") as fh:
        fh.write("# %s · speech outside any prompt\n\nTranscript: %s · Rec time = offset from recording start\n\n"
                 % (rid, t["file"]))
        fh.write("| Rec time | UTC | Where | Speech |\n|---|---|---|---|\n")
        for u in outside:
            fh.write("| %s | %s | %s%s | %s |\n" % (kl.fmt_t(u["abs_start"] - t["rec0"]), utc(u["abs_start"])[11:-1],
                                                    u["where"], " (runs into next prompt)" if u["crosses"] else "",
                                                    _cell(kl.q(u["text"]))))


def slice_records(rid, w, n, t, units, rows, offset_ms, before_ms, after_ms):
    e0 = w["start"]
    for u in units:
        rel = u["abs_start"] - e0
        doc = doc_at(w["d"], u["abs_start"])
        yield {"response_id": rid, "prompt_id": w["pid"], "display_pos": w["pos"], "n_prompts": n,
               "skipped": w["skipped"], "transcript": t["file"], "cue_index": u["idx"], "text": u["text"],
               "abs_start_ms": u["abs_start"], "abs_end_ms": u["abs_end"], "abs_start_utc": utc(u["abs_start"]),
               "rel_start_ms": rel, "rel_end_ms": u["abs_end"] - e0, "rel_start": kl.fmt_t(rel),
               "crosses_boundary": u["crosses"], "doc_text": doc, "doc_len": len(doc), "offset_ms": offset_ms,
               "window_before_ms": before_ms, "window_after_ms": after_ms,
               "events": [{"rel_ms": r[0], "rel": kl.fmt_t(r[0]), "desc": r[1]} for r in rows
                          if rel - before_ms <= r[0] <= rel + after_ms]}


def write_report(path, res):
    L = ["# Alignment report", "",
         "Offset %.1f s · clock tolerance %.1f s · slice window −%.1f s / +%.1f s" % (
             res["offset_ms"] / 1000, res["tol_ms"] / 1000, res["before_ms"] / 1000, res["after_ms"] / 1000),
         "", "Notes: speech times are ±1 s (the recording start in the filename has 1-second resolution). "
         "The clock check compares browser epochs with Qualtrics server dates, which are whole seconds and assume "
         "the prompts are the first and last survey pages, so it only catches gross clock skew.", ""]
    for rid, p in res["participants"].items():
        L += ["## %s" % rid, ""]
        t = p.get("transcript")
        if t:
            L.append("- Transcript: `%s` · time cross-check PASS (Begins inside recording %s – %s)"
                     % (t["file"], utc(t["span"][0]), utc(t["span"][1])))
        elif p.get("span_fail"):
            L.append("- Transcript: `%s` · time cross-check **FAIL**: Begins not inside the recording span. "
                     "Likely typo or wrong file; skipped." % p["span_fail"])
        else:
            L.append("- Transcript: **none**")
        c = p["clock"]
        L.append("- Clock check: **%s**: %s · RecordedDate %s" % (c["status"], c["msg"], c["recorded"]))
        if p["order_warn"]:
            L.append("- Order warning: %s" % p["order_warn"])
        L += ["", "| Pos | Prompt | Skipped | Window (UTC) | Speech units |", "|---|---|---|---|---|"]
        for w in p["windows"]:
            L.append("| %d | %s | %s | %s – %s%s | %s |" % (
                w["pos"], w["pid"], "yes" if w["skipped"] else "", utc(w["start"])[11:-1], utc(w["end"])[11:-1],
                " (no submit)" if w["no_submit"] else "", w.get("n_units", "–")))
        L.append("")
        if t:
            if not p["speechless"]:
                L.append("- Prompts with no speech: none")
            else:
                L.append("- Prompts with no speech: %s" % ", ".join(p["speechless"]))
            L.append("- Speech outside prompts: %d (%s)" % (len(p["outside"]), ", ".join(
                "%s: %d" % (k, v) for k, v in p["outside_counts"].items()) or "none"))
            L.append("- Units crossing a boundary (assigned by start): %s" % (", ".join(
                "cue %d @ %s" % (u["idx"], kl.fmt_t(u["abs_start"] - t["rec0"])) for u in p["crossing"]) or "none"))
            L.append("- Researcher cues dropped: %d · cues with no speaker label (kept): %d"
                     % (p["n_researcher"], p["n_unlabeled"]))
        L.append("- Decoder FAIL/MISS lines: %s" % ("none" if not p["decoder"] else ""))
        L += ["  - `%s`" % x for x in p["decoder"]]
        L.append("")
    iss = res["issues"]
    L += ["## Run-level", ""]
    L.append("- ResponseIds with no transcript: %s" % (", ".join(iss["no_transcript"]) or "none"))
    L.append("- Transcripts with no matching ResponseId: %s" % (", ".join(
        "`%s`%s" % (t["file"], " (its time span fits %s; check the prefix)" % ", ".join(t["hints"]) if t["hints"]
                    else "") for t in iss["unmatched"]) or "none"))
    L.append("- Transcripts failing the time cross-check: %s" % (", ".join(
        "`%s`%s" % (t["file"], " (its time span fits %s)" % ", ".join(t["hints"]) if t["hints"] else "")
        for t in iss["span_fail"]) or "none"))
    L.append("- Duplicate prefixes (all skipped): %s" % ("; ".join(
        "%s: %s" % (k, ", ".join(v)) for k, v in iss["duplicates"].items()) or "none"))
    L.append("- Unparseable filenames: %s" % (", ".join(iss["unparseable"]) or "none"))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


# DRIVER -----------------------------------------------------------------------------
def run(csv_path, transcripts, out, prefix="kl", text_cols=None, researchers=(), offset_s=0.0,
        clock_tol_s=5.0, before_s=10.0, after_s=10.0, pause_ms=2000, id_col="ResponseId"):
    offset_ms, tol_ms = int(round(offset_s * 1000)), int(round(clock_tol_s * 1000))
    before_ms, after_ms = int(round(before_s * 1000)), int(round(after_s * 1000))
    researchers = set(researchers)
    participants = load_participants(csv_path, prefix, text_cols, id_col)
    tzs = export_timezones(csv_path)
    for p in participants.values():
        p["windows"], p["order_warn"] = prompt_windows(p)
        p["clock"] = clock_check(p, tzs, tol_ms)
    paths = [os.path.join(transcripts, f) for f in os.listdir(transcripts) if f.lower().endswith(".vtt")]
    matches, issues = match_transcripts(paths, participants, offset_ms)
    for t in issues["span_fail"]:
        participants[t["rid"]]["span_fail"] = t["file"]

    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "slices.jsonl"), "w", encoding="utf-8") as sl:
        for rid, t in matches.items():
            p = participants[rid]
            p["transcript"] = t
            units = segment(t["cues"])
            p["n_researcher"] = sum(u["speaker"] in researchers for u in units)
            p["n_unlabeled"] = sum(u["speaker"] is None for u in units)
            units = [dict(u, abs_start=t["rec0"] + u["start_ms"] + offset_ms,
                          abs_end=t["rec0"] + u["end_ms"] + offset_ms)
                     for u in units if u["speaker"] not in researchers]
            per, outside = assign_cues(units, p["windows"])
            p["outside"] = outside
            p["outside_counts"] = {}
            for u in outside:
                k = u["where"].split(" (")[0] if u["where"].startswith("between") else u["where"]
                k = "between prompts" if k.startswith("between") else k
                p["outside_counts"][k] = p["outside_counts"].get(k, 0) + 1
            p["crossing"] = [u for u in units if u["crosses"]]
            p["speechless"] = []
            n = len(p["windows"])
            for w in p["windows"]:
                d, us = w["d"], per[w["pid"]]
                w["n_units"] = len(us)
                if not us:
                    p["speechless"].append("%s (pos %d)" % (w["pid"], w["pos"]))
                rows = kl.render(d["header"], d["events"], pause_ms)
                write_merged(os.path.join(out, "merged_%s_pos%d_%s.md" % (rid, w["pos"], w["pid"])),
                             rid, w, n, t, us, rows, offset_ms, pause_ms)
                for rec in slice_records(rid, w, n, t, us, rows, offset_ms, before_ms, after_ms):
                    sl.write(json.dumps(rec, ensure_ascii=False) + "\n")
            write_outside(os.path.join(out, "outside_%s.md" % rid), rid, t, outside)

    res = {"participants": participants, "issues": issues, "offset_ms": offset_ms, "tol_ms": tol_ms,
           "before_ms": before_ms, "after_ms": after_ms}
    write_report(os.path.join(out, "report.md"), res)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("csv")
    ap.add_argument("--transcripts", default=os.path.join(here, "..", "data", "transcripts"))
    ap.add_argument("--out", default=os.path.join(here, "..", "data", "output", "aligned"))
    ap.add_argument("--prefix", default="kl")
    ap.add_argument("--text-col", action="append", default=[], help="prompt=COLUMN (passed to the decoder checks)")
    ap.add_argument("--id-col", default="ResponseId")
    ap.add_argument("--researcher", action="append", default=[], help="speaker name whose cues are dropped (repeatable)")
    ap.add_argument("--offset-s", type=float, default=0.0, help="added to every transcript time (default 0)")
    ap.add_argument("--clock-tol-s", type=float, default=5.0, help="clock sanity-check tolerance (default 5)")
    ap.add_argument("--before-s", type=float, default=10.0, help="slice window before an utterance (default 10)")
    ap.add_argument("--after-s", type=float, default=10.0, help="slice window after an utterance (default 10)")
    ap.add_argument("--pause-ms", type=int, default=2000)
    a = ap.parse_args()
    res = run(a.csv, a.transcripts, a.out, a.prefix, dict(tc.split("=", 1) for tc in a.text_col), a.researcher,
              a.offset_s, a.clock_tol_s, a.before_s, a.after_s, a.pause_ms, a.id_col)
    for rid, p in res["participants"].items():
        t = p.get("transcript")
        print("%-20s transcript=%s  clock=%s  order=%s  outside=%s" % (
            rid, t["file"] if t else ("SPAN FAIL" if p.get("span_fail") else "none"), p["clock"]["status"],
            ",".join(w["pid"] for w in p["windows"]), len(p["outside"]) if t else "-"))
    iss = res["issues"]
    for k in ("unmatched", "span_fail"):
        for t in iss[k]:
            print("WARN %s: %s" % (k, t["file"]))
    for k, v in iss["duplicates"].items():
        print("WARN duplicate prefix %s: %s" % (k, ", ".join(v)))
    print("Report: %s" % os.path.join(a.out, "report.md"))


if __name__ == "__main__":
    main()
