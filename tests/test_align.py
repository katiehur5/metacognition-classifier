"""Tests for src/align.py.  Run:  python3 -m unittest discover tests"""
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import align  # noqa: E402
import decode_kl as kl  # noqa: E402

R7 = "R_7jBtXKIS84alAWj"
PILOT = "R_5lE0CYuVeX7odJo"
R7_CSV = os.path.join(ROOT, "data", "kl", R7 + ".csv")
PILOT_CSV = os.path.join(ROOT, "data", "kl", "pilot.csv")
REAL_VTT = os.path.join(ROOT, "data", "transcripts", R7 + "_GMT20261118-122651_Recording.transcript.vtt")
R7_REC = R7 + "_GMT20261118-122651_Recording.transcript.vtt"   # recording start 12:26:51 UTC


def ts(sec):
    ms = int(round(sec * 1000))
    return "%02d:%02d:%02d.%03d" % (ms // 3600000, ms // 60000 % 60, ms // 1000 % 60, ms % 1000)


def make_vtt(folder, name, cues):
    """cues: list of (start_s, end_s, 'Speaker: text')."""
    with open(os.path.join(folder, name), "w", encoding="utf-8") as fh:
        fh.write("WEBVTT\n\n")
        for i, (a, z, text) in enumerate(cues, 1):
            fh.write("%d\n%s --> %s\n%s\n\n" % (i, ts(a), ts(z), text))


def shift_export(src, dst, rid, delta_ms, date_overrides=None):
    """Copy an export, shifting every page-load epoch and Begin epoch of one response by delta_ms.
    Epochs stay 13 digits, so chunk lengths and the _n totals stay valid."""
    with open(src, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.reader(fh))
    cols = rows[0]
    rid_i = cols.index("ResponseId")

    def shift(m):
        return m.group(1) + str(int(m.group(2)) + delta_ms)

    for r in rows[3:]:
        if r[rid_i] != rid:
            continue
        for i, c in enumerate(cols):
            if re.fullmatch(r"(__js_)?kl_[A-Za-z0-9]+_\d+", c) and r[i]:
                r[i] = re.sub(r"(^h\d+,)(\d{13})", shift, r[i])              # header epoch0 (chunk 0)
                r[i] = re.sub(r"(\|b\d+,)(\d{13})", shift, r[i])             # Begin wall-clock epoch
            if date_overrides and c in date_overrides:
                r[i] = date_overrides[c]
    with open(dst, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(rows)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.tr = os.path.join(self.tmp, "tr")
        self.out = os.path.join(self.tmp, "out")
        os.makedirs(self.tr)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def run_align(self, csv_path, **kw):
        return align.run(csv_path, self.tr, self.out, **kw)

    def slices(self):
        with open(os.path.join(self.out, "slices.jsonl"), encoding="utf-8") as fh:
            return [json.loads(l) for l in fh]

    def report(self):
        with open(os.path.join(self.out, "report.md"), encoding="utf-8") as fh:
            return fh.read()


# Known R_7j windows (offsets from recording start 12:26:51):
#   pos1 p3 13.122–140.839 · pos2 p5 141.380–258.288 · pos3 p2 · pos4 p4 · pos5 p1 262.173–263.970
SYNTH = [(5, 6, "P: before"), (20, 21, "P: in p3"), (140.5, 142, "P: crosses"), (141.0, 141.2, "P: between"),
         (150, 151, "P: in p5"), (262.5, 263, "P: in p1"), (300, 301, "P: after")]


class TestRandomizedOrder(Base):
    def test_synthetic_cues_land_in_right_prompts(self):
        make_vtt(self.tr, R7_REC, SYNTH)
        res = self.run_align(R7_CSV)
        p = res["participants"][R7]
        self.assertEqual([w["pid"] for w in p["windows"]], ["p3", "p5", "p2", "p4", "p1"])
        self.assertEqual([w["skipped"] for w in p["windows"]], [False, False, True, True, True])
        by_text = {s["text"]: s for s in self.slices()}
        self.assertEqual((by_text["in p3"]["prompt_id"], by_text["in p3"]["display_pos"]), ("p3", 1))
        self.assertEqual(by_text["in p3"]["rel_start"], "00:06.9")            # 20 − 13.122
        self.assertEqual((by_text["in p5"]["prompt_id"], by_text["in p5"]["display_pos"]), ("p5", 2))
        self.assertEqual(by_text["in p5"]["rel_start"], "00:08.6")            # 150 − 141.380
        self.assertEqual((by_text["in p1"]["prompt_id"], by_text["in p1"]["display_pos"]), ("p1", 5))
        self.assertTrue(by_text["in p1"]["skipped"])
        self.assertEqual(by_text["crosses"]["prompt_id"], "p3")
        self.assertTrue(by_text["crosses"]["crosses_boundary"])
        where = {u["text"]: u["where"] for u in p["outside"]}
        self.assertEqual(where, {"before": "before first prompt", "between": "between pos 1 (p3) and pos 2 (p5)",
                                 "after": "after last prompt"})
        self.assertTrue(os.path.exists(os.path.join(self.out, "merged_%s_pos1_p3.md" % R7)))
        self.assertTrue(os.path.exists(os.path.join(self.out, "merged_%s_pos3_p2.md" % R7)))
        self.assertIn("p2 (pos 3)", self.report())                           # prompt with no speech

    def test_real_transcript(self):
        shutil.copy(REAL_VTT, self.tr)
        self.run_align(R7_CSV)
        by_text = {s["text"]: s for s in self.slices()}
        begin = by_text["All right, let's begin."]
        self.assertEqual(begin["prompt_id"], "p3")
        self.assertTrue(1000 <= begin["rel_start_ms"] <= 3000)
        self.assertEqual(by_text[next(t for t in by_text if t.startswith("Convince your"))]["prompt_id"], "p5")
        with open(os.path.join(self.out, "outside_%s.md" % R7), encoding="utf-8") as fh:
            self.assertRegex(fh.read(), r"after last prompt \| \"blah blah")

    def test_no_labels_in_outputs(self):
        shutil.copy(REAL_VTT, self.tr)
        self.run_align(R7_CSV)
        for f in os.listdir(self.out):
            if f.startswith("merged_"):
                with open(os.path.join(self.out, f), encoding="utf-8") as fh:
                    self.assertNotRegex(fh.read().lower(), r"\b(content|craft)\b")


class TestMatching(Base):
    def test_mistyped_prefix(self):
        make_vtt(self.tr, R7_REC.replace("alAWj", "alAWX"), SYNTH)
        res = self.run_align(R7_CSV)
        iss = res["issues"]
        self.assertEqual(len(iss["unmatched"]), 1)
        self.assertEqual(iss["unmatched"][0]["hints"], [R7])                 # hint only
        self.assertEqual(iss["no_transcript"], [R7])
        self.assertFalse([f for f in os.listdir(self.out) if f.startswith("merged_")])
        self.assertIn("fits %s" % R7, self.report())

    def test_span_mismatch(self):
        make_vtt(self.tr, PILOT + "_GMT20261118-122651_Recording.transcript.vtt", SYNTH)
        res = self.run_align(PILOT_CSV)
        self.assertEqual([t["rid"] for t in res["issues"]["span_fail"]], [PILOT])
        self.assertNotIn("transcript", res["participants"][PILOT])
        self.assertFalse([f for f in os.listdir(self.out) if f.startswith("merged_")])
        self.assertIn("cross-check **FAIL**", self.report())

    def test_duplicate_prefix(self):
        make_vtt(self.tr, R7_REC, SYNTH)
        make_vtt(self.tr, R7 + "_GMT20261118-122652_Recording.transcript.vtt", SYNTH)
        res = self.run_align(R7_CSV)
        self.assertEqual(len(res["issues"]["duplicates"][R7]), 2)
        self.assertFalse([f for f in os.listdir(self.out) if f.startswith("merged_")])

    def test_unparseable_name(self):
        make_vtt(self.tr, "zoom_recording.vtt", SYNTH)
        res = self.run_align(R7_CSV)
        self.assertEqual(res["issues"]["unparseable"], ["zoom_recording.vtt"])


class TestClock(Base):
    def test_pilot_passes(self):
        c = self.run_align(PILOT_CSV)["participants"][PILOT]["clock"]
        self.assertEqual(c["status"], "PASS")
        self.assertAlmostEqual(c["gap_start_ms"], 350, delta=5)

    def test_skew_10s_warns_3s_passes(self):
        csv10 = os.path.join(self.tmp, "skew10.csv")
        shift_export(PILOT_CSV, csv10, PILOT, 10000)
        self.assertEqual(self.run_align(csv10)["participants"][PILOT]["clock"]["status"], "WARN")
        csv3 = os.path.join(self.tmp, "skew3.csv")
        shift_export(PILOT_CSV, csv3, PILOT, 3000)
        self.assertEqual(self.run_align(csv3)["participants"][PILOT]["clock"]["status"], "PASS")

    def test_shift_keeps_decoder_happy(self):
        csv10 = os.path.join(self.tmp, "skew10.csv")
        shift_export(PILOT_CSV, csv10, PILOT, 10000)
        p = self.run_align(csv10)["participants"][PILOT]
        self.assertFalse([l for l in p["decoder"] if "no Begin" not in l])

    def test_minute_only_dates_error(self):
        bad = os.path.join(self.tmp, "excel.csv")
        shift_export(PILOT_CSV, bad, PILOT, 0, {"StartDate": "11/13/26 9:06", "EndDate": "11/13/26 9:27"})
        c = self.run_align(bad)["participants"][PILOT]["clock"]
        self.assertEqual(c["status"], "ERROR")
        self.assertIn("re-export", c["msg"])


class TestSlices(Base):
    def setUp(self):
        super().setUp()
        shutil.copy(REAL_VTT, self.tr)

    def test_doc_text_at_time_not_final(self):
        self.run_align(R7_CSV)
        p = align.load_participants(R7_CSV)[R7]
        d3 = p["prompts"]["p3"]
        for s in self.slices():
            if s["prompt_id"] != "p3":
                continue
            # independent rebuild: replay only the events up to the utterance time
            e0 = d3["header"]["epoch0"]
            evs = [dict(e) for e in d3["events"] if e0 + e["t"] <= s["abs_start_ms"]]
            self.assertEqual(s["doc_text"], kl.replay(d3["header"], evs))
        first = next(s for s in self.slices() if s["prompt_id"] == "p3")
        self.assertEqual(first["doc_text"], "")                               # before any edit -> initial text
        mid = [s for s in self.slices() if s["prompt_id"] == "p3" and 0 < s["doc_len"] < len(d3["final"])]
        self.assertTrue(mid)                                                  # some slices see partial text

    def test_window_params(self):
        self.run_align(R7_CSV, before_s=10, after_s=10)
        wide = {s["cue_index"]: s["events"] for s in self.slices()}
        self.run_align(R7_CSV, before_s=10, after_s=0)
        for s in self.slices():
            self.assertTrue(all(e["rel_ms"] <= s["rel_start_ms"] for e in s["events"]))
        n10 = sum(len(s["events"]) for s in self.slices())
        self.assertGreater(sum(map(len, wide.values())), n10)
        self.run_align(R7_CSV, before_s=30, after_s=0)
        self.assertGreater(sum(len(s["events"]) for s in self.slices()), n10)

    def test_offset(self):
        self.run_align(R7_CSV)
        base = {s["cue_index"]: s["rel_start_ms"] for s in self.slices()}
        self.run_align(R7_CSV, offset_s=2)
        for s in self.slices():
            if s["cue_index"] in base:
                self.assertEqual(s["rel_start_ms"] - base[s["cue_index"]], 2000)


class TestResearcher(Base):
    def test_researcher_dropped(self):
        make_vtt(self.tr, R7_REC, [(20, 21, "Dr Researcher: keep going"), (22, 23, "P: thinking"),
                                   (24, 25, "no label here"), (300, 301, "P: done")])
        res = self.run_align(R7_CSV, researchers=["Dr Researcher"])
        texts = [s["text"] for s in self.slices()]
        self.assertNotIn("keep going", texts)
        self.assertIn("thinking", texts)
        self.assertIn("no label here", texts)
        p = res["participants"][R7]
        self.assertEqual((p["n_researcher"], p["n_unlabeled"]), (1, 1))
        for f in os.listdir(self.out):
            with open(os.path.join(self.out, f), encoding="utf-8") as fh:
                self.assertNotIn("keep going", fh.read())


class TestDecoderCli(unittest.TestCase):
    def test_cli_still_runs(self):
        tmp = tempfile.mkdtemp()
        try:
            r = subprocess.run([sys.executable, os.path.join(ROOT, "src", "decode_kl.py"), PILOT_CSV, "--out", tmp],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("2 OK, 3 problem(s)", r.stdout)
            self.assertTrue(os.path.exists(os.path.join(tmp, "timeline_%s_p1.md" % PILOT)))
        finally:
            shutil.rmtree(tmp)


if __name__ == "__main__":
    unittest.main()
