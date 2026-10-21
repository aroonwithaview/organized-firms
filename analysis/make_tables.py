"""Run every analysis script behind the proposal's Table 1, Figure 2, Section 3 and Appendix B, and write the results
to analysis/results/.

Usage, from anywhere:  python analysis/make_tables.py

The scripts read the recorded runs in data/runs/ and make no model call. Each is run as a separate process from the
repository root, with the arguments listed in STEPS. REPRODUCE.md says which output file and line holds each number.
A result file that already exists is compared with the one just written (line endings ignored), so a run of this
script also checks that the shipped results are what the code writes now. The exit status is 1 if a script fails.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = HERE / "results"

POOLED_ROOTS = "data/runs/pilot17;data/runs/followup17;data/runs/random17"

# (script, arguments, files that the script writes in analysis/results/)
STEPS = [
    ("pilot17_analysis.py", [], ["pilot17_results.txt"]),
    ("pilot17_analysis.py", [POOLED_ROOTS, "analysis/results/followup17_pooled_results.txt"],
     ["followup17_pooled_results.txt"]),
    ("followup17_primary.py", [], ["followup17_primary_results.txt"]),
    ("departures18_analysis.py", [], ["departures18_results.txt", "departures18_results_message_sample.md"]),
    ("replication18_analysis.py", [], ["replication18_results.txt"]),
    ("panel_reanalysis.py", [], ["panel_reanalysis_results.txt"]),
    ("pooled40_numbers.py", [], ["pooled40_results.txt"]),
    ("departures18_extra_checks.py", [], ["departures18_extra_checks.txt"]),
    ("departures18_floor_bunching.py", [], ["departures18_floor_bunching.txt"]),
    ("departures18_check/adversarial.py", [], ["adversarial_out.txt"]),
    ("efficient_surplus.py", [], ["efficient_surplus.txt"]),
    ("market_combinations.py", [], ["market_combinations.txt"]),
    ("opening_lower_with_record.py", [], ["opening_lower_with_record.txt"]),
    ("confirm17_analysis.py", [], ["confirm17_results.txt"]),
]


def read_lf(path):
    """The bytes of a file with CRLF read as LF, or None if it does not exist."""
    return path.read_bytes().replace(b"\r\n", b"\n") if path.exists() else None


def main():
    env = dict(os.environ, PYTHONUTF8="1")  # the scripts print non-ASCII text; a redirected stream needs UTF-8
    failed = []
    for script, args, outputs in STEPS:
        before = {name: read_lf(RESULTS / name) for name in outputs}
        started = time.time()
        done = subprocess.run([sys.executable, str(HERE / script), *args], cwd=ROOT, env=env,
                              capture_output=True, text=True, encoding="utf-8", stdin=subprocess.DEVNULL)
        seconds = time.time() - started
        if done.returncode != 0:
            failed.append(script)
            print(f"FAILED  {script} {' '.join(args)} (exit {done.returncode})\n{done.stderr[-2000:]}")
            continue
        for name in outputs:
            after = read_lf(RESULTS / name)
            if after is None:
                failed.append(f"{script}: {name} was not written")
                print(f"MISSING {name}")
                continue
            state = ("new file" if before[name] is None else
                     "identical to the file that was there" if before[name] == after else "DIFFERS from the file that was there")
            print(f"wrote   analysis/results/{name}  ({len(after)} bytes, {seconds:.0f} s; {state})")
    if failed:
        print(f"\n{len(failed)} step(s) failed: {failed}")
        return 1
    print(f"\nAll {len(STEPS)} steps ran. Results are in analysis/results/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
