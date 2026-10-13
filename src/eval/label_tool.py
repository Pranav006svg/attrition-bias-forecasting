"""Blind keyboard labelling tool for annotation/validation_sample.csv.

The sample holds 100 raw reason strings (34 rule / 33 llm / 33 knn tier,
shuffled; tiers are not shown). Shows one raw_reason at a time with a
progress counter; a single keypress records its category.
The file is saved after every keypress, so you can quit and resume any time:
labelling restarts at the first row without a category. Only the id,
raw_reason and category columns are read; no predicted label is ever shown.

Keys:
  1-9, 0, n, o  record a category and move on
  b             go back one row
  q             save and quit

Usage:
    python -m src.eval.label_tool [--file annotation/validation_sample.csv]
"""
import argparse
import csv
import os
import shutil
import sys
import textwrap
import time

KEYMAP = {
    "1": "ADVERSE_EVENT", "2": "LACK_OF_EFFICACY", "3": "DISEASE_PROGRESSION",
    "4": "DEATH", "5": "WITHDRAWAL_BY_SUBJECT", "6": "LOST_TO_FOLLOW_UP",
    "7": "PHYSICIAN_DECISION", "8": "PROTOCOL_VIOLATION", "9": "PRE_TREATMENT_EXCLUSION",
    "0": "SPONSOR_ADMINISTRATIVE", "n": "NOT_ATTRITION", "o": "OTHER_UNSPECIFIED",
}
MENU = """\
  1 ADVERSE_EVENT           2 LACK_OF_EFFICACY        3 DISEASE_PROGRESSION
  4 DEATH                   5 WITHDRAWAL_BY_SUBJECT   6 LOST_TO_FOLLOW_UP
  7 PHYSICIAN_DECISION      8 PROTOCOL_VIOLATION      9 PRE_TREATMENT_EXCLUSION
  0 SPONSOR_ADMINISTRATIVE  n NOT_ATTRITION           o OTHER_UNSPECIFIED

  b = go back one row,  q = save and quit"""

BOLD, CYAN, DIM, RESET = "\033[1m", "\033[96m", "\033[2m", "\033[0m"


def getch() -> str:
    """Read a single keypress without waiting for Enter."""
    if os.name == "nt":
        import msvcrt
        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):  # arrow/function key prefix: swallow the second code
            msvcrt.getwch()
            return ""
        return ch
    import termios
    import tty
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)
    for col in ("id", "raw_reason", "category"):
        if col not in fields:
            sys.exit(f"{path} is missing column '{col}'")
    return fields, rows


def _write(target, fields, rows):
    with open(target, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def save(path, fields, rows) -> bool:
    """Save atomically; on Windows, retry while another program (OneDrive sync, an
    editor) briefly locks the file, then fall back to an in-place write. Returns
    False if the main file could not be updated (labels remain in path + '.tmp')."""
    tmp = path + ".tmp"
    _write(tmp, fields, rows)
    for _ in range(20):
        try:
            os.replace(tmp, path)
            return True
        except PermissionError:
            time.sleep(0.25)
    try:
        _write(path, fields, rows)
        os.remove(tmp)
        return True
    except PermissionError:
        return False


def recover_tmp(path, fields, rows):
    """Use a leftover .tmp from an interrupted save if it keeps every saved label and adds more."""
    tmp = path + ".tmp"
    if not os.path.exists(tmp):
        return rows
    try:
        _, trows = load(tmp)
    except SystemExit:
        return rows
    same_rows = [(r["id"], r["raw_reason"]) for r in rows] == [(r["id"], r["raw_reason"]) for r in trows]
    keeps_saved = all(r["category"] == t["category"] for r, t in zip(rows, trows) if r["category"].strip())
    if same_rows and keeps_saved and sum(bool(t["category"].strip()) for t in trows) > \
            sum(bool(r["category"].strip()) for r in rows):
        if save(path, fields, trows):
            print("Recovered labels from an interrupted save.")
        return trows
    return rows


def render(i, rows, message=""):
    width = min(shutil.get_terminal_size((100, 30)).columns, 100)
    done = sum(1 for r in rows if r["category"].strip())
    r = rows[i]
    os.system("cls" if os.name == "nt" else "clear")
    print(f"{BOLD}Row {i + 1}/{len(rows)}{RESET}   {DIM}(labelled: {done}/{len(rows)}, id {r['id']}){RESET}")
    print("=" * width)
    print()
    for line in textwrap.wrap(r["raw_reason"] or "(empty)", width=width - 4) or [""]:
        print(f"  {BOLD}{CYAN}{line}{RESET}")
    print()
    print("=" * width)
    if r["category"].strip():
        print(f"{DIM}  Your current label for this row: {r['category']}{RESET}")
    else:
        print()
    print()
    print(MENU)
    print()
    if message:
        print(f"  {message}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="annotation/validation_sample.csv")
    args = ap.parse_args()
    if os.name == "nt":
        os.system("")  # enable ANSI escape codes in Windows terminals

    fields, rows = load(args.file)
    if not rows:
        sys.exit("No rows to label.")
    rows = recover_tmp(args.file, fields, rows)
    unlabelled = [k for k, r in enumerate(rows) if not r["category"].strip()]
    if not unlabelled:
        print("All rows already labelled. Done")
        return
    i, message = unlabelled[0], ""

    while True:
        render(i, rows, message)
        message = ""
        key = getch().lower()
        if key in ("q", "\x03"):  # q or Ctrl+C
            done = sum(1 for r in rows if r["category"].strip())
            if save(args.file, fields, rows):
                print(f"\nSaved {done}/{len(rows)} labels to {args.file}. Run again to resume.")
            else:
                print(f"\n{args.file} is locked by another program; {done} labels are kept in {args.file}.tmp "
                      "and will be recovered automatically on the next run.")
            return
        if key == "b":
            if i > 0:
                i -= 1
            else:
                message = "Already at the first row."
            continue
        if key not in KEYMAP:
            if key:
                message = f"Key '{key}' is not in the menu."
            continue

        rows[i]["category"] = KEYMAP[key]
        saved = save(args.file, fields, rows)
        if not saved:
            message = ("Warning: the file is locked by another program (OneDrive/editor?). Labels are kept "
                       "and will be saved on the next keypress.")
        remaining = [k for k, r in enumerate(rows) if not r["category"].strip()]
        if not remaining:
            os.system("cls" if os.name == "nt" else "clear")
            if not saved:
                print(f"All rows labelled, but {args.file} is locked; labels are in {args.file}.tmp. "
                      "Close the program locking it and run the tool again to finish saving.")
                return
            print(f"All {len(rows)} rows labelled and saved to {args.file}.")
            print("Done")
            return
        # Next row; past the end, jump to the first row still unlabelled.
        i = i + 1 if i + 1 < len(rows) else remaining[0]


if __name__ == "__main__":
    main()
