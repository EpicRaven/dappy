"""
DAPPY - Synthetic Data Generator | By EpicRaven
Synthetic data with SDV, trained on a temporary private Kaggle kernel.

Usage (command):  dappy                    (interactive)
                  dappy --ui               (Streamlit web UI)
                  dappy --dataset owner/slug --method gan --rows 10000 --out ./out --yes
                  dappy --config dappy.yaml --yes
Usage (script):   python dappy.py
Usage (notebook): import dappy; dappy.run()

Requirements (local): `pip install kaggle pandas numpy`
Optional: rich (tables/progress), matplotlib + seaborn (charts), scipy (KS / chi-square),
pyarrow (parquet), openpyxl (Excel), streamlit (UI), pyyaml (yaml config),
sdv (only to re-sample from a saved model locally).
Kaggle credentials: ~/.kaggle/kaggle.json or KAGGLE_USERNAME / KAGGLE_KEY.

Interactive flow
----------------
1. Paste a Kaggle link (dataset / competition / notebook) or a file/folder path.
   Formats: csv, tsv, parquet, xlsx, json(l). Several files -> pick one, or several
   for a relational multi-table run (HMASynthesizer).
2. Type the target column (Enter = none / unsupervised). dappy reports whether it is
   a regression or a classification dataset, then offers the Streamlit UI.
3. Overview, column exclusion, missing-value policy, PII anonymisation, synthesizer
   (GAN / GaussianCopula / TVAE / Auto), rows, optional advanced options
   (constraints, hyperparameters, conditional sampling, train/test validation ...).
4. SDV runs on a temporary private Kaggle kernel (deleted afterwards, even on Ctrl-C).
5. Local checks (threshold warnings, KS / chi-square, leakage, distance-to-closest-record),
   optional charts, then deliver: upload to Kaggle (new dataset or new version) or save
   locally with optional Parquet / Excel export, HTML report and zip bundle.

Session logs: ~/.dappy/logs/   Last-run settings: ~/.dappy/last_run.json
"""
from __future__ import annotations

import argparse
import base64
import builtins
import contextlib
import difflib
import html as _html
import io
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import uuid
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:  # optional: prettier CLI output (pip install rich). Falls back to plain text.
    from rich import box
    from rich.console import Console
    from rich.markup import escape
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    console = Console()
    RICH = True
except ImportError:
    RICH = False

__all__ = ["KaggleKernel", "run", "run_pipeline", "compare", "main"]

APP_TITLE = "DAPPY"
APP_TAGLINE = "Synthetic Data Generator | By EpicRaven"
HOME = Path.home() / ".dappy"

_LOG_FH = None
_QUIET_LOG = False          # True while a live progress display owns the terminal
_NONINTERACTIVE = False     # True for --yes / config / UI runs: prompting is an error
_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


class _Tee:
    """Mirror stdout into the session log (ANSI codes stripped)."""

    def __init__(self, stream):
        self.stream = stream

    def write(self, s):
        n = self.stream.write(s)
        if _LOG_FH is not None and not _QUIET_LOG:
            _LOG_FH.write(_ANSI.sub("", s))
        return n

    def flush(self):
        self.stream.flush()
        if _LOG_FH is not None:
            _LOG_FH.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def _log(msg: str):
    if _LOG_FH is not None:
        _LOG_FH.write(_ANSI.sub("", msg) + "\n")
        _LOG_FH.flush()


def _start_session_log() -> Path | None:
    global _LOG_FH
    if _LOG_FH is not None:
        return None
    try:
        (HOME / "logs").mkdir(parents=True, exist_ok=True)
        path = HOME / "logs" / f"dappy_{datetime.now():%Y%m%d_%H%M%S}.log"
        _LOG_FH = open(path, "a", encoding="utf-8")
    except OSError:
        return None
    if not isinstance(sys.stdout, _Tee):
        sys.stdout = _Tee(sys.stdout)
    _log(f"# dappy session {datetime.now().isoformat(timespec='seconds')}")
    return path


def input(prompt: str = "") -> str:  # noqa: A001 - shadows the builtin so every answer is logged
    if _NONINTERACTIVE:
        raise RuntimeError(f"Non-interactive run needs a value for: {prompt.strip() or 'a prompt'} "
                           "(set it with a flag or in the config file).")
    ans = builtins.input(prompt)
    _log(f"> {ans}")
    return ans


BANNER = r"""
██████╗  █████╗ ██████╗ ██████╗ ██╗   ██╗
██╔══██╗██╔══██╗██╔══██╗██╔══██╗╚██╗ ██╔╝
██║  ██║███████║██████╔╝██████╔╝ ╚████╔╝ 
██║  ██║██╔══██║██╔═══╝ ██╔═══╝   ╚██╔╝  
██████╔╝██║  ██║██║     ██║        ██║   
╚═════╝ ╚═╝  ╚═╝╚═╝     ╚═╝        ╚═╝   
"""

METHODS = [
    ("gan", "GAN (CTGAN)"),
    ("copula", "GaussianCopula"),
    ("tvae", "TVAE"),
]


def _banner():
    try:  # box characters need UTF-8 (matters on Windows consoles)
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    if RICH:
        console.print(Text(BANNER, style="bold cyan"))
        console.print(Text("        Synthetic Data Generator |  By EpicRaven\n", style="bold magenta"))
    else:
        print(BANNER)
        print("        Synthetic data with SDV  |  By EpicRaven\n")
        print("        (tip: pip install rich for prettier tables)\n")



def _fmt(v) -> str:
    if isinstance(v, (bool, np.bool_)):
        return str(v)
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    if isinstance(v, (float, np.floating)):
        return "-" if pd.isna(v) else f"{v:,.2f}"
    return "-" if v is None else str(v)


def show_table(df: pd.DataFrame, title: str | None = None, index: bool = True):
    """Render a DataFrame as a styled table (plain text if `rich` isn't installed)."""
    if isinstance(df, pd.Series):
        df = df.to_frame()
    if not RICH:
        if title:
            print(f"\n{title}")
        print(df.to_string(index=index))
        return
    table = Table(title=title, box=box.ROUNDED, header_style="bold cyan",
                  title_style="bold magenta", row_styles=["", "dim"], title_justify="left")
    if index:
        table.add_column(escape(str(df.index.name or "")), style="bold yellow", no_wrap=True)
    for col in df.columns:
        name = "\n".join(map(str, col)) if isinstance(col, tuple) else str(col)
        numeric = pd.api.types.is_numeric_dtype(df[col]) if not isinstance(col, tuple) else True
        if numeric:  # never split numbers across lines
            table.add_column(escape(name), justify="right", no_wrap=True)
        else:
            table.add_column(escape(name), justify="left", overflow="fold")
    for idx, row in df.iterrows():
        cells = [escape(_fmt(v)) for v in row]
        if index:
            label = " x ".join(map(str, idx)) if isinstance(idx, tuple) else str(idx)
            cells.insert(0, escape(label))
        table.add_row(*cells)
    console.print(table)


# --------------------------------------------------------------------------- #
# Kaggle CLI helpers
# --------------------------------------------------------------------------- #
def _kaggle_bin() -> str:
    """Locate the Kaggle CLI: PATH first, then next to the running Python (venv / Scripts folder)."""
    found = shutil.which("kaggle")
    if found:
        return found
    exe_dir = Path(sys.executable).resolve().parent
    for d in (exe_dir, exe_dir / "Scripts", exe_dir / "bin"):
        for name in ("kaggle.exe", "kaggle"):
            if (d / name).is_file():
                return str(d / name)
    raise RuntimeError(
        "Kaggle CLI not found. Install it into the same Python you run dappy with:\n"
        f"    \"{sys.executable}\" -m pip install kaggle\n"
        "then check it with `kaggle --version` (if that fails, add Python's Scripts folder to PATH)."
    )


def _cli(*args: str, check: bool = True, stdin: str | None = None) -> subprocess.CompletedProcess:
    proc = subprocess.run([_kaggle_bin(), *args], capture_output=True, text=True, input=stdin)
    if check and proc.returncode != 0:
        raise RuntimeError(f"kaggle {' '.join(args)} failed:\n{proc.stdout}\n{proc.stderr}")
    return proc


def _cli_delete(kind: str, ref: str) -> bool:
    """Delete a kernel/dataset. Tries `-y`, then falls back to answering the prompt."""
    for extra, stdin in ((["-y"], None), ([], "y\n")):
        try:
            if _cli(kind, "delete", ref, *extra, check=False, stdin=stdin).returncode == 0:
                return True
        except Exception:
            pass
    return False


def _wait_dataset_ready(ref: str, timeout: int = 600):
    """Poll `kaggle datasets status` until the dataset finishes processing."""
    start = time.time()
    while True:
        out = _cli("datasets", "status", ref, check=False).stdout.lower()
        if "ready" in out:
            return
        if "error" in out or "fail" in out:
            raise RuntimeError(f"Dataset processing failed: {out.strip()}")
        if time.time() - start > timeout:
            raise TimeoutError(f"Dataset processing exceeded {timeout // 60} minutes.")
        print("[dappy] processing upload...")
        time.sleep(5)



def _dataset_exists(ref: str) -> bool:
    p = _cli("datasets", "status", ref, check=False)
    out = (p.stdout + p.stderr).lower()
    return p.returncode == 0 and not any(w in out for w in ("404", "not found", "error"))


# --------------------------------------------------------------------------- #
# Code that executes REMOTELY on the Kaggle kernel
# It runs in a fresh interpreter (so a pip install can't break already-imported
# numpy/pandas) and reads its settings from /tmp/dappy_cfg.json.
# --------------------------------------------------------------------------- #
_JOB = r'''
import glob, json, os, time, traceback
import pandas as pd

CFG = json.load(open("/tmp/dappy_cfg.json"))
OUT = "/kaggle/working"
HP = CFG.get("hp") or {}
METHOD = CFG["method"]
EPOCHS = int(CFG.get("epochs", 300))
T0 = time.time()


def find(fname):
    hits = [f for f in glob.glob("/kaggle/input/**/*", recursive=True) if os.path.basename(f) == fname]
    if not hits:
        raise SystemExit("File not found in the attached data: " + fname)
    return hits[0]


tables = {}
for _name, _fname in CFG["tables"].items():
    tables[_name] = pd.read_csv(find(_fname))
    print("Loaded", _name, tables[_name].shape, flush=True)
MULTI = len(tables) > 1
NAME = next(iter(tables))

try:  # newer SDV: Metadata class (SingleTableMetadata is deprecated and breaks evaluation)
    from sdv.metadata import Metadata
except ImportError:
    Metadata = None
    from sdv.metadata import SingleTableMetadata

if MULTI:
    from sdv.evaluation.multi_table import evaluate_quality
else:
    from sdv.evaluation.single_table import evaluate_quality
try:
    if MULTI:
        from sdv.evaluation.multi_table import run_diagnostic
    else:
        from sdv.evaluation.single_table import run_diagnostic
except ImportError:
    run_diagnostic = None


def single_metadata(df, name):
    if Metadata is not None:
        md = Metadata.detect_from_dataframe(df, table_name=name)
    else:
        md = SingleTableMetadata()
        md.detect_from_dataframe(df)
    for col, sdtype in (CFG.get("pii") or {}).items():  # anonymisation preset: Faker-generated values
        if col not in df.columns:
            continue
        try:
            if Metadata is not None:
                md.update_column(column_name=col, table_name=name, sdtype=sdtype, pii=True)
            else:
                md.update_column(column_name=col, sdtype=sdtype, pii=True)
            print("Anonymising", col, "as", sdtype, flush=True)
        except Exception as e:
            print("Could not anonymise", col, "->", e, flush=True)
    return md


def multi_metadata():
    md = Metadata()
    if hasattr(md, "detect_table_from_dataframe"):
        for n, df in tables.items():
            md.detect_table_from_dataframe(n, df)
    else:
        md = Metadata.detect_from_dataframes(tables)
    for t, pk in (CFG.get("primary_keys") or {}).items():
        md.update_column(column_name=pk, table_name=t, sdtype="id")
        md.set_primary_key(pk, table_name=t)
    for r in CFG.get("relationships") or []:
        md.update_column(column_name=r["child_fk"], table_name=r["child"], sdtype="id")
        try:
            md.add_relationship(parent_table_name=r["parent"], child_table_name=r["child"],
                                parent_primary_key=r["parent_pk"], child_foreign_key=r["child_fk"])
        except Exception as e:  # e.g. already auto-detected
            print("Relationship note:", e, flush=True)
    md.validate()
    return md


def constraints():
    out = []
    for c in CFG.get("constraints") or []:
        t = c["type"]
        if t == "Inequality":
            p = {"low_column_name": c["low"], "high_column_name": c["high"],
                 "strict_boundaries": bool(c.get("strict", False))}
        elif t == "ScalarInequality":
            p = {"column_name": c["column"], "relation": c["relation"], "value": c["value"]}
        else:  # FixedCombinations, Unique
            p = {"column_names": c["columns"]}
        out.append({"constraint_class": t, "constraint_parameters": p})
    return out


def build(method, md, epochs):
    if MULTI:
        from sdv.multi_table import HMASynthesizer
        return HMASynthesizer(md)
    if method == "gan":
        from sdv.single_table import CTGANSynthesizer
        kw = {k: HP[k] for k in ("batch_size", "generator_lr", "discriminator_lr",
                                 "discriminator_steps", "embedding_dim") if k in HP}
        s = CTGANSynthesizer(md, epochs=epochs, verbose=True, **kw)
    elif method == "tvae":
        from sdv.single_table import TVAESynthesizer
        kw = {k: HP[k] for k in ("batch_size", "embedding_dim") if k in HP}
        s = TVAESynthesizer(md, epochs=epochs, **kw)
    else:
        from sdv.single_table import GaussianCopulaSynthesizer
        kw = {k: HP[k] for k in ("default_distribution",) if k in HP}
        s = GaussianCopulaSynthesizer(md, **kw)
    cons = constraints()
    if cons:
        s.add_constraints(cons)
    return s


def details(rep, prop):
    try:
        return rep.get_details(prop).to_dict("records")
    except Exception:
        return []


# ---- metadata + optional train / held-out split -------------------------------------------
test = None
if MULTI:
    metadata = multi_metadata()
    train = tables
else:
    real = tables[NAME]
    metadata = single_metadata(real, NAME)
    train = real
    if CFG.get("holdout") and len(real) >= 50:
        test = real.sample(frac=0.2, random_state=0)
        train = real.drop(test.index).reset_index(drop=True)
        test = test.reset_index(drop=True)
        print("Holdout: fitting on", len(train), "rows, validating on", len(test), flush=True)

# ---- choose the method -----------------------------------------------------------------------
auto_scores = None
if MULTI:
    method = "hma"
elif METHOD == "auto":
    sample = train.sample(min(len(train), int(CFG.get("auto_rows", 2000))), random_state=0)
    auto_scores = {}
    for m in ("copula", "gan", "tvae"):
        try:
            print("Auto-select: trying", m, "on", len(sample), "rows ...", flush=True)
            trial = build(m, metadata, min(EPOCHS, 50))
            trial.fit(sample)
            f = trial.sample(num_rows=len(sample))
            auto_scores[m] = float(evaluate_quality(sample, f, metadata, verbose=False).get_score())
            print("  ", m, "quality:", round(auto_scores[m], 4), flush=True)
        except Exception:
            traceback.print_exc()
            auto_scores[m] = None
    valid = {k: v for k, v in auto_scores.items() if v is not None}
    if not valid:
        raise SystemExit("Auto-select: every synthesizer failed (see log above).")
    method = max(valid, key=valid.get)
    print("Auto-select winner:", method, flush=True)
else:
    method = METHOD

# ---- fit + sample ---------------------------------------------------------------------------------
print("Fitting synthesizer (" + method + ") ...", flush=True)
T_FIT = time.time()
synth = build(method, metadata, EPOCHS)
synth.fit(train)
fit_seconds = time.time() - T_FIT

print("Sampling...", flush=True)
if MULTI:
    fake = synth.sample(scale=float(CFG.get("scale", 1.0)))
elif CFG.get("conditions"):
    from sdv.sampling import Condition
    conds = [Condition(num_rows=int(c["num_rows"]), column_values=c["column_values"])
             for c in CFG["conditions"]]
    fake = synth.sample_from_conditions(conds)
else:
    fake = synth.sample(num_rows=int(CFG["rows"]))

# ---- evaluate -------------------------------------------------------------------------------------
print("Evaluating quality...", flush=True)
eval_real = tables if MULTI else (test if test is not None else tables[NAME])
report = evaluate_quality(eval_real, fake, metadata, verbose=False)
quality = {
    "overall_score": float(report.get_score()),
    "properties": report.get_properties().to_dict("records"),
    "column_shapes": details(report, "Column Shapes"),
}
if test is not None:
    r2 = evaluate_quality(train, fake, metadata, verbose=False)
    quality["holdout"] = {"vs_train": float(r2.get_score()), "vs_test": quality["overall_score"]}

diagnostic = None
if run_diagnostic is not None:
    try:
        d = run_diagnostic(eval_real, fake, metadata, verbose=False)
        diagnostic = {"properties": d.get_properties().to_dict("records")}
    except Exception as e:
        print("Diagnostic skipped:", e, flush=True)

# ---- write outputs --------------------------------------------------------------------------------
if MULTI:
    for n, df in fake.items():
        df.to_csv(OUT + "/synthetic_" + n + ".csv", index=False)
else:
    fake.to_csv(OUT + "/synthetic.csv", index=False)
with open(OUT + "/quality.json", "w") as f:
    json.dump(quality, f, indent=2, default=str)
if diagnostic:
    with open(OUT + "/diagnostic.json", "w") as f:
        json.dump(diagnostic, f, indent=2, default=str)
try:
    synth.save(OUT + "/synthesizer.pkl")
except Exception as e:
    print("Could not save the fitted synthesizer:", e, flush=True)
try:
    import sdv
    sdv_version = sdv.__version__
except Exception:
    sdv_version = None
with open(OUT + "/run_meta.json", "w") as f:
    json.dump({"method": method, "auto_scores": auto_scores, "hyperparameters": HP, "epochs": EPOCHS,
               "fit_seconds": round(fit_seconds, 1), "total_seconds": round(time.time() - T0, 1),
               "sdv_version": sdv_version, "holdout": test is not None}, f, indent=2, default=str)
print("DAPPY_DONE", flush=True)
'''


def _build_launcher(cfg: dict) -> str:
    return (
        "import importlib.util, json, subprocess, sys\n"
        f"CFG = {cfg!r}\n"
        f"JOB = {_JOB!r}\n"
        "if importlib.util.find_spec('sdv') is None:\n"
        "    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'sdv'])\n"
        "open('/tmp/dappy_cfg.json', 'w').write(json.dumps(CFG))\n"
        "open('/tmp/dappy_job.py', 'w').write(JOB)\n"
        "subprocess.check_call([sys.executable, '/tmp/dappy_job.py'])\n"
    )



class _Progress:
    """Spinner + elapsed time (rich, on a real terminal) or plain lines (otherwise)."""

    def __init__(self):
        self.live = RICH and console.is_terminal
        self.p = None

    def __enter__(self):
        global _QUIET_LOG
        if self.live:
            from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
            _QUIET_LOG = True
            self.p = Progress(SpinnerColumn(), TextColumn("[cyan]{task.description}"),
                              TimeElapsedColumn(), console=console, transient=True)
            self.task = self.p.add_task("starting ...", total=None)
            self.p.start()
        return self

    def __exit__(self, *exc):
        global _QUIET_LOG
        if self.p is not None:
            self.p.stop()
            _QUIET_LOG = False
        return False

    def update(self, text: str):
        _log(f"[dappy] {text}")
        if self.p is not None:
            self.p.update(self.task, description=text)
        else:
            print(f"[dappy] {text}")

    def line(self, text: str):
        _log(f"  | {text}")
        if self.p is not None:
            self.p.console.print(f"[dim]  | {escape(text)}[/dim]")
        else:
            print(f"  | {text}")


# --------------------------------------------------------------------------- #
# Context manager: owns the kernel (and temp dataset) lifecycle
# --------------------------------------------------------------------------- #
class KaggleKernel:
    """
    with KaggleKernel(dataset="owner/slug", data_file="x.csv") as k:   # attach a Kaggle dataset
        k.execute(method="gan", rows=1000)

    with KaggleKernel(local_files=["a.csv", "b.csv"]) as k:            # upload local files privately
        k.run_job(method="copula", rows=1000)

    - Remembers the local directory on entry and restores it on exit.
    - On exit (normal, error, or Ctrl-C) deletes the kernel, the temporary
      dataset (local mode) and temp files. Ctrl-C is ignored during cleanup.
    """

    def __init__(self, dataset: str | None = None, local_csv: str | os.PathLike | None = None,
                 data_file: str | None = None, out_dir: str | os.PathLike | None = None,
                 gpu: bool = False, poll_seconds: int = 10, timeout: int = 3600,
                 local_files: list | None = None, accelerator: str | None = None,
                 stream_logs: bool = True):
        files = [Path(f) for f in (local_files or [])]
        if local_csv:
            files.append(Path(local_csv))
        if bool(dataset) == bool(files):
            raise ValueError("Provide exactly one of `dataset` or `local_csv` / `local_files`.")
        self.dataset = dataset
        self.local_files = [f.expanduser().resolve() for f in files]
        for f in self.local_files:
            if not f.is_file():
                raise FileNotFoundError(f"File not found: {f}")
        if self.local_files:
            self.data_files = [f.name for f in self.local_files]
        else:
            self.data_files = [data_file] if data_file else []
        self.gpu = gpu
        self.accelerator = accelerator
        self.poll_seconds = poll_seconds
        self.timeout = timeout
        self.stream_logs = stream_logs
        self.origin = Path.cwd()  # <- the local directory we always come back to
        self.out_dir = Path(out_dir) if out_dir else self.origin / "dappy_output"
        self.username = self._get_username()
        self.slug = f"dappy-{uuid.uuid4().hex[:8]}"
        self.ref = f"{self.username}/{self.slug}"
        self.upload_ref: str | None = None
        self._pushed = False
        self._tmp: Path | None = None

    @staticmethod
    def _get_username() -> str:
        if os.environ.get("KAGGLE_USERNAME"):
            return os.environ["KAGGLE_USERNAME"]
        cfg = Path.home() / ".kaggle" / "kaggle.json"
        if cfg.exists():
            return json.loads(cfg.read_text())["username"]
        return input("Kaggle username: ").strip()

    # -- lifecycle ---------------------------------------------------------- #
    def __enter__(self) -> "KaggleKernel":
        _kaggle_bin()  # raises a clear error if the CLI is missing
        self._tmp = Path(tempfile.mkdtemp(prefix="dappy_"))
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None and issubclass(exc_type, KeyboardInterrupt):
            print("\n[dappy] Interrupted. Cleaning up Kaggle resources (please wait) ...")

        prev_handler = None
        try:  # a second Ctrl-C must not abort the cleanup
            prev_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
        except Exception:
            pass
        try:
            if self._pushed and not _cli_delete("kernels", self.ref):
                print(f"[dappy] Could not auto-delete kernel {self.ref}. "
                      f"Remove it manually: https://www.kaggle.com/code/{self.ref}")
            if self.upload_ref and not _cli_delete("datasets", self.upload_ref):
                print(f"[dappy] Could not auto-delete private dataset {self.upload_ref}. "
                      f"Remove it manually: https://www.kaggle.com/datasets/{self.upload_ref}")
        finally:
            if prev_handler is not None:
                try:
                    signal.signal(signal.SIGINT, prev_handler)
                except Exception:
                    pass
            if self._tmp and self._tmp.exists():
                shutil.rmtree(self._tmp, ignore_errors=True)
            os.chdir(self.origin)  # bypass: always return to the local directory
        return False  # never swallow exceptions

    # -- local files -> private Kaggle dataset ------------------------------ #
    def _upload_local(self) -> str:
        assert self._tmp is not None
        stage = self._tmp / "upload"
        stage.mkdir()
        for f in self.local_files:
            shutil.copy(f, stage / f.name)

        slug = f"dappy-data-{self.slug[-8:]}"
        self.upload_ref = f"{self.username}/{slug}"  # set first so cleanup covers Ctrl-C mid-upload
        (stage / "dataset-metadata.json").write_text(json.dumps({
            "title": slug,
            "id": self.upload_ref,
            "licenses": [{"name": "CC0-1.0"}],
        }))

        names = ", ".join(f.name for f in self.local_files)
        print(f"[dappy] Uploading {names} as a private dataset ({self.upload_ref}) ...")
        _cli("datasets", "create", "-p", str(stage))  # private by default
        _wait_dataset_ready(self.upload_ref)
        return self.upload_ref

    @property
    def stage_dir(self) -> Path:
        """Temporary folder (deleted on exit) for results that should not land in `out_dir`."""
        assert self._tmp is not None, "Use KaggleKernel inside a `with` block."
        return self._tmp / "results"

    # -- main action -------------------------------------------------------- #
    def execute(self, method: str, rows: int, epochs: int = 300, **extra) -> Path:
        """Run the job and download the results into `out_dir`."""
        self.run_job(method=method, rows=rows, epochs=epochs, **extra)
        return self.fetch(self.out_dir)

    def fetch(self, dest: str | os.PathLike) -> Path:
        """Download the finished kernel's output files into `dest`."""
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        print("[dappy] Downloading results ...")
        _cli("kernels", "output", self.ref, "-p", str(dest))
        return dest

    def run_job(self, method: str, rows: int, epochs: int = 300, **extra) -> None:
        """Push the kernel and wait for it to finish. Does NOT download anything.

        `extra` is merged into the remote config: hp, scale, tables, relationships,
        primary_keys, constraints, pii, conditions, holdout, auto_rows.
        """
        assert self._tmp is not None, "Use KaggleKernel inside a `with` block."
        source = self.dataset or self._upload_local()

        cfg = {"method": method, "rows": int(rows), "epochs": int(epochs)}
        cfg.update(extra)
        if "tables" not in cfg:
            cfg["tables"] = {re.sub(r"\W+", "_", Path(f).stem): f for f in self.data_files}
        kdir = self._tmp / "kernel"
        kdir.mkdir()
        (kdir / "script.py").write_text(_build_launcher(cfg))
        gpu = self.gpu and (method in {"gan", "tvae", "auto"} or bool(self.accelerator))
        (kdir / "kernel-metadata.json").write_text(json.dumps({
            "id": self.ref,
            "title": self.slug,
            "code_file": "script.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": str(gpu).lower(),
            "enable_internet": "true",
            "dataset_sources": [source],
        }, indent=2))

        print(f"[dappy] Pushing kernel {self.ref} ...")
        self._pushed = True  # set first so cleanup covers Ctrl-C mid-push
        if self.accelerator and gpu:
            proc = _cli("kernels", "push", "-p", str(kdir), "--accelerator", self.accelerator, check=False)
            if proc.returncode != 0:
                print("[dappy] This Kaggle CLI rejected --accelerator; using the default GPU instead.")
                _cli("kernels", "push", "-p", str(kdir))
        else:
            _cli("kernels", "push", "-p", str(kdir))

        start = time.time()
        seen = [0]
        last_log = 0.0
        with _Progress() as prog:
            while True:
                out = _cli("kernels", "status", self.ref, check=False).stdout.lower()
                if "complete" in out:
                    break
                if "error" in out or "cancel" in out:
                    if prog.p is not None:
                        prog.p.stop()
                    self._show_failure_log()
                    raise RuntimeError(f"Kernel failed. Status: {out.strip()}")
                if time.time() - start > self.timeout:
                    raise TimeoutError(f"Kernel run exceeded the {self.timeout // 60}-minute timeout.")
                prog.update(f"running on Kaggle ({int(time.time() - start)}s)")
                if self.stream_logs and time.time() - last_log > 30:
                    last_log = time.time()
                    for line in self._live_log(seen)[-12:]:
                        prog.line(line)
                time.sleep(self.poll_seconds)

        print("[dappy] Kernel finished.")

    @staticmethod
    def _log_text(path: Path) -> str:
        text = path.read_text(errors="ignore")
        try:  # Kaggle logs are usually a JSON list of {"data": ...} chunks
            return "".join(e.get("data", "") for e in json.loads(text))
        except Exception:
            return text

    def _live_log(self, seen: list) -> list[str]:
        """Best effort: pull the kernel log while it runs and return the lines not shown yet."""
        try:
            assert self._tmp is not None
            d = self._tmp / "livelog"
            d.mkdir(exist_ok=True)
            _cli("kernels", "output", self.ref, "-p", str(d), check=False)
            logs = sorted(d.glob("*.log"))
            if not logs:
                return []
            lines = [ln for ln in self._log_text(logs[-1]).splitlines() if ln.strip()]
            new = lines[seen[0]:]
            seen[0] = len(lines)
            return new
        except Exception:
            return []

    def _show_failure_log(self, tail: int = 40):
        """Download the kernel log and print its last lines so the real error is visible."""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        _cli("kernels", "output", self.ref, "-p", str(self.out_dir), check=False)
        logs = sorted(self.out_dir.glob(f"{self.slug}*.log")) or sorted(self.out_dir.glob("*.log"))
        if not logs:
            print("[dappy] No log file was returned by Kaggle.")
            return
        text = self._log_text(logs[-1])
        print(f"\n---- kernel log (last {tail} lines) ----")
        print("\n".join(text.splitlines()[-tail:]))
        print(f"---- full log: {logs[-1]} ----\n")


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #
def _menu(title: str, options: list[str]) -> int:
    """Numbered list; keeps asking until a valid number is entered. Returns 0-based index."""
    print(f"\n{title}")
    for i, opt in enumerate(options, 1):
        print(f"  [{i}] {opt}")
    while True:
        raw = input(f"Choose (1-{len(options)}): ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return int(raw) - 1
        shown = raw if len(raw) <= 40 else raw[:37] + "..."
        print(f"[dappy] Invalid choice '{shown}'. Enter a number from 1 to {len(options)}.")


def _ask_rows(n_original: int) -> int:
    pick = _menu("Rows for the synthetic dataset:",
                 [f"Same as original ({n_original:,} rows)", "Custom number of rows"])
    if pick == 0:
        return n_original
    while True:
        raw = input("Number of rows: ").strip().replace(",", "")
        if raw.isdigit() and int(raw) > 0:
            return int(raw)
        print("[dappy] Enter a positive whole number.")


def _ask_yes_no(question: str, default: bool = False) -> bool:
    hint = "Y/n" if default else "y/N"
    while True:
        raw = input(f"{question} [{hint}]: ").strip().lower()
        if not raw:
            return default
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        print("[dappy] Please answer y or n.")


def _ask_text(prompt: str, default: str = "", min_len: int = 0, max_len: int = 100_000,
              optional: bool = False) -> str:
    """Prompt until the answer's length is within limits.

    Blank input returns `default` if one is given, "" if the field is optional
    (or has no minimum length), otherwise it is rejected.
    """
    hint = f" [{default}]" if default else ""
    while True:
        raw = input(f"{prompt}{hint}: ").strip()
        if not raw:
            if default:
                return default
            if optional or min_len == 0:
                return ""
        if min_len <= len(raw) <= max_len:
            return raw
        print(f"[dappy] Enter {min_len}-{max_len} characters (you typed {len(raw)}).")



def _ask_number(prompt: str, default, cast=int, lo=None, hi=None, multiple_of=None):
    while True:
        raw = input(f"{prompt} [{default}]: ").strip().replace(",", "")
        if not raw:
            return default
        try:
            v = cast(raw)
        except ValueError:
            print(f"[dappy] Enter a number ({cast.__name__}).")
            continue
        if (lo is not None and v < lo) or (hi is not None and v > hi):
            print(f"[dappy] Must be between {lo} and {hi}.")
        elif multiple_of and v % multiple_of:
            print(f"[dappy] Must be a multiple of {multiple_of}.")
        else:
            return v


@contextlib.contextmanager
def _tab_complete(options: list[str]):
    """Tab-completion (readline) for column names; silently a no-op where readline is missing."""
    try:
        import readline
    except ImportError:
        yield
        return
    old, old_delims = readline.get_completer(), readline.get_completer_delims()
    matches: list[str] = []

    def completer(text, state):
        if state == 0:
            t = text.strip().lower()
            matches[:] = ([o for o in options if o.lower().startswith(t)]
                          or [o for o in options if t in o.lower()])
        return matches[state] if state < len(matches) else None

    readline.set_completer(completer)
    readline.set_completer_delims("\n,")
    readline.parse_and_bind("bind ^I rl_complete" if "libedit" in (readline.__doc__ or "")
                            else "tab: complete")
    try:
        yield
    finally:
        readline.set_completer(old)
        readline.set_completer_delims(old_delims)


def _resolve_column(tok: str, cols: list[str]) -> str | None:
    """Exact / case-insensitive / 1-based index / fuzzy match of what the user typed."""
    tok = tok.strip().strip("\"'")
    if tok in cols:
        return tok
    low = {c.lower(): c for c in cols}
    if tok.lower() in low:
        return low[tok.lower()]
    if tok.isdigit() and 1 <= int(tok) <= len(cols):
        return cols[int(tok) - 1]
    close = difflib.get_close_matches(tok, cols, n=5, cutoff=0.5)
    close += [c for c in cols if tok.lower() in c.lower() and c not in close][:3]
    if not close:
        print(f"[dappy] No column like '{tok}'.")
        return None
    if len(close) == 1:
        return close[0] if _ask_yes_no(f"'{tok}' not found. Use '{close[0]}'?", True) else None
    pick = _menu(f"'{tok}' not found. Did you mean:", close + ["None of these"])
    return close[pick] if pick < len(close) else None


def _pick_column(cols: list[str], prompt: str, blank_ok: bool = True) -> str | None:
    with _tab_complete(cols):
        while True:
            raw = input(f"{prompt}: ").strip()
            if not raw:
                if blank_ok:
                    return None
                continue
            hit = _resolve_column(raw, cols)
            if hit:
                return hit


def _pick_columns(cols: list[str], prompt: str) -> list[str]:
    """Comma-separated column names (tab-complete + fuzzy). Blank = none."""
    with _tab_complete(cols):
        raw = input(f"{prompt}: ").strip()
    out: list[str] = []
    for tok in filter(None, (t.strip() for t in raw.split(","))):
        hit = _resolve_column(tok, cols)
        if hit and hit not in out:
            out.append(hit)
    return out


# --------------------------------------------------------------------------- #
# Reading tables / resolving sources (Kaggle dataset, competition, notebook, local)
# --------------------------------------------------------------------------- #
_EXTS = (".csv", ".tsv", ".parquet", ".pq", ".xlsx", ".xls", ".json", ".jsonl", ".ndjson")


def _read_table(p: Path) -> pd.DataFrame:
    s = p.suffix.lower()
    try:
        if s == ".csv":
            return pd.read_csv(p)
        if s == ".tsv":
            return pd.read_csv(p, sep="\t")
        if s in (".parquet", ".pq"):
            return pd.read_parquet(p)
        if s in (".xlsx", ".xls"):
            return pd.read_excel(p)
        if s in (".jsonl", ".ndjson"):
            return pd.read_json(p, lines=True)
        if s == ".json":
            try:
                return pd.read_json(p)
            except ValueError:
                return pd.read_json(p, lines=True)
    except ImportError as e:
        need = "pyarrow" if s in (".parquet", ".pq") else "openpyxl"
        raise RuntimeError(f"Reading {s} files needs {need}:  pip install {need}") from e
    raise ValueError(f"Unsupported file type: {p.name} (supported: {', '.join(_EXTS)})")


def _table_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in _EXTS
             and not p.name.startswith((".", "~$")) and p.name != "dataset-metadata.json"]
    return sorted(files, key=lambda p: p.stat().st_size, reverse=True)


def _detect_source(text: str) -> dict:
    """Turn whatever was pasted (Kaggle link, owner/slug, file or folder path) into a source dict."""
    t = text.strip().strip("\"'")
    m = re.search(r"kaggle\.com/(datasets|competitions|c|code|notebooks)/([^?#\s]+)", t)
    if m:
        where, parts = m.group(1), m.group(2).strip("/").split("/")
        if where == "datasets" and len(parts) >= 2:
            return {"kind": "dataset", "ref": f"{parts[0]}/{parts[1]}"}
        if where in ("competitions", "c"):
            return {"kind": "competition", "ref": parts[0]}
        if where in ("code", "notebooks") and len(parts) >= 2:
            return {"kind": "notebook", "ref": f"{parts[0]}/{parts[1]}"}
        raise ValueError("Could not read that Kaggle link.")
    p = Path(t).expanduser()
    if p.exists():
        return {"kind": "local", "ref": str(p.resolve())}
    if re.fullmatch(r"[\w.-]+/[\w.-]+", t):
        return {"kind": "dataset", "ref": t}
    raise ValueError("Not a Kaggle link (dataset / competition / notebook) and not an existing path.")


def _fetch_source(src: dict, tmp: Path) -> list[Path]:
    """Download (if needed) and return every readable table file, largest first."""
    kind, ref = src["kind"], src["ref"]
    if kind == "local":
        files = _table_files(Path(ref))
    else:
        d = tmp / f"src_{kind}_{re.sub(r'[^A-Za-z0-9]+', '_', ref)}"
        d.mkdir(exist_ok=True)
        print(f"[dappy] Downloading Kaggle {kind} {ref} locally ...")
        if kind == "dataset":
            _cli("datasets", "download", "-d", ref, "-p", str(d), "--unzip")
        elif kind == "competition":
            _cli("competitions", "download", "-c", ref, "-p", str(d))
            for z in d.glob("*.zip"):
                with zipfile.ZipFile(z) as zf:
                    zf.extractall(d)
        else:
            _cli("kernels", "output", ref, "-p", str(d))
        files = _table_files(d)
    if not files:
        raise RuntimeError(f"No readable table files found ({', '.join(_EXTS)}).")
    return files


def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n} B"


def _choose_files(paths: list[Path], interactive: bool, wanted: list[str] | None = None) -> list[Path]:
    if wanted:
        by_name = {p.name: p for p in paths}
        missing = [w for w in wanted if w not in by_name]
        if missing:
            raise RuntimeError(f"File(s) not found in the source: {', '.join(missing)}. "
                               f"Available: {', '.join(by_name)}")
        return [by_name[w] for w in wanted]
    if len(paths) == 1:
        return paths
    if not interactive:
        print(f"[dappy] {len(paths)} files found; using the largest: {paths[0].name} "
              "(use --file to choose, repeat it for a multi-table run)")
        return paths[:1]
    print(f"\n{len(paths)} files found:")
    for i, p in enumerate(paths, 1):
        print(f"  [{i}] {p.name}  ({_human_size(p.stat().st_size)})")
    print("Pick ONE file (single table) or several separated by commas, e.g. 1,3 "
          "(relational multi-table, HMASynthesizer). Enter = largest.")
    while True:
        raw = input("Files: ").strip()
        if not raw:
            return paths[:1]
        try:
            idx = [int(x) for x in raw.replace(" ", "").split(",") if x]
        except ValueError:
            idx = []
        if idx and all(1 <= i <= len(paths) for i in idx):
            return [paths[i - 1] for i in dict.fromkeys(idx)]
        print(f"[dappy] Enter numbers from 1 to {len(paths)}.")


def _load_tables(paths: list[Path]) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for p in paths:
        name = re.sub(r"\W+", "_", p.stem).strip("_") or "table"
        base, n = name, 2
        while name in tables:
            name, n = f"{base}_{n}", n + 1
        tables[name] = _read_table(p)
    return tables


def _open_source(tmp: Path, first: str | None = None):
    """Interactive: ask for a link / path. Returns (source, chosen_paths, tables)."""
    while True:
        text = first or input("\nPaste a Kaggle link (dataset / competition / notebook) "
                              "or a file / folder path: ").strip()
        first = None
        if not text:
            continue
        try:
            src = _detect_source(text)
            paths = _fetch_source(src, tmp)
            chosen = _choose_files(paths, interactive=True)
            return src, chosen, _load_tables(chosen)
        except (ValueError, RuntimeError, FileNotFoundError) as e:
            print(f"[dappy] {e}")
        except Exception as e:  # unreadable file etc.
            print(f"[dappy] Could not read that data: {e}")


# --------------------------------------------------------------------------- #
# Target column + regression / classification detection
# --------------------------------------------------------------------------- #
def _detect_task(df: pd.DataFrame, target: str | None) -> str:
    if not target or target not in df.columns:
        return "unsupervised"
    s = df[target].dropna()
    if s.empty:
        return "unsupervised"
    if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        return "classification"
    nun = s.nunique()
    integer_like = bool((s % 1 == 0).all())
    if integer_like and (nun <= 15 or (nun <= 50 and nun / len(s) < 0.05)):
        return "classification"
    return "regression"


def _describe_task(df: pd.DataFrame, target: str | None) -> str:
    task = _detect_task(df, target)
    if task == "unsupervised":
        return "Task type: unsupervised (no target column)"
    if task == "classification":
        return f"Task type: classification (target '{target}', {df[target].nunique()} classes)"
    s = df[target].dropna()
    return f"Task type: regression (target '{target}', range {s.min():,.4g} to {s.max():,.4g})"


def _ask_target(df: pd.DataFrame) -> str | None:
    cols = list(map(str, df.columns))
    target = _pick_column(cols, "Target column - type its name (Tab completes), "
                                "or press Enter if there is none (unsupervised)")
    line = _describe_task(df, target)
    if RICH:
        console.print(Panel(line, border_style="green", expand=False))
    else:
        print(f"\n{line}")
    return target


# --------------------------------------------------------------------------- #
# Multi-table relationships
# --------------------------------------------------------------------------- #
def _ask_relationships(tables: dict[str, pd.DataFrame]):
    """Returns (primary_keys {table: col}, relationships [{parent, parent_pk, child, child_fk}])."""
    names = list(tables)
    pks: dict[str, str] = {}
    print("\nMulti-table mode (HMASynthesizer). Define keys so child rows can be tied to parents.")
    for t, df in tables.items():
        cols = list(map(str, df.columns))
        unique = [c for c in cols if df[c].is_unique and df[c].notna().all()]
        hint = f" (unique columns: {', '.join(unique[:6])})" if unique else ""
        pk = _pick_column(cols, f"Primary key of '{t}'{hint} - Enter if none")
        if pk:
            pks[t] = pk
    rels: list[dict] = []
    parents = [t for t in names if t in pks]
    while parents and len(names) > 1 and _ask_yes_no("Add a foreign-key relationship?", default=not rels):
        parent = parents[_menu("Parent table (holds the primary key):", parents)]
        kids = [t for t in names if t != parent]
        child = kids[_menu("Child table (holds the foreign key):", kids)]
        fk = _pick_column(list(map(str, tables[child].columns)),
                          f"Column in '{child}' that points to '{parent}.{pks[parent]}'", blank_ok=False)
        rels.append({"parent": parent, "parent_pk": pks[parent], "child": child, "child_fk": fk})
    if not rels:
        print("[dappy] No relationships defined: tables will be modelled independently.")
    return pks, rels


# --------------------------------------------------------------------------- #
# Preprocessing: exclusion, missing values, row sampling, PII detection
# --------------------------------------------------------------------------- #
MISSING_POLICIES = ("keep", "impute", "drop")


def apply_missing_policy(df: pd.DataFrame, policy: str) -> tuple[pd.DataFrame, str]:
    n = int(df.isna().sum().sum())
    if n == 0:
        return df, "Missing values: none found, nothing to do."
    cols = int(df.isna().any().sum())
    if policy == "drop":
        out = df.dropna().reset_index(drop=True)
        return out, f"Missing values: dropped {len(df) - len(out):,} of {len(df):,} rows that had NaNs."
    if policy == "impute":
        out = df.copy()
        for c in out.columns[out.isna().any()]:
            if pd.api.types.is_numeric_dtype(out[c]):
                out[c] = out[c].fillna(out[c].median())
            else:
                mode = out[c].mode(dropna=True)
                out[c] = out[c].fillna(mode.iloc[0] if len(mode) else "unknown")
        return out, (f"Missing values: imputed {n:,} cells in {cols} column(s) "
                     "(median for numbers, most frequent value for text).")
    return df, f"Missing values: kept {n:,} NaN cells in {cols} column(s) as they are."


def prepare_tables(s: dict, tables: dict[str, pd.DataFrame]):
    """Apply exclusion -> missing-value policy -> row sampling. Returns (tables, notes)."""
    notes: list[str] = []
    multi = len(tables) > 1
    out: dict[str, pd.DataFrame] = {}
    exclude = [c for c in (s.get("exclude") or []) if c != s.get("target")]
    for name, df in tables.items():
        cur = df
        if exclude and not multi:
            drop = [c for c in exclude if c in cur.columns]
            unknown = [c for c in exclude if c not in cur.columns]
            if unknown:
                notes.append(f"Excluded columns not found (ignored): {', '.join(unknown)}")
            if drop:
                cur = cur.drop(columns=drop)
                notes.append(f"Excluded {len(drop)} column(s) before fitting: {', '.join(drop)}")
        cur, note = apply_missing_policy(cur, s.get("missing", "keep"))
        notes.append(note if not multi else f"[{name}] {note}")
        cap = s.get("train_rows")
        if cap and not multi and len(cur) > int(cap):
            cur = cur.sample(int(cap), random_state=42).reset_index(drop=True)
            notes.append(f"Row sampling: fitting on a random {int(cap):,} of {len(df):,} rows.")
        out[name] = cur
    if multi and (exclude or s.get("train_rows")):
        notes.append("Column exclusion and row sampling are skipped in multi-table mode "
                     "(they could break key relationships).")
    return out, notes


_PII_PATTERNS = [
    ("email", r"e[-_ ]?mail"),
    ("phone_number", r"phone|mobile|telephone|cell[-_ ]?(no|num)"),
    ("ssn", r"\bssn\b|social[-_ ]?security"),
    ("credit_card_number", r"credit[-_ ]?card|card[-_ ]?number"),
    ("ipv4_address", r"ip[-_ ]?address|^ip$"),
    ("first_name", r"^(first|given|f)[-_ ]?name$"),
    ("last_name", r"^(last|sur|family|l)[-_ ]?name$"),
    ("street_address", r"address|street"),
    ("city", r"^city$|^town$"),
]


def detect_pii(df: pd.DataFrame) -> dict[str, str]:
    """Guess PII columns from their names -> SDV sdtype (values get replaced by generated ones)."""
    found: dict[str, str] = {}
    for c in df.columns:
        for sdtype, pat in _PII_PATTERNS:
            if re.search(pat, str(c).lower()):
                found[str(c)] = sdtype
                break
    return found


# --------------------------------------------------------------------------- #
# Constraints, hyperparameters, conditional sampling
# --------------------------------------------------------------------------- #
def _coerce(raw: str, series: pd.Series):
    raw = raw.strip()
    if pd.api.types.is_bool_dtype(series):
        return raw.lower() in {"1", "true", "t", "yes", "y"}
    if pd.api.types.is_integer_dtype(series):
        return int(float(raw))
    if pd.api.types.is_numeric_dtype(series):
        return float(raw)
    return raw


def _ask_constraints(df: pd.DataFrame) -> list[dict]:
    cols = list(map(str, df.columns))
    num = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    out: list[dict] = []
    kinds = ["Inequality  (column A < column B, e.g. start < end)",
             "ScalarInequality  (column >= a value, e.g. age >= 18)",
             "FixedCombinations  (columns only appear in combinations seen in real data)",
             "Unique  (column values never repeat)",
             "Done"]
    while True:
        k = _menu("Add a constraint:", kinds)
        if k == 4:
            return out
        if k == 0:
            lo = _pick_column(cols, "LOW column", blank_ok=False)
            hi = _pick_column(cols, "HIGH column", blank_ok=False)
            strict = _ask_yes_no("Strict (low < high instead of low <= high)?", False)
            out.append({"type": "Inequality", "low": lo, "high": hi, "strict": strict})
        elif k == 1:
            col = _pick_column(cols, "Column", blank_ok=False)
            rel = [">", ">=", "<", "<="][_menu("Relation (column ? value):", [">", ">=", "<", "<="])]
            while True:
                try:
                    val = _coerce(input("Value: "), df[col])
                    break
                except ValueError:
                    print("[dappy] That value doesn't fit the column type.")
            out.append({"type": "ScalarInequality", "column": col, "relation": rel, "value": val})
        else:
            picked = _pick_columns(cols, "Columns (comma-separated)")
            if picked:
                out.append({"type": "FixedCombinations" if k == 2 else "Unique", "columns": picked})
        print(f"[dappy] {len(out)} constraint(s) so far.")


def _ask_hparams(method: str) -> tuple[int, dict]:
    """Returns (epochs, hyperparameters). Defaults are SDV's own."""
    epochs, hp = 300, {}
    if method in ("gan", "tvae", "auto"):
        epochs = _ask_number("Epochs", 300, int, lo=1)
    if method in ("gan", "auto"):
        print("CTGAN settings (Enter keeps SDV's default):")
        hp["batch_size"] = _ask_number("  batch_size (multiple of 10)", 500, int, lo=10, multiple_of=10)
        hp["generator_lr"] = _ask_number("  generator learning rate", 2e-4, float, lo=1e-8)
        hp["discriminator_lr"] = _ask_number("  discriminator learning rate", 2e-4, float, lo=1e-8)
        hp["discriminator_steps"] = _ask_number("  discriminator_steps", 1, int, lo=1)
        hp["embedding_dim"] = _ask_number("  embedding_dim", 128, int, lo=2)
    if method in ("tvae", "auto"):
        print("TVAE settings:")
        hp.setdefault("batch_size", _ask_number("  batch_size", 500, int, lo=2))
        hp.setdefault("embedding_dim", _ask_number("  embedding_dim", 128, int, lo=2))
    if method in ("copula", "auto"):
        dists = ["beta", "norm", "truncnorm", "uniform", "gamma", "gaussian_kde"]
        hp["default_distribution"] = dists[_menu("GaussianCopula numerical distribution:", dists)]
    return epochs, hp


def _ask_conditions(df: pd.DataFrame) -> list[dict]:
    """Conditional sampling: 'give me N rows where col = value'."""
    cols = list(map(str, df.columns))
    out: list[dict] = []
    while True:
        cv: dict = {}
        while True:
            col = _pick_column(cols, "Column to condition on (Enter when done)")
            if not col:
                break
            s = df[col]
            if s.nunique() <= 15:
                print("  values:", ", ".join(map(str, s.dropna().unique()[:15])))
            while True:
                try:
                    cv[col] = _coerce(input(f"  {col} = "), s)
                    break
                except ValueError:
                    print("[dappy] That value doesn't fit the column type.")
        if not cv:
            return out
        n = _ask_number("How many rows with these values", 1000, int, lo=1)
        out.append({"column_values": cv, "num_rows": n})
        if not _ask_yes_no("Add another condition?", False):
            return out


LICENSES = [
    ("CC0 1.0 (public domain)", "CC0-1.0"),
    ("CC BY-SA 4.0", "CC-BY-SA-4.0"),
    ("ODbL 1.0", "ODbL-1.0"),
    ("Unknown / other", "unknown"),
]


def _slugify(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:50].strip("-")
    return slug if len(slug) >= 3 else "dappy-synthetic-data"


# Partial static list of common Kaggle tags. Kaggle only warns about (and drops) unknown
# tags, so this is a helpful nudge, not a hard gate.
KNOWN_TAGS = {
    "beginner", "intermediate", "advanced", "classification", "regression", "clustering",
    "data visualization", "exploratory data analysis", "tabular", "time series", "nlp",
    "computer vision", "deep learning", "machine learning", "neural networks", "pandas",
    "numpy", "python", "statistics", "mathematics", "artificial intelligence", "business",
    "finance", "banking", "insurance", "economics", "marketing", "retail", "e-commerce",
    "real estate", "education", "health", "healthcare", "medicine", "biology", "chemistry",
    "physics", "earth science", "climate", "environment", "energy", "agriculture", "food",
    "sports", "football", "basketball", "music", "movies and tv shows", "games", "video games",
    "social networks", "internet", "news", "politics", "government", "transportation",
    "travel", "psychology", "linguistics", "text data", "image data", "logistic regression",
    "random forest", "xgboost", "tabular data", "data cleaning", "feature engineering",
}


def _validate_tags(tags: list[str]) -> list[str]:
    out: list[str] = []
    known = sorted(KNOWN_TAGS)
    for t in tags:
        if t in KNOWN_TAGS:
            out.append(t)
            continue
        close = difflib.get_close_matches(t, known, n=1, cutoff=0.7)
        if close and _ask_yes_no(f"Tag '{t}' isn't a known Kaggle tag. Use '{close[0]}' instead?", True):
            out.append(close[0])
        elif not close and _ask_yes_no(f"Tag '{t}' isn't in dappy's list of Kaggle tags "
                                       "(Kaggle may ignore it). Keep it?", False):
            out.append(t)
    return list(dict.fromkeys(out))


def _ask_dataset_meta(default_title: str) -> dict:
    """Ask for the details of the Kaggle dataset that will be created."""
    print("\nKaggle dataset details (press Enter to accept a [default]):")
    title = _ask_text("Dataset title (6-50 chars)", default_title, 6, 50)
    subtitle = _ask_text("Subtitle, one line (20-80 chars, blank to skip)", min_len=20, max_len=80,
                         optional=True)
    description = _ask_text("Description (blank for an auto-generated one)")
    raw_tags = _ask_text("Tags, comma-separated (blank for none)")
    tags = _validate_tags(list(dict.fromkeys(t.strip().lower() for t in raw_tags.split(",") if t.strip())))
    lic = LICENSES[_menu("License:", [label for label, _ in LICENSES])][1]
    public = _menu("Visibility:", ["Private (only you)", "Public"]) == 1
    return {"title": title, "slug": _slugify(title), "subtitle": subtitle,
            "description": description, "tags": tags, "license": lic, "public": public}


def _auto_description(source: str, method_label: str, rows: int, cols: int, quality: dict | None) -> str:
    text = (f"Synthetic version of {source}, generated with dappy using SDV ({method_label}). "
            f"{rows:,} rows x {cols:,} columns. Every row is sampled from a model fitted on the "
            "original data.")
    if quality:
        text += f" Overall SDV quality score: {quality['overall_score']:.1%}."
    return text


def _publish_to_kaggle(files: list[Path], meta: dict, username: str, interactive: bool = True) -> str:
    """Create a Kaggle dataset (or a new version of an existing one). Returns the ref (owner/slug)."""
    stage = Path(tempfile.mkdtemp(prefix="dappy_pub_"))
    try:
        for f in files:
            shutil.copy(f, stage / f.name)
        ref = f"{username}/{meta['slug']}"
        version = False
        if _dataset_exists(ref):
            pick = 0 if not interactive else _menu(
                f"A dataset named {ref} already exists on your account.",
                ["Publish as a new version of it", "Publish under a new name instead", "Cancel upload"])
            if pick == 2:
                raise RuntimeError("Upload cancelled.")
            if pick == 1:
                ref = f"{username}/{meta['slug'][:42].strip('-')}-{uuid.uuid4().hex[:5]}"
            else:
                version = True
        md = {"title": meta["title"], "id": ref, "licenses": [{"name": meta["license"]}]}
        if meta.get("subtitle"):
            md["subtitle"] = meta["subtitle"]
        if meta.get("description"):
            md["description"] = meta["description"]
        if meta.get("tags"):
            md["keywords"] = meta["tags"]
        (stage / "dataset-metadata.json").write_text(json.dumps(md, indent=2))

        if version:
            print(f"[dappy] Uploading a new version of {ref} ...")
            out = _cli("datasets", "version", "-p", str(stage), "-m",
                       f"dappy synthetic data {datetime.now():%Y-%m-%d %H:%M}")
        else:
            args = ["datasets", "create", "-p", str(stage)]
            if meta.get("public"):
                args.append("--public")
            print(f"[dappy] Uploading to Kaggle as {ref} ({'public' if meta.get('public') else 'private'}) ...")
            out = _cli(*args)
        if out.stdout.strip():  # e.g. Kaggle warns here about tags it doesn't recognise
            print(out.stdout.strip())
        _wait_dataset_ready(ref)
        return ref
    finally:
        shutil.rmtree(stage, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Local validation of the synthetic data
# --------------------------------------------------------------------------- #
def shape_warnings(quality: dict, threshold: float = 0.7) -> list[str]:
    out = []
    for r in quality.get("column_shapes") or []:
        try:
            score = float(r.get("Score"))
        except (TypeError, ValueError):
            continue
        if score < threshold:
            where = f"{r['Table']}." if r.get("Table") else ""
            out.append(f"{where}{r.get('Column')}: column-shape score {score:.2f} < {threshold:.2f}")
    return out


def leakage_check(real: pd.DataFrame, fake: pd.DataFrame) -> dict:
    """How many synthetic rows are verbatim copies of a real row."""
    cols = [c for c in real.columns if c in fake.columns]
    if not cols or fake.empty:
        return {"copied_rows": 0, "share": 0.0, "checked": 0}
    hr = pd.util.hash_pandas_object(real[cols].astype(str), index=False)
    hf = pd.util.hash_pandas_object(fake[cols].astype(str), index=False)
    n = int(hf.isin(set(hr)).sum())
    return {"copied_rows": n, "share": n / len(fake), "checked": len(fake)}


def _encode_for_distance(real: pd.DataFrame, fake: pd.DataFrame, cols: list[str]):
    parts_r, parts_f = [], []
    for c in cols:
        if pd.api.types.is_numeric_dtype(real[c]) and not pd.api.types.is_bool_dtype(real[c]):
            r = pd.to_numeric(real[c], errors="coerce")
            f = pd.to_numeric(fake[c], errors="coerce")
            med, sd = r.median(), r.std()
            sd = sd if sd and not np.isnan(sd) else 1.0
            parts_r.append(((r.fillna(med) - r.mean()) / sd).to_numpy()[:, None])
            parts_f.append(((f.fillna(med) - r.mean()) / sd).to_numpy()[:, None])
        elif real[c].nunique() <= 30:
            cats = sorted(real[c].astype(str).unique())
            for cat in cats:
                parts_r.append((real[c].astype(str) == cat).to_numpy(dtype=float)[:, None])
                parts_f.append((fake[c].astype(str) == cat).to_numpy(dtype=float)[:, None])
    if not parts_r:
        return None, None
    return np.hstack(parts_r).astype(np.float32), np.hstack(parts_f).astype(np.float32)


def _min_dist(a: np.ndarray, b: np.ndarray, chunk: int = 256) -> np.ndarray:
    b2 = (b ** 2).sum(1)
    out = np.empty(len(a), dtype=np.float32)
    for i in range(0, len(a), chunk):
        x = a[i:i + chunk]
        d = (x ** 2).sum(1)[:, None] + b2[None, :] - 2 * x @ b.T
        out[i:i + chunk] = np.sqrt(np.maximum(d, 0).min(axis=1))
    return out


def dcr_check(real: pd.DataFrame, fake: pd.DataFrame, max_real: int = 5000, max_fake: int = 2000) -> dict | None:
    """Distance to closest record. Baseline = how close *unseen real* rows are to the reference rows."""
    cols = [c for c in real.columns if c in fake.columns]
    if not cols or len(real) < 20 or fake.empty:
        return None
    r = real[cols].sample(min(len(real), max_real), random_state=0).reset_index(drop=True)
    f = fake[cols].sample(min(len(fake), max_fake), random_state=0).reset_index(drop=True)
    R, F = _encode_for_distance(r, f, cols)
    if R is None:
        return None
    h = min(1000, len(R) // 4)
    ref, hold = R[:-h], R[-h:]
    d_fake, d_hold = _min_dist(F, ref), _min_dist(hold, ref)
    med_f, med_h = float(np.median(d_fake)), float(np.median(d_hold))
    res = {"median_synth": med_f, "median_baseline": med_h,
           "p5_synth": float(np.percentile(d_fake, 5)), "exact_share": float((d_fake < 1e-6).mean())}
    res["verdict"] = ("possible memorisation: synthetic rows sit closer to real rows than unseen real rows do"
                      if med_h > 0 and med_f < 0.5 * med_h else "OK: not closer to real rows than unseen real data")
    return res


def _ks_2samp(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    a, b = np.sort(a), np.sort(b)
    allv = np.concatenate([a, b])
    d = float(np.max(np.abs(np.searchsorted(a, allv, side="right") / len(a)
                            - np.searchsorted(b, allv, side="right") / len(b))))
    en = math.sqrt(len(a) * len(b) / (len(a) + len(b)))
    lam = (en + 0.12 + 0.11 / en) * d
    p = 2 * sum((-1) ** (j - 1) * math.exp(-2 * j * j * lam * lam) for j in range(1, 101))
    return d, float(min(max(p, 0.0), 1.0))


def stat_tests(real: pd.DataFrame, fake: pd.DataFrame, cap: int = 5000) -> pd.DataFrame:
    """KS test for numeric columns, chi-square (needs scipy) for categorical ones."""
    try:
        from scipy import stats
    except ImportError:
        stats = None
    rows = []
    for c in real.columns:
        if c not in fake.columns:
            continue
        if pd.api.types.is_numeric_dtype(real[c]) and not pd.api.types.is_bool_dtype(real[c]):
            a = pd.to_numeric(real[c], errors="coerce").dropna()
            b = pd.to_numeric(fake[c], errors="coerce").dropna()
            if len(a) < 5 or len(b) < 5:
                continue
            a = a.sample(min(len(a), cap), random_state=0).to_numpy(float)
            b = b.sample(min(len(b), cap), random_state=0).to_numpy(float)
            d, p = stats.ks_2samp(a, b)[:2] if stats else _ks_2samp(a, b)
            rows.append({"column": c, "test": "KS", "statistic": float(d), "p_value": float(p)})
        elif stats is not None:
            if real[c].nunique() > 0.5 * len(real):  # ID-like text: a chi-square is meaningless
                continue
            ra, fb = real[c].astype(str), fake[c].astype(str)
            top = ra.value_counts().head(30).index
            ra, fb = ra.where(ra.isin(top), "(other)"), fb.where(fb.isin(top), "(other)")
            tab = pd.concat([ra.value_counts(), fb.value_counts()], axis=1).fillna(0)
            if len(tab) < 2:
                continue
            chi2, p = stats.chi2_contingency(tab.to_numpy().T)[:2]
            rows.append({"column": c, "test": "chi-square", "statistic": float(chi2), "p_value": float(p)})
    df = pd.DataFrame(rows, columns=["column", "test", "statistic", "p_value"])
    if not df.empty:
        df["verdict"] = np.where(df["p_value"] < 0.01, "differs", "similar")
    return df


def analyze(res: dict, s: dict):
    """Print + store: threshold warnings, KS / chi-square, leakage, DCR."""
    thr = float(s.get("min_shape", 0.7))
    warns = shape_warnings(res["quality"], thr)
    res["warnings"] = warns
    if warns:
        print(f"\n[dappy] WARNING - {len(warns)} column(s) below the {thr:.2f} column-shape threshold:")
        for w in warns[:15]:
            print(f"   - {w}")
    else:
        print(f"\n[dappy] All column-shape scores are >= {thr:.2f}.")
    if res["quality"].get("holdout"):
        h = res["quality"]["holdout"]
        print(f"[dappy] Train/test validation: quality vs held-out 20% = {h['vs_test']:.2%}, "
              f"vs the 80% it trained on = {h['vs_train']:.2%}.")

    res["checks"] = {}
    for name, fake in res["fake"].items():
        real = res["prepared"][name]
        chk: dict = {"leakage": leakage_check(real, fake), "dcr": dcr_check(real, fake)}
        tests = stat_tests(real, fake)
        chk["tests"] = tests
        res["checks"][name] = chk
        tag = f" [{name}]" if len(res["fake"]) > 1 else ""
        lk = chk["leakage"]
        print(f"\n[dappy] Leakage check{tag}: {lk['copied_rows']:,} of {lk['checked']:,} synthetic rows "
              f"({lk['share']:.2%}) are verbatim copies of real rows.")
        if chk["dcr"]:
            d = chk["dcr"]
            print(f"[dappy] Distance to closest record{tag}: synthetic median {d['median_synth']:.3f} vs "
                  f"unseen-real baseline {d['median_baseline']:.3f}, exact matches {d['exact_share']:.2%} "
                  f"-> {d['verdict']}")
        if not tests.empty:
            show_table(tests.round(4), f"Statistical parity{tag} (p < 0.01 = distributions differ)", index=False)


# --------------------------------------------------------------------------- #
# Report / exports / bundle
# --------------------------------------------------------------------------- #
_FIG_SINK = None  # callable(fig, name) -> when set, charts go here instead of a window


@contextlib.contextmanager
def _sink(fn):
    global _FIG_SINK
    prev, _FIG_SINK = _FIG_SINK, fn
    try:
        yield
    finally:
        _FIG_SINK = prev


def _collect_charts(real: pd.DataFrame, fake: pd.DataFrame, quality: dict | None) -> list[tuple[str, bytes]]:
    """Render the comparison charts to PNG bytes (empty list if matplotlib/seaborn are missing)."""
    if _plotting() is None:
        return []
    charts: list[tuple[str, bytes]] = []

    def grab(fig, name):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
        charts.append((name, buf.getvalue()))

    with _sink(grab):
        _plot_numeric_compare(real, fake)
        _plot_categorical_compare(real, fake)
        _plot_corr_compare(real, fake)
        if quality:
            _plot_quality(quality)
    return charts


def _df_html(df: pd.DataFrame) -> str:
    return df.to_html(index=False, border=0, na_rep="-", float_format=lambda x: f"{x:,.3f}", classes="tbl")


def write_report(res: dict, path: Path, s: dict, charts: list[tuple[str, bytes]]):
    q, meta = res["quality"], res.get("run_meta") or {}
    esc = _html.escape
    parts = [f"<h1>{APP_TITLE}</h1><p class='sub'>{esc(APP_TAGLINE)}</p>",
             f"<p class='meta'>Generated {datetime.now():%Y-%m-%d %H:%M} | method: "
             f"<b>{esc(str(meta.get('method', s.get('method'))))}</b> | SDV {esc(str(meta.get('sdv_version') or '?'))}"
             f" | target: {esc(str(s.get('target') or 'none (unsupervised)'))}</p>",
             f"<div class='score'>{q['overall_score']:.1%}<span>overall SDV quality score</span></div>"]
    if q.get("holdout"):
        h = q["holdout"]
        parts.append(f"<p>Train/test validation: vs held-out 20% <b>{h['vs_test']:.1%}</b>, "
                     f"vs training 80% <b>{h['vs_train']:.1%}</b>.</p>")
    if meta.get("auto_scores"):
        sc = {k: v for k, v in meta["auto_scores"].items() if v is not None}
        parts.append("<p>Auto-select scores: " + ", ".join(f"{k} {v:.1%}" for k, v in sc.items()) + "</p>")
    if res.get("warnings"):
        parts.append("<h2>Warnings</h2><ul class='warn'>" + "".join(f"<li>{esc(w)}</li>" for w in res["warnings"]) + "</ul>")
    parts.append("<h2>Quality by property</h2>" + _df_html(pd.DataFrame(q["properties"])))
    if q.get("column_shapes"):
        parts.append("<h2>Column shapes</h2>" + _df_html(pd.DataFrame(q["column_shapes"])))
    for name, chk in res.get("checks", {}).items():
        parts.append(f"<h2>Privacy and parity - {esc(name)}</h2>")
        lk = chk["leakage"]
        parts.append(f"<p>Verbatim copies of real rows: <b>{lk['copied_rows']:,}</b> of {lk['checked']:,} "
                     f"({lk['share']:.2%}).</p>")
        if chk["dcr"]:
            d = chk["dcr"]
            parts.append(f"<p>Distance to closest record: synthetic median <b>{d['median_synth']:.3f}</b> vs "
                         f"unseen-real baseline {d['median_baseline']:.3f}; exact matches {d['exact_share']:.2%}. "
                         f"{esc(d['verdict'])}.</p>")
        if not chk["tests"].empty:
            parts.append(_df_html(chk["tests"]))
    if res.get("diagnostic"):
        parts.append("<h2>SDV diagnostic</h2>" + _df_html(pd.DataFrame(res["diagnostic"]["properties"])))
    for name, fake in res["fake"].items():
        parts.append(f"<h2>Synthetic preview - {esc(name)}</h2>" + _df_html(fake.head(10)))
    if charts:
        parts.append("<h2>Charts</h2>")
        for title, png in charts:
            parts.append(f"<h3>{esc(title)}</h3><img alt='{esc(title)}' "
                         f"src='data:image/png;base64,{base64.b64encode(png).decode()}'>")
    css = ("body{font-family:system-ui,sans-serif;max-width:1000px;margin:2rem auto;padding:0 1rem;color:#222}"
           "h1{margin-bottom:0}.sub{color:#7a3fa0;margin-top:.2rem;font-weight:600}.meta{color:#666}"
           ".score{font-size:3rem;font-weight:700;color:#2a7}.score span{display:block;font-size:.9rem;color:#666;font-weight:400}"
           ".tbl{border-collapse:collapse;font-size:.85rem;margin:.5rem 0;display:block;overflow-x:auto}"
           ".tbl th,.tbl td{padding:.3rem .7rem;text-align:right;border-bottom:1px solid #eee}"
           ".tbl th{background:#f5f0fa}img{max-width:100%}.warn{color:#b3261e}")
    path.write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>{APP_TITLE} report</title>"
                    f"<style>{css}</style></head><body>{''.join(parts)}</body></html>", encoding="utf-8")


def build_bundle(res: dict, dest: Path, s: dict) -> list[Path]:
    """Write everything for delivery into `dest`: results, extra formats, report, charts, zip."""
    dest.mkdir(parents=True, exist_ok=True)
    for f in res["stage"].iterdir():
        if f.is_file() and f.suffix != ".log":  # skip the raw kernel log
            shutil.copy2(f, dest / f.name)
    multi = len(res["fake"]) > 1
    fmts = set(s.get("formats") or ["csv"])
    try:
        if "parquet" in fmts:
            for name, df in res["fake"].items():
                df.to_parquet(dest / (f"synthetic_{name}.parquet" if multi else "synthetic.parquet"), index=False)
        if "xlsx" in fmts:
            with pd.ExcelWriter(dest / "synthetic.xlsx") as w:
                for name, df in res["fake"].items():
                    df.to_excel(w, sheet_name=name[:31], index=False)
    except ImportError as e:
        print(f"[dappy] Export skipped, missing package: {e}  (pip install pyarrow openpyxl)")
    except ValueError as e:
        print(f"[dappy] Export skipped: {e}")

    charts: list[tuple[str, bytes]] = []
    real_c, fake_c = res.get("compare", (None, None))
    if s.get("report", True) or "zip" in fmts or s.get("zip"):
        if real_c is not None:
            charts = _collect_charts(real_c, fake_c, res["quality"])
        res["charts"] = charts
    if charts:
        (dest / "charts").mkdir(exist_ok=True)
        for title, png in charts:
            (dest / "charts" / f"{title}.png").write_bytes(png)
    if s.get("report", True):
        write_report(res, dest / "dappy_report.html", s, charts)
    if s.get("zip"):
        zpath = dest / "dappy_bundle.zip"
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(dest.rglob("*")):
                if f.is_file() and f != zpath:
                    z.write(f, f.relative_to(dest))
    return sorted(p for p in dest.rglob("*") if p.is_file())


# --------------------------------------------------------------------------- #
# Overviews
# --------------------------------------------------------------------------- #
def _split_columns(df: pd.DataFrame):
    num = df.select_dtypes(include="number").columns.tolist()
    return num, [c for c in df.columns if c not in num]


def _names(cols) -> str:
    if not cols:
        return "    (none)"
    return textwrap.fill(", ".join(map(str, cols)), width=90,
                         initial_indent="    ", subsequent_indent="    ")


def show_overview(df: pd.DataFrame, label: str = ""):
    num, cat = _split_columns(df)
    if RICH:
        grid = Table.grid(padding=(0, 2))
        grid.add_column(style="bold cyan", no_wrap=True)
        grid.add_column(overflow="fold")
        grid.add_row("Rows", f"{len(df):,}")
        grid.add_row("Columns", f"{df.shape[1]:,}")
        grid.add_row(f"Numerical ({len(num)})", escape(", ".join(map(str, num))) or "(none)")
        grid.add_row(f"Categorical ({len(cat)})", escape(", ".join(map(str, cat))) or "(none)")
        console.print(Panel(grid, title=f"Data Overview{' - ' + escape(label) if label else ''}",
                            border_style="magenta", expand=False))
        return
    print("\n" + "=" * 64)
    print(f" DATA OVERVIEW {('- ' + label) if label else ''}")
    print("=" * 64)
    print(f" Rows:                  {len(df):,}")
    print(f" Columns:               {df.shape[1]:,}")
    print(f" Numerical features:    {len(num)}")
    print(_names(num))
    print(f" Categorical features:  {len(cat)}")
    print(_names(cat))
    print("=" * 64)


def _ov_preview(df):
    show_table(df.head(10), "Preview (first 10 rows)")


def _ov_types(df):
    show_table(pd.DataFrame({"dtype": df.dtypes.astype(str), "non_null": df.notna().sum(),
                             "unique": df.nunique()}), "Column types")


def _ov_missing(df):
    miss = df.isna().sum()
    miss = miss[miss > 0]
    if miss.empty:
        print("No missing (NaN) values.")
    else:
        show_table(pd.DataFrame({"missing": miss, "pct": (miss / len(df) * 100).round(2)}),
                   "Missing values")
    obj = df.select_dtypes(exclude="number")
    placeholders = {c: int(obj[c].astype(str).str.strip().isin(["?", "", "NA", "N/A", "null", "None"]).sum())
                    for c in obj.columns}
    placeholders = {c: n for c, n in placeholders.items() if n}
    if placeholders:
        show_table(pd.Series(placeholders, name="count"),
                   "Placeholder-like values ('?', '', 'NA') in categorical columns")
    if not miss.empty and _ask_yes_no("\nView missing values as a chart?"):
        _plot_missing(df)


def _ov_numeric(df):
    num, _ = _split_columns(df)
    if num:
        show_table(df[num].describe().T.round(3), "Numerical summary")
        if _ask_yes_no("\nView the distributions as histograms?"):
            _plot_numeric_dist(df, num)
    else:
        print("No numerical columns.")


def _ov_categorical(df):
    _, cat = _split_columns(df)
    if cat:
        show_table(df[cat].describe().T, "Categorical summary")
        if _ask_yes_no("\nView the category counts as bar charts?"):
            _plot_categorical_counts(df, cat)
    else:
        print("No categorical columns.")


def _ov_dupes_memory(df):
    print(f"Duplicate rows: {int(df.duplicated().sum()):,}")
    print(f"Memory usage:   {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")


def _ov_corr(df):
    num, _ = _split_columns(df)
    if len(num) < 2:
        print("Need at least 2 numerical columns.")
        return
    corr = df[num].corr()
    pairs = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack().dropna()
    top = pairs.reindex(pairs.abs().sort_values(ascending=False).index).head(10)
    show_table(top.round(3).rename("correlation"), "Top numerical correlations")
    if _ask_yes_no("\nView the full correlation matrix visually (heatmap)?"):
        _plot_corr(df)


_OVERVIEWS = [
    ("Preview (first 10 rows)", _ov_preview),
    ("Column data types, non-null & unique counts", _ov_types),
    ("Missing values", _ov_missing),
    ("Numerical summary", _ov_numeric),
    ("Categorical summary", _ov_categorical),
    ("Duplicates & memory usage", _ov_dupes_memory),
    ("Correlations (numerical)", _ov_corr),
]


def _more_overview(df: pd.DataFrame):
    labels = [name for name, _ in _OVERVIEWS] + ["Back to main menu"]
    while True:
        pick = _menu("More Overview:", labels)
        if pick == len(_OVERVIEWS):  # last option = back
            return
        print()
        _OVERVIEWS[pick][1](df)



# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #
def compare(real: pd.DataFrame, fake: pd.DataFrame, quality: dict | None = None):
    print(f"\n=== Shapes ===\noriginal: {real.shape} | synthetic: {fake.shape}")

    if len(real) != len(fake):
        n = min(len(real), len(fake))
        if len(real) > n:
            real = real.sample(n, random_state=42).reset_index(drop=True)
            bigger = "original"
        else:
            fake = fake.sample(n, random_state=42).reset_index(drop=True)
            bigger = "synthetic"
        print(f"[dappy] Row counts differ: randomly downsampled the {bigger} dataset to {n:,} rows "
              "so both are compared at the same size.")

    if quality:
        print(f"\n=== SDV quality score: {quality['overall_score']:.2%} (computed on the full datasets) ===")
        show_table(pd.DataFrame(quality["properties"]), index=False)

    num = [c for c in real.select_dtypes("number").columns if c in fake.columns]
    if num:
        stats = pd.concat(
            {"orig": real[num].describe().T[["mean", "std", "min", "max"]],
             "synth": fake[num].describe().T[["mean", "std", "min", "max"]]},
            axis=1,
        ).round(3)
        show_table(stats, "Numerical columns: original vs synthetic")

    cat = [c for c in real.columns if c not in real.select_dtypes("number").columns and c in fake.columns]
    for col in cat[:5]:
        both = pd.concat(
            [real[col].value_counts(normalize=True).rename("original"),
             fake[col].value_counts(normalize=True).rename("synthetic")], axis=1
        ).fillna(0).round(3).head(8)
        show_table(both, f"{col}: top categories (share of rows)")

    return real, fake, quality


# --------------------------------------------------------------------------- #
# Charts (matplotlib + seaborn, imported lazily so they stay optional)
# --------------------------------------------------------------------------- #
MAX_CHART_COLS = 12          # columns per chart grid; keeps figures readable
_MAX_PLOT_ROWS = 100_000     # histograms/KDEs are drawn from at most this many rows
_NON_GUI_BACKENDS = {"agg", "pdf", "ps", "svg", "cairo", "template"}
_COLORS = {"original": "#4c72b0", "synthetic": "#dd8452"}


def _plotting():
    """Return (plt, sns), or None (with a hint) if they aren't installed."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("[dappy] Charts need matplotlib and seaborn:  pip install matplotlib seaborn")
        return None
    sns.set_theme(style="whitegrid")
    return plt, sns



def _show_fig(plt, fig, name: str):
    """Show the figure; hand it to the active sink (report/UI); or save a PNG on a headless backend."""
    fig.tight_layout()
    if _FIG_SINK is not None:
        _FIG_SINK(fig, name)
    elif plt.get_backend().lower() in _NON_GUI_BACKENDS:
        out = Path.cwd() / "dappy_charts"
        out.mkdir(exist_ok=True)
        path = out / f"{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[dappy] No display available - chart saved to {path}")
    else:
        plt.show()
    plt.close(fig)


def _grid(plt, n: int, ncols: int = 3, cell: tuple[float, float] = (4.5, 3.2)):
    ncols = max(1, min(ncols, n))
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(cell[0] * ncols, cell[1] * nrows), squeeze=False)
    flat = axes.ravel()
    for ax in flat[n:]:
        ax.set_visible(False)
    return fig, flat[:n]


def _cap_cols(cols: list) -> list:
    if len(cols) > MAX_CHART_COLS:
        print(f"[dappy] Showing the first {MAX_CHART_COLS} of {len(cols)} columns.")
        return cols[:MAX_CHART_COLS]
    return cols


def _sample(s: pd.Series) -> pd.Series:
    s = s.dropna()
    return s.sample(_MAX_PLOT_ROWS, random_state=0) if len(s) > _MAX_PLOT_ROWS else s


def _short(label, n: int = 18) -> str:
    s = str(label)
    return s if len(s) <= n else s[: n - 1] + "…"


def _plot_corr(df: pd.DataFrame):
    lib = _plotting()
    if not lib:
        return
    plt, sns = lib
    num, _ = _split_columns(df)
    corr = df[num].corr()
    n = len(num)
    side = max(5.0, min(0.6 * n + 3, 16.0))
    fig, ax = plt.subplots(figsize=(side * 1.1, side))
    mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)  # lower triangle only
    sns.heatmap(corr, mask=mask, annot=n <= 15, fmt=".2f", annot_kws={"size": 8}, cmap="coolwarm",
                vmin=-1, vmax=1, center=0, square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Correlation heatmap (numerical columns)")
    _show_fig(plt, fig, "correlation_heatmap")


def _plot_missing(df: pd.DataFrame):
    lib = _plotting()
    if not lib:
        return
    plt, sns = lib
    pct = (df.isna().mean() * 100)
    pct = pct[pct > 0].sort_values(ascending=False)
    if pct.empty:
        print("No missing values to plot.")
        return
    fig, ax = plt.subplots(figsize=(8, max(3.0, 0.4 * len(pct) + 1.5)))
    sns.barplot(x=pct.values, y=[_short(c, 30) for c in pct.index], orient="h", color="#c44e52", ax=ax)
    ax.bar_label(ax.containers[0], fmt="%.1f%%", padding=3)
    ax.set_xlabel("% of rows missing")
    ax.set_xlim(0, min(100, pct.max() * 1.15 + 1))
    ax.set_title("Missing values by column")
    _show_fig(plt, fig, "missing_values")


def _plot_numeric_dist(df: pd.DataFrame, cols: list):
    lib = _plotting()
    if not lib:
        return
    plt, sns = lib
    cols = _cap_cols([c for c in cols if df[c].notna().any()])
    if not cols:
        print("No numerical data to plot.")
        return
    fig, axes = _grid(plt, len(cols))
    for ax, c in zip(axes, cols):
        s = _sample(df[c])
        sns.histplot(s, kde=s.nunique() > 1, color=_COLORS["original"], ax=ax)
        ax.set_title(_short(c, 30))
        ax.set_xlabel("")
    fig.suptitle("Numerical distributions", y=1.0)
    _show_fig(plt, fig, "numerical_distributions")


def _plot_categorical_counts(df: pd.DataFrame, cols: list, top: int = 8):
    lib = _plotting()
    if not lib:
        return
    plt, sns = lib
    cols = _cap_cols([c for c in cols if df[c].notna().any()])
    if not cols:
        print("No categorical data to plot.")
        return
    fig, axes = _grid(plt, len(cols))
    for ax, c in zip(axes, cols):
        vc = df[c].value_counts().head(top)
        sns.barplot(x=vc.values, y=[_short(i) for i in vc.index], orient="h",
                    color=_COLORS["original"], ax=ax)
        n_unique = df[c].nunique()
        ax.set_title(f"{_short(c, 24)} (top {len(vc)} of {n_unique})")
        ax.set_xlabel("rows")
    fig.suptitle("Categorical value counts", y=1.0)
    _show_fig(plt, fig, "categorical_counts")


def _plot_numeric_compare(real: pd.DataFrame, fake: pd.DataFrame):
    lib = _plotting()
    if not lib:
        return
    plt, sns = lib
    cols = [c for c in real.select_dtypes("number").columns if c in fake.columns
            and (real[c].notna().any() or fake[c].notna().any())]
    cols = _cap_cols(cols)
    if not cols:
        print("No shared numerical columns to plot.")
        return
    fig, axes = _grid(plt, len(cols))
    for i, (ax, c) in enumerate(zip(axes, cols)):
        a, b = _sample(real[c]), _sample(fake[c])
        long = pd.DataFrame({c: pd.concat([a, b], ignore_index=True),
                             "source": ["original"] * len(a) + ["synthetic"] * len(b)})
        discrete = pd.api.types.is_integer_dtype(real[c]) and real[c].nunique() <= 20
        sns.histplot(data=long, x=c, hue="source", stat="density", common_norm=False, common_bins=True,
                     element="step", discrete=discrete, palette=_COLORS, ax=ax)
        ax.set_title(_short(c, 30))
        ax.set_xlabel("")
        if i > 0 and ax.get_legend() is not None:  # one legend is enough
            ax.get_legend().remove()
    fig.suptitle("Numerical distributions: original vs synthetic", y=1.0)
    _show_fig(plt, fig, "compare_numerical")


def _plot_categorical_compare(real: pd.DataFrame, fake: pd.DataFrame, top: int = 8):
    lib = _plotting()
    if not lib:
        return
    plt, sns = lib
    num = set(real.select_dtypes("number").columns)
    cols = _cap_cols([c for c in real.columns if c not in num and c in fake.columns])
    if not cols:
        print("No shared categorical columns to plot.")
        return
    fig, axes = _grid(plt, len(cols))
    for i, (ax, c) in enumerate(zip(axes, cols)):
        both = pd.concat([real[c].value_counts(normalize=True).rename("original"),
                          fake[c].value_counts(normalize=True).rename("synthetic")], axis=1).fillna(0)
        both = both.loc[both.max(axis=1).sort_values(ascending=False).index].head(top)
        both.index = [_short(x) for x in both.index]
        both.index.name = "category"
        long = both.reset_index().melt(id_vars="category", var_name="source", value_name="share")
        sns.barplot(data=long, x="category", y="share", hue="source", palette=_COLORS, ax=ax)
        ax.set_title(_short(c, 30))
        ax.set_xlabel("")
        ax.set_ylabel("share of rows")
        ax.tick_params(axis="x", rotation=45)
        for lbl in ax.get_xticklabels():
            lbl.set_horizontalalignment("right")
        if i > 0 and ax.get_legend() is not None:
            ax.get_legend().remove()
    fig.suptitle("Category shares: original vs synthetic", y=1.0)
    _show_fig(plt, fig, "compare_categorical")


def _plot_corr_compare(real: pd.DataFrame, fake: pd.DataFrame):
    lib = _plotting()
    if not lib:
        return
    plt, sns = lib
    num = [c for c in real.select_dtypes("number").columns if c in fake.columns]
    if len(num) < 2:
        print("Need at least 2 shared numerical columns.")
        return
    cr, cf = real[num].corr(), fake[num].corr()
    diff = (cr - cf).abs()
    n = len(num)
    cell = max(4.0, min(0.55 * n + 2.5, 8.0))
    fig, axes = plt.subplots(1, 3, figsize=(cell * 3, cell * 0.95))
    annot = n <= 10
    common = dict(square=True, linewidths=0.5, annot=annot, fmt=".2f", annot_kws={"size": 7},
                  cbar_kws={"shrink": 0.7})
    sns.heatmap(cr, cmap="coolwarm", vmin=-1, vmax=1, center=0, ax=axes[0], **common)
    sns.heatmap(cf, cmap="coolwarm", vmin=-1, vmax=1, center=0, ax=axes[1], **common)
    sns.heatmap(diff, cmap="Reds", vmin=0, vmax=1, ax=axes[2], **common)
    axes[0].set_title("Original")
    axes[1].set_title("Synthetic")
    axes[2].set_title("|Original - Synthetic|")
    _show_fig(plt, fig, "compare_correlation")


def _plot_quality(quality: dict):
    lib = _plotting()
    if not lib:
        return
    plt, sns = lib
    props = pd.DataFrame(quality.get("properties", []))
    shapes = pd.DataFrame(quality.get("column_shapes", []))
    panels = []
    if {"Property", "Score"} <= set(props.columns):
        props["Score"] = pd.to_numeric(props["Score"], errors="coerce")
        panels.append(("SDV quality by property", props["Property"].astype(str), props["Score"]))
    if {"Column", "Score"} <= set(shapes.columns):
        shapes["Score"] = pd.to_numeric(shapes["Score"], errors="coerce")
        shapes = shapes.dropna(subset=["Score"]).sort_values("Score").head(MAX_CHART_COLS * 2)
        panels.append(("Column shape score (lowest first)", shapes["Column"].astype(str), shapes["Score"]))
    panels = [(t, l, s) for t, l, s in panels if s.notna().any()]
    if not panels:
        print("No quality scores available to plot.")
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(6.5 * len(panels), 4), squeeze=False)
    for ax, (title, labels, scores) in zip(axes[0], panels):
        sns.barplot(x=scores.values, y=[_short(x, 30) for x in labels], orient="h",
                    color=_COLORS["synthetic"], ax=ax)
        ax.bar_label(ax.containers[0], fmt="%.2f", padding=3)
        ax.set_xlim(0, 1.1)
        ax.set_xlabel("score (1 = best)")
        ax.set_title(title)
    fig.suptitle(f"Overall SDV quality score: {quality['overall_score']:.1%}", y=1.0)
    _show_fig(plt, fig, "quality_scores")


def _compare_charts(real: pd.DataFrame, fake: pd.DataFrame, quality: dict | None):
    if not _ask_yes_no("\nView the original vs synthetic comparison as charts?"):
        return
    charts = [
        ("Numerical distributions (overlaid)", lambda: _plot_numeric_compare(real, fake)),
        ("Categorical shares (side by side)", lambda: _plot_categorical_compare(real, fake)),
        ("Correlation heatmaps (original / synthetic / difference)", lambda: _plot_corr_compare(real, fake)),
    ]
    if quality:
        charts.append(("Quality scores", lambda: _plot_quality(quality)))
    labels = [name for name, _ in charts] + ["Done"]
    while True:
        pick = _menu("Compare visually:", labels)
        if pick == len(charts):
            return
        charts[pick][1]()



# --------------------------------------------------------------------------- #
# Settings + pipeline (shared by the interactive CLI, batch mode and the UI)
# --------------------------------------------------------------------------- #
def _default_settings() -> dict:
    return {
        "source": None, "files": [], "target": None, "exclude": [], "missing": "keep",
        "train_rows": None, "anonymize": False, "pii": {}, "constraints": [],
        "primary_keys": {}, "relationships": [],
        "method": "gan", "epochs": 300, "hp": {}, "rows": None, "scale": 1.0,
        "conditions": [], "holdout": False,
        "gpu": False, "accelerator": None, "timeout": 3600,
        "out_dir": None, "formats": ["csv"], "zip": False, "report": True,
        "deliver": "local", "kaggle_meta": {}, "min_shape": 0.7,
    }


def _method_label(method: str, multi: bool) -> str:
    if multi:
        return "HMASynthesizer"
    return dict(METHODS, auto="Auto-selected").get(method, method)


def run_pipeline(s: dict, tables: dict[str, pd.DataFrame], tmp: Path) -> dict:
    """Prepare data -> run the Kaggle kernel -> download -> validate locally. No prompts."""
    multi = len(tables) > 1
    prepared, notes = prepare_tables(s, tables)
    for n in notes:
        print(f"[dappy] {n}")
    if s.get("anonymize") and not multi and not s.get("pii"):
        s["pii"] = detect_pii(next(iter(prepared.values())))

    stage_in = tmp / "input"
    stage_in.mkdir(exist_ok=True)
    files = []
    for name, df in prepared.items():
        p = stage_in / f"{name}.csv"
        df.to_csv(p, index=False)
        files.append(p)

    n_main = len(next(iter(prepared.values())))
    conditions = [] if multi else (s.get("conditions") or [])
    rows = sum(int(c["num_rows"]) for c in conditions) if conditions else int(s.get("rows") or n_main)
    method = "hma" if multi else s.get("method", "gan")
    extra = {
        "hp": s.get("hp") or {}, "scale": float(s.get("scale") or 1.0),
        "tables": {name: f"{name}.csv" for name in prepared},
        "relationships": s.get("relationships") or [], "primary_keys": s.get("primary_keys") or {},
        "constraints": [] if multi else (s.get("constraints") or []),
        "pii": {} if multi else (s.get("pii") or {}), "conditions": conditions,
        "holdout": bool(s.get("holdout")) and not multi,
    }
    if s.get("holdout") and multi:
        print("[dappy] Train/test validation is single-table only; skipping it.")

    with KaggleKernel(local_files=files, gpu=bool(s.get("gpu")), accelerator=s.get("accelerator"),
                      timeout=int(s.get("timeout") or 3600)) as k:
        k.run_job(method=method, rows=rows, epochs=int(s.get("epochs") or 300), **extra)
        stage = k.fetch(tmp / "results")
        username = k.username

    def _json(name):
        p = stage / name
        return json.loads(p.read_text()) if p.exists() else None

    fake = {}
    for name in prepared:
        f = stage / ("synthetic.csv" if not multi else f"synthetic_{name}.csv")
        fake[name] = pd.read_csv(f)
    res = {"stage": stage, "prepared": prepared, "fake": fake, "quality": _json("quality.json"),
           "diagnostic": _json("diagnostic.json"), "run_meta": _json("run_meta.json") or {},
           "username": username, "rows": rows, "settings": s,
           "method_label": _method_label(method, multi)}
    meta = res["run_meta"]
    if meta.get("auto_scores"):
        sc = ", ".join(f"{k}: {v:.1%}" for k, v in meta["auto_scores"].items() if v is not None)
        print(f"[dappy] Auto-select scores -> {sc}; winner: {meta.get('method')}")
        res["method_label"] = f"Auto ({meta.get('method')})"
    if (stage / "synthesizer.pkl").exists():
        print("[dappy] The fitted synthesizer was saved (synthesizer.pkl) - re-sample later without refitting.")

    first = next(iter(prepared))
    res["compare"] = compare(prepared[first], fake[first], res["quality"])[:2]
    for name in list(prepared)[1:]:
        print(f"\n[{name}] original {prepared[name].shape} | synthetic {fake[name].shape}")
    analyze(res, s)
    return res


# --------------------------------------------------------------------------- #
# Last run / config files
# --------------------------------------------------------------------------- #
def _save_last_run(s: dict):
    try:
        HOME.mkdir(parents=True, exist_ok=True)
        (HOME / "last_run.json").write_text(json.dumps(
            {"saved": datetime.now().isoformat(timespec="seconds"), "settings": s}, indent=2, default=str))
    except OSError:
        pass


def _load_last_run() -> dict | None:
    try:
        return json.loads((HOME / "last_run.json").read_text())
    except (OSError, ValueError):
        return None


def _load_config(path: str) -> dict:
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"Config file not found: {p}")
    text, suffix = p.read_text(encoding="utf-8"), p.suffix.lower()
    if suffix == ".toml":
        try:
            import tomllib
        except ImportError:
            raise RuntimeError("TOML config needs Python 3.11+ (or use .yaml / .json).")
        return tomllib.loads(text)
    if suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError:
            raise RuntimeError("YAML config needs PyYAML:  pip install pyyaml")
        return yaml.safe_load(text) or {}
    return json.loads(text)


def _save_config(s: dict, path: str):
    p = Path(path).expanduser()
    data = {k: v for k, v in s.items() if v not in (None, [], {}, "")}
    if p.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError:
            raise RuntimeError("Writing YAML needs PyYAML:  pip install pyyaml (or use a .json path)")
        p.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    elif p.suffix.lower() == ".json":
        p.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    else:
        raise RuntimeError("Use a .yaml or .json path to save the config.")
    print(f"[dappy] Saved config to {p}")


# --------------------------------------------------------------------------- #
# Delivery
# --------------------------------------------------------------------------- #
def _out_dir(s: dict) -> Path:
    return Path(s["out_dir"]).expanduser() if s.get("out_dir") else Path.cwd() / "dappy_output"


def _deliver_local(res: dict, s: dict) -> Path:
    out = _out_dir(s)
    files = build_bundle(res, out, s)
    print(f"[dappy] Saved {len(files)} file(s) to {out}")
    return out


def _kaggle_files(res: dict, tmp: Path, s: dict) -> list[Path]:
    pub = tmp / "publish"
    build_bundle(res, pub, {**s, "formats": ["csv"], "zip": False})
    keep = sorted(pub.glob("synthetic*.csv")) + sorted(pub.glob("dappy_report.html"))
    return keep


def _deliver_interactive(res: dict, s: dict, tmp: Path):
    pick = _menu("Synthetic data is ready. What would you like to do with it?",
                 ["Upload it to my Kaggle account", "Download it to my local directory"])
    if pick == 1:
        return _deliver_local(res, s)

    files = _kaggle_files(res, tmp, s)
    first = next(iter(res["fake"]))
    n_cols = res["fake"][first].shape[1]
    src = s["source"]["ref"] if isinstance(s.get("source"), dict) else str(s.get("source"))
    auto_desc = _auto_description(src, res["method_label"], res["rows"], n_cols, res["quality"])
    default_title = f"Synthetic {Path(src).stem}"[:50]

    while True:
        meta = _ask_dataset_meta(default_title)
        description = meta["description"] or auto_desc
        summary = {
            "Title": meta["title"],
            "URL": f"kaggle.com/datasets/{res['username']}/{meta['slug']}",
            "Files": ", ".join(f.name for f in files),
            "Subtitle": meta["subtitle"] or "-",
            "Description": description if len(description) <= 120 else description[:117] + "...",
            "Tags": ", ".join(meta["tags"]) or "-",
            "License": meta["license"],
            "Visibility": "Public" if meta["public"] else "Private",
        }
        show_table(pd.Series(summary, name="value").to_frame(), "Kaggle dataset")
        act = _menu("Ready to upload?", ["Upload now", "Edit the details", "Download to my local directory instead"])
        if act == 1:
            continue
        if act == 2:
            break
        meta["description"] = description
        try:
            ref = _publish_to_kaggle(files, meta, res["username"], interactive=True)
        except (RuntimeError, TimeoutError) as e:
            print(f"[dappy] Upload failed: {e}")
            if _menu("What now?", ["Try again with different details",
                                    "Download to my local directory instead"]) == 0:
                continue
            break
        print(f"[dappy] Uploaded: https://www.kaggle.com/datasets/{ref}")
        return None
    return _deliver_local(res, s)


def _deliver_batch(res: dict, s: dict, tmp: Path):
    if s.get("deliver") == "kaggle":
        m = dict(s.get("kaggle_meta") or {})
        src = s["source"]["ref"] if isinstance(s.get("source"), dict) else str(s.get("source"))
        title = (m.get("title") or f"Synthetic {Path(src).stem}")[:50].ljust(6, "_")
        first = next(iter(res["fake"]))
        meta = {"title": title, "slug": _slugify(title), "subtitle": m.get("subtitle", ""),
                "description": m.get("description") or _auto_description(
                    src, res["method_label"], res["rows"], res["fake"][first].shape[1], res["quality"]),
                "tags": [t for t in (m.get("tags") or []) if t in KNOWN_TAGS],
                "license": m.get("license", "CC0-1.0"), "public": bool(m.get("public", False))}
        ref = _publish_to_kaggle(_kaggle_files(res, tmp, s), meta, res["username"], interactive=False)
        print(f"[dappy] Uploaded: https://www.kaggle.com/datasets/{ref}")
        return None
    return _deliver_local(res, s)


def _ask_output_options(s: dict):
    raw = _ask_text("Extra outputs, comma-separated: parquet, xlsx, zip (Enter to skip)").lower()
    picks = {t.strip() for t in raw.split(",") if t.strip()}
    s["formats"] = ["csv"] + [f for f in ("parquet", "xlsx") if f in picks]
    s["zip"] = "zip" in picks
    s["report"] = _ask_yes_no("Create a self-contained HTML report (tables + charts + scores)?", True)


def _execute(s: dict, tables: dict, tmp: Path, interactive: bool) -> dict:
    res = run_pipeline(s, tables, tmp)
    _save_last_run(s)
    if interactive:
        real_c, fake_c = res["compare"]
        _compare_charts(real_c, fake_c, res["quality"])
        _ask_output_options(s)
        _deliver_interactive(res, s, tmp)
        if _ask_yes_no("\nSave these choices to a config file for reproducible runs?", False):
            path = _ask_text("Path (.yaml or .json)", "dappy.yaml")
            try:
                _save_config({k: v for k, v in s.items() if k != "kaggle_meta"}, path)
            except RuntimeError as e:
                print(f"[dappy] {e}")
    else:
        _deliver_batch(res, s, tmp)
    return res


# --------------------------------------------------------------------------- #
# Re-sample from a saved synthesizer (no refit, no Kaggle)
# --------------------------------------------------------------------------- #
def resample_from_model(pkl: str, rows: int, out_dir: Path | None = None) -> Path:
    try:
        from sdv.utils import load_synthesizer
    except ImportError:
        raise RuntimeError("Re-sampling locally needs SDV:  pip install sdv")
    synth = load_synthesizer(str(Path(pkl).expanduser()))
    out = (out_dir or Path.cwd() / "dappy_output")
    out.mkdir(parents=True, exist_ok=True)
    data = synth.sample(num_rows=int(rows))
    path = out / f"resampled_{datetime.now():%Y%m%d_%H%M%S}.csv"
    data.to_csv(path, index=False)
    print(f"[dappy] Wrote {len(data):,} rows to {path}")
    return path


def _resample_flow():
    pkl = _ask_text("Path to synthesizer.pkl", min_len=1).strip("\"'")
    rows = _ask_number("Rows to generate", 1000, int, lo=1)
    return resample_from_model(pkl, rows)


# --------------------------------------------------------------------------- #
# Streamlit UI
# --------------------------------------------------------------------------- #
def _launch_ui(preset: dict) -> bool:
    try:
        import streamlit  # noqa: F401
    except ImportError:
        print("[dappy] The UI needs Streamlit:  pip install streamlit")
        return False
    env = dict(os.environ, DAPPY_UI_PRESET=json.dumps(preset))
    cmd = [sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve()),
           "--browser.gatherUsageStats", "false"]
    print(f"\n[dappy] Starting the {APP_TITLE} UI - it opens in your browser. Press Ctrl-C here to stop it.")
    try:
        subprocess.run(cmd, env=env)
    except KeyboardInterrupt:
        pass
    return True


def _in_streamlit() -> bool:
    try:
        from streamlit.runtime import exists
        return bool(exists())
    except Exception:
        return False


def ui_main():
    import streamlit as st

    global _NONINTERACTIVE
    _NONINTERACTIVE = True
    st.set_page_config(page_title=APP_TITLE, page_icon="🧬", layout="wide")
    st.title(APP_TITLE)
    st.markdown(f"##### {APP_TAGLINE}")
    preset = json.loads(os.environ.get("DAPPY_UI_PRESET", "{}") or "{}")
    ss = st.session_state
    if "tmp" not in ss:
        ss.tmp = tempfile.mkdtemp(prefix="dappy_ui_")
    tmp = Path(ss.tmp)

    # 1 - data
    st.header("1. Data")
    c1, c2 = st.columns(2)
    text = c1.text_input("Kaggle link (dataset / competition / notebook) or a file / folder path",
                         value=preset.get("source", ""))
    up = c2.file_uploader("...or upload a file", type=[e.lstrip(".") for e in _EXTS])
    if st.button("Load data", type="primary"):
        try:
            with st.spinner("Loading ..."):
                if up is not None:
                    dest = tmp / "upload"
                    dest.mkdir(exist_ok=True)
                    (dest / up.name).write_bytes(up.getvalue())
                    src = {"kind": "local", "ref": str(dest / up.name)}
                else:
                    src = _detect_source(text)
                ss.src, ss.paths = src, [str(p) for p in _fetch_source(src, tmp)]
                ss.pop("res", None)
        except Exception as e:
            st.error(str(e))
    if not ss.get("paths"):
        st.info("Load a dataset to begin.")
        return
    names = [Path(p).name for p in ss.paths]
    pick = st.selectbox("File", names)
    if len(names) > 1:
        st.caption("Multi-table (HMASynthesizer) runs are available from the CLI / config file.")
    key = ("table", ss.src["ref"], pick)
    if ss.get("table_key") != key:
        ss.df = _read_table(Path(ss.paths[names.index(pick)]))
        ss.table_key = key
    df: pd.DataFrame = ss.df
    st.write(f"**{len(df):,} rows x {df.shape[1]} columns**")
    st.dataframe(df.head(20), use_container_width=True)

    # 2 - target
    st.header("2. Target")
    cols = list(map(str, df.columns))
    opts = ["(none - unsupervised)"] + cols
    idx = opts.index(preset["target"]) if preset.get("target") in cols else 0
    choice = st.selectbox("Target column", opts, index=idx)
    target = None if choice == opts[0] else choice
    st.info(_describe_task(df, target))

    # 3 - options
    st.header("3. Options")
    a, b, c = st.columns(3)
    exclude = a.multiselect("Exclude columns (IDs, timestamps, PII)", [x for x in cols if x != target])
    missing = b.radio("Missing values", list(MISSING_POLICIES), horizontal=True)
    train_rows = int(c.number_input("Fit on at most N rows (0 = all)", 0, value=0, step=1000))
    pii_guess = detect_pii(df)
    anonymize = a.checkbox(f"Anonymise PII columns ({', '.join(pii_guess) or 'none detected'})",
                           value=False, disabled=not pii_guess)
    method = b.selectbox("Synthesizer", [m for m, _ in METHODS] + ["auto"],
                         format_func=lambda m: dict(METHODS, auto="Auto (compare all three)")[m])
    rows = int(c.number_input("Synthetic rows", 1, value=len(df), step=max(1, len(df) // 10)))
    gpu = a.checkbox("Use a Kaggle GPU (GAN / TVAE)", value=False)
    holdout = b.checkbox("Train/test validation (fit on 80%, compare with held-out 20%)", value=False)
    epochs, hp = 300, {}
    with st.expander("Advanced"):
        if method in ("gan", "tvae", "auto"):
            epochs = int(st.number_input("Epochs", 1, value=300))
            hp["batch_size"] = int(st.number_input("batch_size", 10, value=500, step=10))
            hp["embedding_dim"] = int(st.number_input("embedding_dim", 2, value=128))
        if method in ("gan", "auto"):
            hp["discriminator_steps"] = int(st.number_input("discriminator_steps", 1, value=1))
            hp["generator_lr"] = float(st.number_input("generator_lr", value=2e-4, format="%.6f"))
            hp["discriminator_lr"] = float(st.number_input("discriminator_lr", value=2e-4, format="%.6f"))
        if method in ("copula", "auto"):
            hp["default_distribution"] = st.selectbox(
                "Copula distribution", ["beta", "norm", "truncnorm", "uniform", "gamma", "gaussian_kde"])
        timeout = int(st.number_input("Kernel timeout (minutes)", 5, value=60)) * 60
        cons_txt = st.text_area("Constraints (JSON list)", placeholder=(
            '[{"type": "Inequality", "low": "start", "high": "end"}, '
            '{"type": "ScalarInequality", "column": "age", "relation": ">=", "value": 18}]'))
        cond_txt = st.text_area("Conditional sampling (JSON list)", placeholder=(
            '[{"column_values": {"churned": "Yes"}, "num_rows": 500}]'))

    if st.button("Generate synthetic data", type="primary"):
        try:
            constraints = json.loads(cons_txt) if cons_txt.strip() else []
            conditions = json.loads(cond_txt) if cond_txt.strip() else []
        except ValueError as e:
            st.error(f"Constraints / conditions must be valid JSON: {e}")
            return
        s = {**_default_settings(), "source": ss.src, "files": [pick], "target": target,
             "exclude": exclude, "missing": missing, "train_rows": train_rows or None,
             "anonymize": anonymize, "method": method, "epochs": epochs, "hp": hp, "rows": rows,
             "holdout": holdout, "gpu": gpu, "timeout": timeout, "constraints": constraints,
             "conditions": conditions, "formats": ["csv", "parquet", "xlsx"], "zip": True, "report": True}
        name = re.sub(r"\W+", "_", Path(pick).stem).strip("_") or "table"
        box = st.empty()

        class _Live(io.StringIO):
            def write(self, x):
                n = super().write(x)
                box.code("\n".join(self.getvalue().splitlines()[-25:]) or " ")
                return n

        with st.status("Running on a private Kaggle kernel - this can take a while ...", expanded=True) as status:
            try:
                with contextlib.redirect_stdout(_Live()):
                    res = run_pipeline(s, {name: df}, tmp)
                    res["files"] = build_bundle(res, tmp / "bundle", s)
                ss.res, ss.settings = res, s
                status.update(label="Done", state="complete")
            except Exception as e:
                status.update(label="Failed", state="error")
                st.error(str(e))
                return

    res = ss.get("res")
    if not res:
        return
    st.header("4. Results")
    q = res["quality"]
    st.metric("Overall SDV quality score", f"{q['overall_score']:.1%}")
    if q.get("holdout"):
        h = q["holdout"]
        st.write(f"Train/test validation: vs held-out {h['vs_test']:.1%} | vs training {h['vs_train']:.1%}")
    for w in res.get("warnings", []):
        st.warning(w)
    st.dataframe(pd.DataFrame(q["properties"]), use_container_width=True)
    name = next(iter(res["fake"]))
    chk = res["checks"][name]
    lk = chk["leakage"]
    st.write(f"**Leakage:** {lk['copied_rows']:,} of {lk['checked']:,} synthetic rows ({lk['share']:.2%}) "
             "are verbatim copies of real rows.")
    if chk["dcr"]:
        d = chk["dcr"]
        st.write(f"**Distance to closest record:** synthetic median {d['median_synth']:.3f} vs baseline "
                 f"{d['median_baseline']:.3f} - {d['verdict']}")
    if not chk["tests"].empty:
        st.dataframe(chk["tests"], use_container_width=True)
    st.subheader("Synthetic preview")
    st.dataframe(res["fake"][name].head(50), use_container_width=True)
    for title, png in res.get("charts", []):
        st.image(png, caption=title)
    bundle = tmp / "bundle"
    d1, d2, d3 = st.columns(3)
    d1.download_button("Download CSV", (bundle / "synthetic.csv").read_bytes(), "synthetic.csv", "text/csv")
    if (bundle / "dappy_report.html").exists():
        d2.download_button("Download HTML report", (bundle / "dappy_report.html").read_bytes(),
                           "dappy_report.html", "text/html")
    if (bundle / "dappy_bundle.zip").exists():
        d3.download_button("Download zip bundle", (bundle / "dappy_bundle.zip").read_bytes(),
                           "dappy_bundle.zip", "application/zip")


# --------------------------------------------------------------------------- #
# Interactive entry point
# --------------------------------------------------------------------------- #
def _ask_settings(df: pd.DataFrame, s: dict, multi: bool):
    """Interactive questions that turn a loaded table into run settings."""
    cols = list(map(str, df.columns))
    if not multi:
        s["exclude"] = _pick_columns([c for c in cols if c != s.get("target")],
                                     "Columns to drop before fitting - IDs, timestamps, PII "
                                     "(comma-separated, Tab completes, Enter = none)")
        left = df.drop(columns=s["exclude"])
    else:
        left = df
    if int(left.isna().sum().sum()) > 0:
        n = int(left.isna().sum().sum())
        s["missing"] = MISSING_POLICIES[_menu(
            f"{n:,} missing values found. Policy:",
            ["Keep NaNs (SDV models them)", "Impute (median / most frequent)", "Drop rows with NaNs"])]
    if not multi:
        pii = detect_pii(left)
        if pii:
            show_table(pd.Series(pii, name="detected as").to_frame(), "Possible PII columns")
            s["anonymize"] = _ask_yes_no("Replace these with synthetic values (anonymisation preset)?", True)
            if s["anonymize"]:
                s["pii"] = pii
        if len(left) > 50_000:
            if _ask_yes_no(f"{len(left):,} rows is large. Fit on a random subsample to speed things up?", True):
                s["train_rows"] = _ask_number("Rows to fit on", 50_000, int, lo=100)

    if multi:
        s["method"] = "hma"
        s["scale"] = _ask_number("Scale factor (1.0 = same size as the originals)", 1.0, float, lo=0.01)
    else:
        idx = _menu("Synthesizer type:", [label for _, label in METHODS] + ["Auto (try all three, keep the best)"])
        s["method"] = (METHODS[idx][0] if idx < len(METHODS) else "auto")
        s["rows"] = _ask_rows(len(left))
    if s["method"] in ("gan", "tvae", "auto"):
        s["gpu"] = _ask_yes_no("Use a Kaggle GPU for this run?", False)

    if _ask_yes_no("\nAdvanced options (constraints, hyperparameters, conditional sampling, validation)?", False):
        if not multi:
            if _ask_yes_no("Add constraints (Inequality, ScalarInequality, FixedCombinations, Unique)?", False):
                s["constraints"] = _ask_constraints(left)
            if _ask_yes_no("Generate only rows matching conditions (e.g. more churned customers)?", False):
                s["conditions"] = _ask_conditions(left)
            s["holdout"] = _ask_yes_no("Validate on a held-out 20% (fit on 80%)?", False)
        if s["method"] != "hma" and _ask_yes_no("Tune hyperparameters?", False):
            s["epochs"], s["hp"] = _ask_hparams(s["method"])
        if s["gpu"]:
            acc = _menu("Kaggle accelerator:", ["Default GPU", "NVIDIA T4", "NVIDIA P100"])
            s["accelerator"] = [None, "NvidiaTeslaT4", "NvidiaTeslaP100"][acc]
        s["timeout"] = _ask_number("Kernel timeout in minutes", 60, int, lo=5) * 60
        s["min_shape"] = _ask_number("Warn when a column-shape score is below", 0.7, float, lo=0.0, hi=1.0)


def _repeat_last(last: dict, tmp: Path):
    s = {**_default_settings(), **last["settings"]}
    src = s["source"]
    print(f"[dappy] Repeating the run from {last.get('saved')} on {src['ref']}")
    paths = _fetch_source(src, tmp)
    chosen = _choose_files(paths, interactive=True, wanted=s.get("files") or None)
    tables = _load_tables(chosen)
    return _execute(s, tables, tmp, interactive=True)


def run(source: str | None = None):
    """Interactive session. Returns (real, synthetic, quality) for single-table runs."""
    _banner()
    _start_session_log()
    _kaggle_bin()  # fail fast: every path needs the CLI
    tmp = Path(tempfile.mkdtemp(prefix="dappy_data_"))
    try:
        last = _load_last_run()
        if last and not source:
            what = last["settings"].get("source") or {}
            pick = _menu("Start:", ["New run",
                                    f"Repeat last run ({what.get('ref', '?')}, {last['settings'].get('method')})",
                                    "Re-sample from a saved synthesizer (.pkl), no refit"])
            if pick == 1:
                res = _repeat_last(last, tmp)
                return res["compare"][0], res["compare"][1], res["quality"]
            if pick == 2:
                return _resample_flow()

        first = source
        while True:
            src, chosen, tables = _open_source(tmp, first)
            first = None
            multi = len(tables) > 1
            s = {**_default_settings(), "source": src, "files": [p.name for p in chosen]}
            if multi:
                for n, t in tables.items():
                    show_overview(t, n)
                s["primary_keys"], s["relationships"] = _ask_relationships(tables)
                df = next(iter(tables.values()))
            else:
                df = next(iter(tables.values()))
                s["target"] = _ask_target(df)
                if _ask_yes_no("Open the Streamlit UI instead of the terminal?", False):
                    if _launch_ui({"source": src["ref"], "target": s["target"]}):
                        return None
                show_overview(df, chosen[0].name)

            back = False
            while True:
                pick = _menu("What next?", ["More Overview", "Synthesize", "Change data source"])
                if pick == 0:
                    if multi:
                        names = list(tables)
                        _more_overview(tables[names[_menu("Which table?", names)]])
                    else:
                        _more_overview(df)
                elif pick == 2:
                    back = True
                    break
                else:
                    break
            if not back:
                break

        _ask_settings(df, s, multi)
        res = _execute(s, tables, tmp, interactive=True)
        if not multi:
            return res["compare"][0], res["compare"][1], res["quality"]
        return res
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Batch (non-interactive) mode + CLI
# --------------------------------------------------------------------------- #
def run_batch(s: dict):
    """No prompts: everything comes from flags / config. Delivers locally (or to Kaggle if configured)."""
    global _NONINTERACTIVE
    _NONINTERACTIVE = True
    _start_session_log()
    _kaggle_bin()
    s = {**_default_settings(), **s}
    if not s.get("source"):
        raise ValueError("Non-interactive mode needs a source: --dataset / --csv / --source (or `source:` in the config).")
    src = s["source"] if isinstance(s["source"], dict) else _detect_source(str(s["source"]))
    tmp = Path(tempfile.mkdtemp(prefix="dappy_data_"))
    try:
        chosen = _choose_files(_fetch_source(src, tmp), interactive=False, wanted=s.get("files") or None)
        tables = _load_tables(chosen)
        s.update(source=src, files=[p.name for p in chosen])
        if len(tables) == 1:
            df = next(iter(tables.values()))
            if s.get("target") and s["target"] not in df.columns:
                raise ValueError(f"Target column '{s['target']}' not found. Columns: {', '.join(map(str, df.columns))}")
            print(f"[dappy] {_describe_task(df, s.get('target'))}")
        return _execute(s, tables, tmp, interactive=False)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="dappy", description=f"{APP_TITLE} - {APP_TAGLINE}")
    p.add_argument("--source", "--dataset", "--csv", dest="source",
                   help="Kaggle link / owner/slug / file or folder path")
    p.add_argument("--file", action="append", dest="files",
                   help="table file to use from the source (repeat for multi-table)")
    p.add_argument("--target", help="target column (for the task-type report)")
    p.add_argument("--exclude", help="comma-separated columns to drop before fitting")
    p.add_argument("--missing", choices=MISSING_POLICIES)
    p.add_argument("--train-rows", type=int, help="fit on at most N random rows")
    p.add_argument("--anonymize", action="store_true", default=None, help="replace detected PII columns")
    p.add_argument("--method", choices=["gan", "copula", "tvae", "auto"])
    p.add_argument("--epochs", type=int)
    p.add_argument("--rows", type=int)
    p.add_argument("--scale", type=float, help="multi-table scale factor")
    p.add_argument("--holdout", action="store_true", default=None, help="fit on 80%%, validate on 20%%")
    p.add_argument("--gpu", action="store_true", default=None)
    p.add_argument("--accelerator", help="e.g. NvidiaTeslaT4, NvidiaTeslaP100")
    p.add_argument("--timeout", type=int, help="kernel timeout in seconds")
    p.add_argument("--out", dest="out_dir", help="output directory")
    p.add_argument("--formats", help="comma-separated extras: parquet,xlsx")
    p.add_argument("--zip", action="store_true", default=None)
    p.add_argument("--no-report", action="store_true", help="skip the HTML report")
    p.add_argument("--min-shape", type=float, dest="min_shape")
    p.add_argument("--config", help="dappy.yaml / .toml / .json with the same keys as the flags")
    p.add_argument("--save-config", help="write the effective settings to this .yaml / .json path")
    p.add_argument("--yes", "-y", action="store_true", help="non-interactive: never prompt")
    p.add_argument("--repeat-last", action="store_true", help="re-run the last saved settings")
    p.add_argument("--resample", metavar="MODEL.pkl", help="sample from a saved synthesizer (needs sdv locally)")
    p.add_argument("--ui", action="store_true", help="open the Streamlit UI")
    return p.parse_args(argv)


def _settings_from_args(a: argparse.Namespace) -> dict:
    s: dict = {}
    if a.repeat_last:
        last = _load_last_run()
        if not last:
            raise ValueError("No saved last run found.")
        s.update(last["settings"])
    if a.config:
        s.update(_load_config(a.config))
    for key in ("source", "files", "target", "missing", "train_rows", "anonymize", "method", "epochs",
                "rows", "scale", "holdout", "gpu", "accelerator", "timeout", "out_dir", "zip", "min_shape"):
        v = getattr(a, key)
        if v is not None:
            s[key] = v
    if a.exclude:
        s["exclude"] = [c.strip() for c in a.exclude.split(",") if c.strip()]
    if a.formats:
        s["formats"] = ["csv"] + [f.strip() for f in a.formats.split(",") if f.strip() in ("parquet", "xlsx")]
    if a.no_report:
        s["report"] = False
    if isinstance(s.get("exclude"), str):
        s["exclude"] = [c.strip() for c in s["exclude"].split(",") if c.strip()]
    return s


def main(argv=None):
    """Command-line entry point (the `dappy` command)."""
    try:
        a = _parse_args(argv)
        if a.ui:
            _launch_ui({"source": a.source or "", "target": a.target})
            return
        if a.resample:
            resample_from_model(a.resample, a.rows or 1000,
                                Path(a.out_dir) if a.out_dir else None)
            return
        s = _settings_from_args(a)
        if a.save_config:
            _save_config(s, a.save_config)
        if a.yes or a.repeat_last:
            run_batch(s)
        else:
            run(s.get("source") if isinstance(s.get("source"), str) else None)
    except KeyboardInterrupt:
        sys.exit("\n[dappy] Cancelled. Kaggle kernel/dataset cleaned up.")
    except (RuntimeError, ValueError, TimeoutError, FileNotFoundError) as e:
        sys.exit(f"[dappy] {e}")


if __name__ == "__main__":
    if _in_streamlit():
        ui_main()
    else:
        main()
