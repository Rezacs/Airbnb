import ast
import re
from pathlib import Path
from typing import List, Set, Tuple, Dict

import nbformat as nbf

# --------- CONFIG ---------
INPUT = "Exploring+PreProcessing+VER2.ipynb"
OUTPUT = "Exploring+PreProcessing+VER2_with_explanations.ipynb"
MARKER = "🧭 Detailed Summary (auto-generated)"
MAX_INLINE_REPR = 160  # shorten long text/plain reprs
# If you want to execute the notebook first to refresh outputs, set EXECUTE = True (requires nbclient)
EXECUTE = False
# --------------------------

# ---- Optional execution (captures up-to-date outputs) ----
def maybe_execute(path: Path) -> None:
    if not EXECUTE:
        return
    try:
        from nbclient import NotebookClient
        nb = nbf.read(str(path), as_version=4)
        client = NotebookClient(nb, timeout=180, kernel_name=nb.metadata.get("kernelspec", {}).get("name", "python3"))
        client.execute()
        nbf.write(nb, str(path))
        print("Executed notebook to refresh outputs.")
    except Exception as e:
        print(f"[warn] Could not execute notebook: {e}. Proceeding with existing outputs.")

# ----------------- AST Utilities -----------------
class NameCollector(ast.NodeVisitor):
    def __init__(self):
        self.read: Set[str] = set()
        self.written: Set[str] = set()

    def visit_Name(self, node: ast.Name):
        if isinstance(node.ctx, ast.Load):
            self.read.add(node.id)
        elif isinstance(node.ctx, (ast.Store, ast.Del)):
            self.written.add(node.id)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        # ignore attr chains for read/written vars; base Name will be visited
        self.generic_visit(node)

def get_names_read_written(tree: ast.AST) -> Tuple[Set[str], Set[str]]:
    nc = NameCollector()
    nc.visit(tree)
    # Avoid counting python keywords/builtins accidentally
    blacklist = {"self", "cls", "__name__"}
    return {x for x in nc.read if x not in blacklist}, {x for x in nc.written if x not in blacklist}

def call_names(tree: ast.AST) -> List[str]:
    names = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Name):
                names.append(f.id)
            elif isinstance(f, ast.Attribute):
                # collect 'sns.scatterplot' -> 'sns.scatterplot'
                parts = []
                while isinstance(f, ast.Attribute):
                    parts.append(f.attr)
                    f = f.value
                if isinstance(f, ast.Name):
                    parts.append(f.id)
                names.append(".".join(reversed(parts)))
    return names

def class_inits(tree: ast.AST) -> List[str]:
    inits = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call):
            f = n.value.func
            name = None
            if isinstance(f, ast.Name):
                name = f.id
            elif isinstance(f, ast.Attribute):
                parts = []
                while isinstance(f, ast.Attribute):
                    parts.append(f.attr)
                    f = f.value
                if isinstance(f, ast.Name):
                    parts.append(f.id)
                name = ".".join(reversed(parts))
            if name:
                inits.append(name)
    return inits

def kwargs_of_first_init_for(prefixes: Tuple[str, ...], tree: ast.AST) -> Dict[str, str]:
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call):
            f = n.value.func
            qname = None
            if isinstance(f, ast.Name):
                qname = f.id
            elif isinstance(f, ast.Attribute):
                parts = []
                while isinstance(f, ast.Attribute):
                    parts.append(f.attr)
                    f = f.value
                if isinstance(f, ast.Name):
                    parts.append(f.id)
                qname = ".".join(reversed(parts))
            if qname and qname.startswith(prefixes):
                # collect simple kwarg reprs
                kv = []
                for kw in n.value.keywords:
                    try:
                        kv.append(f"{kw.arg}={ast.unparse(kw.value)}")
                    except Exception:
                        kv.append(f"{kw.arg}=<expr>")
                return {"estimator": qname, "params": ", ".join(kv)}
    return {}

# ----------------- Domain Heuristics -----------------
def summarize_io(code: str) -> List[str]:
    items = []
    # reads
    for pat, label in [
        (r"\b(pd\.)?read_csv\(([^)]*)\)", "reads CSV"),
        (r"\b(pd\.)?read_parquet\(([^)]*)\)", "reads Parquet"),
        (r"\b(pd\.)?read_excel\(([^)]*)\)", "reads Excel"),
        (r"\b(pd\.)?read_json\(([^)]*)\)", "reads JSON"),
        (r"\bjoblib\.load\(([^)]*)\)", "loads joblib artifact"),
        (r"\bpickle\.load\(", "loads pickle"),
    ]:
        for m in re.finditer(pat, code):
            arg = (m.group(2) if m.lastindex else "") or ""
            path = _extract_pathlike(arg)
            items.append(f"{label}{f' from {path}' if path else ''}")
    # writes
    for pat, label in [
        (r"\.to_csv\(([^)]*)\)", "writes CSV"),
        (r"\.to_parquet\(([^)]*)\)", "writes Parquet"),
        (r"\.to_excel\(([^)]*)\)", "writes Excel"),
        (r"\.to_json\(([^)]*)\)", "writes JSON"),
        (r"\bjoblib\.dump\(([^)]*)\)", "saves joblib artifact"),
        (r"\bpickle\.dump\(", "saves pickle"),
        (r"\bplt\.savefig\(([^)]*)\)", "saves figure"),
    ]:
        for m in re.finditer(pat, code):
            arg = m.group(1) or ""
            path = _extract_pathlike(arg)
            items.append(f"{label}{f' to {path}' if path else ''}")
    return items

def _extract_pathlike(arg: str) -> str:
    # crude heuristic to fish out first string literal that looks like a path/filename
    m = re.search(r"""['"]([^'"]+\.(csv|parquet|xlsx?|json|png|jpg|jpeg|svg|pkl|joblib))['"]""", arg, re.I)
    return m.group(1) if m else ""

def summarize_pandas(code: str, calls: List[str]) -> List[str]:
    hints = []
    patterns = [
        (r"\bmerge\(", "merges DataFrames"),
        (r"\bjoin\(", "joins DataFrames"),
        (r"\bconcat\(", "concatenates DataFrames"),
        (r"\bgroupby\(", "groups data (groupby)"),
        (r"\bpivot(_table)?\(", "pivots data"),
        (r"\bmelt\(", "melts/unpivots data"),
        (r"\bassign\(", "creates/overwrites columns (.assign)"),
        (r"\brename\(", "renames columns/index"),
        (r"\bdropna\(|\bfillna\(|\bisna\(", "handles missing values"),
        (r"\bastype\(", "casts dtypes"),
        (r"\bsort_values\(", "sorts rows"),
        (r"\bfilter\(|\bloc\[|\biloc\[", "filters rows/columns"),
        (r"\bdescribe\(\)", "describes summary stats"),
        (r"\bvalue_counts\(\)", "computes value counts"),
    ]
    for pat, msg in patterns:
        if re.search(pat, code):
            hints.append(msg)
    # minor: detect df shapes if printed
    if re.search(r"\.shape\b", code):
        hints.append("checks DataFrame shape")
    return hints

def summarize_plots(code: str, calls: List[str]) -> List[str]:
    tips = []
    plot_calls = [
        ("plt.scatter", "scatter plot"),
        ("plt.hist", "histogram"),
        ("plt.bar", "bar chart"),
        ("plt.plot", "line plot"),
        ("plt.boxplot", "box plot"),
        ("sns.scatterplot", "scatter plot (seaborn)"),
        ("sns.histplot", "histogram (seaborn)"),
        ("sns.barplot", "bar chart (seaborn)"),
        ("sns.lineplot", "line plot (seaborn)"),
        ("sns.boxplot", "box plot (seaborn)"),
        ("sns.heatmap", "heatmap"),
        ("px.", "interactive plotly figure"),
    ]
    for prefix, label in plot_calls:
        if any(c.startswith(prefix) for c in calls):
            tips.append(f"creates {label}")
    if "plt.savefig" in code:
        tips.append("saves figure to file")
    if re.search(r"\bplt\.show\(\)", code):
        tips.append("displays figure")
    return tips

def summarize_ml(tree: ast.AST, calls: List[str], code: str) -> List[str]:
    notes = []
    # estimator object + params
    est = kwargs_of_first_init_for((
        "sklearn.", "xgboost.", "lightgbm.", "catboost.", "XGB", "LGBM", "CatBoost"
    ), tree)
    if est:
        s = f"initializes estimator: `{est['estimator']}`"
        if est.get("params"):
            s += f" with params ({est['params']})"
        notes.append(s)

    if any(re.search(r"\bfit\(", line) for line in code.splitlines()):
        notes.append("fits/trains model on data")
    if any(re.search(r"\bpredict\(", line) for line in code.splitlines()):
        notes.append("generates predictions")
    if re.search(r"\btransform\(|\bfit_transform\(", code):
        notes.append("applies preprocessing transform")
    if re.search(r"\btrain_test_split\(", code):
        notes.append("splits data into train/test")
    if re.search(r"\bGridSearchCV\(|\bRandomizedSearchCV\(", code):
        notes.append("performs hyperparameter search")
    if re.search(r"\bPipeline\(", code):
        notes.append("builds an ML pipeline")
    if re.search(r"\b(accuracy_score|roc_auc_score|f1_score|precision_score|recall_score|confusion_matrix|classification_report|mean_squared_error|r2_score)\b", code):
        notes.append("computes model metrics")
    return notes

# ----------------- Output summarizer -----------------
def summarize_outputs(cell) -> str:
    outs = cell.get("outputs", [])
    if not outs:
        return "No stored output."

    pieces = []
    for o in outs:
        otype = o.get("output_type", "")
        if otype == "stream":
            text = o.get("text", "")
            lines = text.strip("\n").count("\n") + (1 if text.strip() else 0)
            pieces.append(f"prints text (~{lines} line{'s' if lines != 1 else ''})")
        elif otype in ("display_data", "execute_result"):
            data = o.get("data", {})
            mimes = set(data.keys())
            if "image/png" in mimes or "image/jpeg" in mimes:
                pieces.append("displays a figure/image")
            if "text/html" in mimes:
                pieces.append("renders HTML (e.g., table/pretty output)")
            if "application/vnd.plotly.v1+json" in mimes:
                pieces.append("shows interactive Plotly figure")
            if "text/plain" in mimes:
                txt = str(data.get("text/plain", "")).strip()
                if len(txt) > MAX_INLINE_REPR:
                    txt = txt[:MAX_INLINE_REPR] + "…"
                if txt:
                    pieces.append(f"text/plain repr: `{txt}`")
        elif otype == "error":
            ename = o.get("ename", "Error")
            evalue = o.get("evalue", "")
            pieces.append(f"raises **{ename}** – {evalue}")
    pieces = list(dict.fromkeys(pieces))
    return "; ".join(pieces) if pieces else "Produces output."

# ----------------- Master per-cell summarizer -----------------
def summarize_code_cell(source: str) -> List[str]:
    if not source.strip():
        return ["Empty cell."]
    try:
        tree = ast.parse(source)
    except Exception:
        # un-parseable: magics or non-Python content
        io = summarize_io(source)
        misc = []
        if re.search(r"^%", source, flags=re.M):  # IPython magics
            misc.append("uses notebook magic(s)")
        return io + misc + ["Executes code (could not fully parse)."]

    reads, writes = get_names_read_written(tree)
    calls = call_names(tree)

    bullets = []

    # I/O
    io = summarize_io(source)
    if io:
        bullets += io

    # pandas
    pd_ops = summarize_pandas(source, calls)
    if pd_ops:
        bullets += pd_ops

    # plots
    p = summarize_plots(source, calls)
    if p:
        bullets += p

    # ML
    ml = summarize_ml(tree, calls, source)
    if ml:
        bullets += ml

    # general signals
    if any(isinstance(n, (ast.For, ast.While)) for n in ast.walk(tree)):
        bullets.append("loops/iterates over data")
    if any(isinstance(n, ast.If) for n in ast.walk(tree)):
        bullets.append("applies conditional logic")
    if any(isinstance(n, ast.Try) for n in ast.walk(tree)):
        bullets.append("handles exceptions")

    # variables
    defined = sorted(writes)
    used = sorted(reads - writes)  # “inputs” to this cell
    if defined:
        bullets.append("defines variables: " + ", ".join(defined[:10]) + (" …" if len(defined) > 10 else ""))
    if used:
        bullets.append("uses prior variables: " + ", ".join(used[:10]) + (" …" if len(used) > 10 else ""))

    if not bullets:
        bullets.append("Executes general-purpose Python code.")
    return bullets

def build_markdown_desc(code_cell) -> str:
    src = code_cell.get("source", "")
    bullets = summarize_code_cell(src)
    out = summarize_outputs(code_cell)

    # Format as a compact, high-signal summary
    lines = [f"**{MARKER}**", "", "**What it does**"]
    lines += [f"- {b}" for b in bullets]
    lines += ["", f"**Output**", f"- {out}", "", "---"]
    return "\n".join(lines)

def already_has_summary(prev_cell) -> bool:
    return (
        prev_cell
        and prev_cell.cell_type == "markdown"
        and MARKER in (prev_cell.get("source") or "")
    )

def main():
    in_path = Path(INPUT)
    out_path = Path(OUTPUT)

    if not in_path.exists():
        raise FileNotFoundError(f"Could not find input notebook: {in_path}")

    # Optional: execute to refresh outputs before summarizing
    maybe_execute(in_path)

    nb = nbf.read(str(in_path), as_version=4)

    new_cells = []
    for cell in nb.cells:
        if cell.cell_type == "code":
            if not (new_cells and already_has_summary(new_cells[-1])):
                md_cell = nbf.v4.new_markdown_cell(build_markdown_desc(cell))
                new_cells.append(md_cell)
        new_cells.append(cell)

    nb.cells = new_cells
    nbf.write(nb, str(out_path))
    print(f"Done. Wrote: {out_path}")

if __name__ == "__main__":
    main()
