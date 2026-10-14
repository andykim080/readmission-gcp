"""Execute saved-run notebooks with the project environment and persist outputs."""
import os
import sys
import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--retrain", action="store_true", help="Exercise every training cell as well as the saved-result path.")
parser.add_argument("--notebook", action="append", choices=["01_model_experiments.ipynb", "02_gcp_workflow.ipynb", "03_model_improvements.ipynb"], help="Execute only the selected notebook(s).")
args = parser.parse_args()
os.chdir(root)
for variable, subdirectory in [
    ("MPLCONFIGDIR", ".matplotlib"), ("IPYTHONDIR", ".ipython"),
    ("JUPYTER_RUNTIME_DIR", ".jupyter/runtime"), ("JUPYTER_CONFIG_DIR", ".jupyter/config"),
]:
    directory = root / subdirectory
    directory.mkdir(parents=True, exist_ok=True)
    os.environ[variable] = str(directory)
# The registered kernel itself supplies OpenMP settings; verify that behavior.
os.environ.pop("DYLD_FALLBACK_LIBRARY_PATH", None)

for name in args.notebook or ["01_model_experiments.ipynb", "02_gcp_workflow.ipynb"]:
    path = root / name
    notebook = nbformat.read(path, as_version=4)
    if args.retrain and name == "01_model_experiments.ipynb":
        for cell in notebook.cells:
            if cell.cell_type == "code":
                cell.source = cell.source.replace("RUN_TRAINING = False", "RUN_TRAINING = True")
    nbformat.validate(notebook)
    client = NotebookClient(notebook, timeout=600, kernel_name="readmission-gcp", resources={"metadata": {"path": str(root)}},
                            on_cell_executed=lambda cell, cell_index, execute_reply: print(f"{name}: cell {cell_index + 1} completed", flush=True))
    client.execute()
    nbformat.write(notebook, path)
    errors = [output for cell in notebook.cells if cell.cell_type == "code" for output in cell.get("outputs", []) if output.output_type == "error"]
    if errors:
        raise RuntimeError(f"{name} has cell errors: {errors}")
    print(f"Executed {name}: {sum(cell.cell_type == 'code' for cell in notebook.cells)} code cells passed.", flush=True)
