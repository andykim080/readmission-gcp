"""Configure a project-local Jupyter kernel, including macOS OpenMP support."""
import json
import argparse
import sys
from pathlib import Path

if sys.version_info < (3, 11):
    raise SystemExit("Run with the project's .venv/bin/python (Python 3.12).")
import sklearn

root = Path(__file__).resolve().parents[1]
kernel_dir = root / ".venv/share/jupyter/kernels/readmission-gcp"
kernel_dir.mkdir(parents=True, exist_ok=True)
environment = {"MPLCONFIGDIR": str(root / ".matplotlib"), "IPYTHONDIR": str(root / ".ipython"), "PYTHONPATH": str(root)}
if sys.platform == "darwin":
    environment["DYLD_FALLBACK_LIBRARY_PATH"] = str(Path(sklearn.__file__).parent / ".dylibs")
for key in ["MPLCONFIGDIR", "IPYTHONDIR"]:
    Path(environment[key]).mkdir(parents=True, exist_ok=True)
spec = {
    "argv": [str(root / ".venv/bin/python"), "-Xfrozen_modules=off", "-m", "ipykernel_launcher", "-f", "{connection_file}"],
    "display_name": "Readmission GCP (Python 3.12)", "language": "python", "env": environment,
}
(kernel_dir / "kernel.json").write_text(json.dumps(spec, indent=2))
parser = argparse.ArgumentParser()
parser.add_argument("--user", action="store_true", help="Also register this kernel in the user's Jupyter kernel list.")
if parser.parse_args().user:
    from jupyter_client.kernelspec import KernelSpecManager
    installed = KernelSpecManager().install_kernel_spec(str(kernel_dir), kernel_name="readmission-gcp", user=True)
    print(f"Registered visible Jupyter kernel: {installed}")
print(f"Configured project kernel: {kernel_dir}")
print(f"Interpreter: {root / '.venv/bin/python'}")
