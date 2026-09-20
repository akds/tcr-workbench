"""Run with python -I after installing the wheel; no models or data downloads."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    import tcr_workbench

    package = Path(tcr_workbench.__file__).resolve().parent
    checkout = Path(__file__).resolve().parents[1]
    assert not package.is_relative_to(checkout), "Smoke test imported the source checkout"
    assert "site-packages" in package.parts, "Smoke test needs an installed wheel"
    workers = ("reconstruct_worker.py", "torch_sequence_worker.py", "torch_checkpoint_worker.py",
               "torch_cache.py", "mlx_worker.py", "precision_contract.py", "preparation_worker.py")
    for name in workers:
        assert (package / "backends" / name).is_file(), f"Missing packaged worker: {name}"
    assert (package / "model_preparation.py").is_file(), "Missing model preparation module"
    from tcr_workbench.species import biological_fingerprint

    reference = biological_fingerprint("mouse")
    assert Path(reference["mhc_reference_path"]).is_relative_to(package)
    command = [sys.executable, "-I", "-m", "tcr_workbench"]
    with tempfile.TemporaryDirectory(prefix="tcr-installed-") as directory:
        root = Path(directory)
        help_result = subprocess.run(command + ["--help"], cwd=root, check=True, timeout=30,
                                     capture_output=True, text=True)
        assert "prepare-model" in help_result.stdout
        print(help_result.stdout)
        subprocess.run(command + ["example", "--out", "example"], cwd=root, check=True, timeout=30)
        subprocess.run(command + ["screen", "--input", "example/receptors.csv",
            "--reference", "example/reference.csv", "--panel", "example/panel.csv",
            "--donors", "example/donors.csv", "--out", "screen"], cwd=root, check=True, timeout=30)
        output = root / "screen"
        assert (output / "report.html").is_file()
        assert (output / "evidence.parquet").is_file()
        manifest = json.loads((output / "manifest.json").read_text())
        assert manifest["outputs"], "Installed CLI produced no auditable outputs"
    print("Installed-wheel CLI and packaged workers passed without model inference.")


if __name__ == "__main__":
    main()
