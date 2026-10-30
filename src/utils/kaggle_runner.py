"""
Kaggle Autonomous Execution & Telemetry Runner
Integrates Kaggle API with Antigravity to autonomously push, execute, monitor,
and retrieve notebooks and training artifacts on Kaggle (Dual Tesla T4 GPUs).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_USERNAME = "mohamedelbasyouni55"
DEFAULT_ACCELERATOR = "NvidiaTeslaT4"
DEFAULT_DATASET = "mohamedelbasyouni55/shoplifting-surveillance-dataset"
DEFAULT_KERNEL_SOURCE = "mohamedelbasyouni55/shoplifting-detection-pipeline"


def ensure_kaggle_env():
    """Ensure KAGGLE_API_TOKEN is available in environment."""
    if "KAGGLE_API_TOKEN" not in os.environ:
        token_path = Path.home() / ".kaggle" / "access_token"
        if token_path.exists():
            os.environ["KAGGLE_API_TOKEN"] = token_path.read_text(encoding="utf-8").strip()


def create_kernel_metadata(
    staging_dir: Path,
    notebook_name: str,
    kernel_slug: str,
    title: str,
    username: str = DEFAULT_USERNAME,
    enable_gpu: bool = True,
    accelerator: str = DEFAULT_ACCELERATOR,
    dataset_sources: list = None,
    kernel_sources: list = None,
):
    """Generate Kaggle kernel-metadata.json file."""
    metadata = {
        "id": f"{username}/{kernel_slug}",
        "title": title,
        "code_file": notebook_name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": enable_gpu,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": dataset_sources or [DEFAULT_DATASET],
        "kernel_sources": kernel_sources or [DEFAULT_KERNEL_SOURCE],
        "machine_shape": accelerator,
    }

    meta_file = staging_dir / "kernel-metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return meta_file


def push_kernel(
    notebook_path: str,
    kernel_slug: str = None,
    title: str = None,
    username: str = DEFAULT_USERNAME,
    accelerator: str = DEFAULT_ACCELERATOR,
    dataset_sources: list = None,
    kernel_sources: list = None,
) -> str:
    """Stage notebook and push to Kaggle to trigger remote cloud execution."""
    ensure_kaggle_env()
    nb_path = Path(notebook_path).resolve()
    if not nb_path.exists():
        raise FileNotFoundError(f"Notebook file not found: {nb_path}")

    if not kernel_slug:
        kernel_slug = nb_path.stem.lower().replace("_", "-")
    if not title:
        title = nb_path.stem

    staging_dir = Path("scratch") / f"staging_{kernel_slug}"
    staging_dir.mkdir(parents=True, exist_ok=True)

    dest_nb = staging_dir / nb_path.name
    shutil.copy2(nb_path, dest_nb)

    create_kernel_metadata(
        staging_dir=staging_dir,
        notebook_name=nb_path.name,
        kernel_slug=kernel_slug,
        title=title,
        username=username,
        accelerator=accelerator,
        dataset_sources=dataset_sources,
        kernel_sources=kernel_sources,
    )

    kernel_ref = f"{username}/{kernel_slug}"
    print(f"[KaggleRunner] Staging complete at {staging_dir}")
    print(f"[KaggleRunner] Pushing {kernel_ref} with accelerator {accelerator}...")

    cmd = [
        sys.executable,
        "-m",
        "kaggle",
        "kernels",
        "push",
        "-p",
        str(staging_dir),
        "--accelerator",
        accelerator,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, env=os.environ)
    if result.returncode != 0:
        print(f"[Error] Failed to push kernel:\n{result.stderr}")
        raise RuntimeError(f"Kaggle push failed: {result.stderr}")

    print(f"[KaggleRunner] Successfully pushed! Output: {result.stdout.strip()}")
    return kernel_ref


def get_kernel_status(kernel_ref: str) -> str:
    """Check live status of the latest kernel execution."""
    ensure_kaggle_env()
    cmd = [sys.executable, "-m", "kaggle", "kernels", "status", kernel_ref]
    result = subprocess.run(cmd, capture_output=True, text=True, env=os.environ)
    if result.returncode != 0:
        return f"UNKNOWN / ERROR: {result.stderr.strip()}"
    return result.stdout.strip()


def wait_for_completion(kernel_ref: str, poll_interval_sec: int = 30, timeout_min: int = 60) -> bool:
    """Poll Kaggle until execution finishes or times out."""
    ensure_kaggle_env()
    start_time = time.time()
    max_sec = timeout_min * 60

    print(f"[KaggleRunner] Monitoring execution for {kernel_ref} (Timeout: {timeout_min}m)...")
    while time.time() - start_time < max_sec:
        status_text = get_kernel_status(kernel_ref)
        elapsed_min = (time.time() - start_time) / 60
        print(f"[{elapsed_min:.1f}m] Status: {status_text}")

        if "COMPLETE" in status_text.upper():
            print(f"[KaggleRunner] Kernel execution SUCCEEDED in {elapsed_min:.1f} minutes!")
            return True
        elif "ERROR" in status_text.upper() or "FAILED" in status_text.upper():
            print(f"[KaggleRunner] Kernel execution FAILED: {status_text}")
            return False

        time.sleep(poll_interval_sec)

    print(f"[KaggleRunner] Polling timed out after {timeout_min} minutes.")
    return False


def download_artifacts(kernel_ref: str, output_dir: str = None, notebook_dir: str = None):
    """Download all output files and the executed notebook."""
    ensure_kaggle_env()
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        print(f"[KaggleRunner] Downloading output files to {out_path}...")
        cmd_out = [sys.executable, "-m", "kaggle", "kernels", "output", kernel_ref, "-p", str(out_path)]
        res_out = subprocess.run(cmd_out, capture_output=True, text=True, env=os.environ)
        print(res_out.stdout.strip() or res_out.stderr.strip())

    if notebook_dir:
        nb_path = Path(notebook_dir)
        nb_path.mkdir(parents=True, exist_ok=True)
        print(f"[KaggleRunner] Pulling executed notebook to {nb_path}...")
        cmd_pull = [sys.executable, "-m", "kaggle", "kernels", "pull", kernel_ref, "-p", str(nb_path)]
        res_pull = subprocess.run(cmd_pull, capture_output=True, text=True, env=os.environ)
        print(res_pull.stdout.strip() or res_pull.stderr.strip())

    print("[KaggleRunner] Download complete!")


def run_pipeline(notebook_path: str, output_dir: str = None, notebook_dir: str = "notebooks", poll_interval: int = 30):
    """Execute complete autonomous loop: push -> wait -> download."""
    ref = push_kernel(notebook_path)
    success = wait_for_completion(ref, poll_interval_sec=poll_interval)
    if success:
        download_artifacts(ref, output_dir=output_dir, notebook_dir=notebook_dir)
        print(f"[KaggleRunner] Pipeline for {ref} completed successfully!")
    else:
        print(f"[KaggleRunner] Pipeline failed or did not complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kaggle Autonomous Execution Runner")
    subparsers = parser.add_subparsers(dest="command")

    push_p = subparsers.add_parser("push", help="Push notebook to Kaggle")
    push_p.add_argument("notebook", help="Path to .ipynb file")
    push_p.add_argument("--slug", help="Kernel slug")
    push_p.add_argument("--title", help="Kernel title")
    push_p.add_argument("--accelerator", default=DEFAULT_ACCELERATOR, help="Hardware accelerator")

    status_p = subparsers.add_parser("status", help="Get kernel status")
    status_p.add_argument("ref", help="Kernel reference: username/slug")

    down_p = subparsers.add_parser("download", help="Download kernel artifacts")
    down_p.add_argument("ref", help="Kernel reference: username/slug")
    down_p.add_argument("--output-dir", "-o", help="Directory for output artifacts")
    down_p.add_argument("--notebook-dir", "-n", help="Directory for executed notebook")

    run_p = subparsers.add_parser("run", help="Push, wait, and download in one command")
    run_p.add_argument("notebook", help="Path to .ipynb file")
    run_p.add_argument("--output-dir", "-o", help="Directory for output artifacts")
    run_p.add_argument("--notebook-dir", "-n", default="notebooks", help="Directory for executed notebook")
    run_p.add_argument("--poll", type=int, default=30, help="Poll interval in seconds")

    args = parser.parse_args()
    if args.command == "push":
        push_kernel(args.notebook, kernel_slug=args.slug, title=args.title, accelerator=args.accelerator)
    elif args.command == "status":
        print(get_kernel_status(args.ref))
    elif args.command == "download":
        download_artifacts(args.ref, output_dir=args.output_dir, notebook_dir=args.notebook_dir)
    elif args.command == "run":
        run_pipeline(args.notebook, output_dir=args.output_dir, notebook_dir=args.notebook_dir, poll_interval=args.poll)
    else:
        parser.print_help()
