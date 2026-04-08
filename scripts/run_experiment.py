from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOLUTION_PATH = ROOT / "scripts" / "solution.py"
RUN_LOG_PATH = ROOT / "run.log"
RESULTS_PATH = ROOT / "results.tsv"
PREDICTION_PATH = ROOT / "Predictions" / "prediction_irr_need.csv"
MODEL_PATH = ROOT / "artifacts" / "irrigation_need_catboost.cbm"
METADATA_PATH = ROOT / "artifacts" / "irrigation_need_metadata.json"
RESULTS_HEADER = [
    "timestamp",
    "run",
    "status",
    "val_balanced_accuracy_score",
    "best_iteration",
    "commit",
    "branch",
    "description",
    "model_config",
    "feature_config",
]
RUNNER_OWNED_ARGS = {
    "--help",
    "-h",
    "--skip-refit",
    "--submission-path",
    "--model-path",
    "--metadata-path",
}
SCORE_EPSILON = 1e-12


@dataclass(frozen=True)
class RunResult:
    score: float
    best_iteration: str
    weakest_class_recall: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run scripts/solution.py in champion-safe mode, record the result in "
            "results.tsv, and only refresh the canonical prediction/model artifacts "
            "when validation produces a strict improvement."
        )
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--description", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--feature-config", required=True)
    parser.add_argument(
        "--log-path",
        type=Path,
        default=RUN_LOG_PATH,
        help="Path for the latest run log. Defaults to run.log at repo root.",
    )
    parser.add_argument(
        "solution_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed to scripts/solution.py after `--`.",
    )
    args = parser.parse_args()
    args.solution_args = normalize_solution_args(args.solution_args)
    validate_solution_args(args.solution_args)
    return args


def normalize_solution_args(solution_args: list[str]) -> list[str]:
    if solution_args and solution_args[0] == "--":
        return solution_args[1:]
    return solution_args


def validate_solution_args(solution_args: list[str]) -> None:
    conflicts = [arg for arg in solution_args if arg in RUNNER_OWNED_ARGS]
    if conflicts:
        conflict_text = ", ".join(sorted(set(conflicts)))
        raise SystemExit(
            "run_experiment.py owns these solution.py arguments: "
            f"{conflict_text}. Remove them from the forwarded argument list."
        )


def ensure_results_file() -> None:
    if RESULTS_PATH.exists():
        return

    with RESULTS_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULTS_HEADER, delimiter="\t")
        writer.writeheader()


def read_results() -> list[dict[str, str]]:
    ensure_results_file()
    with RESULTS_PATH.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def append_result(row: dict[str, str]) -> None:
    rows = read_results()
    rows.append(row)
    with RESULTS_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULTS_HEADER, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def champion_score() -> float | None:
    scores: list[float] = []
    for row in read_results():
        if row.get("status") != "champion":
            continue
        try:
            scores.append(float(row["val_balanced_accuracy_score"]))
        except (TypeError, ValueError, KeyError):
            continue
    if not scores:
        return None
    return max(scores)


def current_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def git_stdout(args: list[str], fallback: str) -> str:
    try:
        completed = subprocess.run(
            args,
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return fallback
    value = completed.stdout.strip()
    return value or fallback


def current_branch() -> str:
    return git_stdout(["git", "branch", "--show-current"], "unknown")


def current_commit() -> str:
    commit = git_stdout(["git", "rev-parse", "--short", "HEAD"], "unknown")
    tracked_dirty = False
    for args in (
        ["git", "diff", "--quiet", "--ignore-submodules", "--"],
        ["git", "diff", "--cached", "--quiet", "--ignore-submodules", "--"],
    ):
        completed = subprocess.run(args, cwd=ROOT, check=False)
        if completed.returncode != 0:
            tracked_dirty = True
            break
    return f"{commit}-dirty" if tracked_dirty else commit


def extract_text(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text, flags=re.MULTILINE)
    if match is None:
        return None
    return match.group(1).strip()


def parse_run_log(log_path: Path) -> RunResult:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    score_text = extract_text(r"^val_balanced_accuracy_score:\s*([0-9.]+)$", text)
    if score_text is None:
        raise ValueError("Missing val_balanced_accuracy_score in run log.")

    best_iteration = extract_text(r"^best_iteration:\s*([0-9]+)$", text) or ""
    weakest_class_recall = (
        extract_text(r"^weakest_class_recall:\s*(.+)$", text) or ""
    )
    return RunResult(
        score=float(score_text),
        best_iteration=best_iteration,
        weakest_class_recall=weakest_class_recall,
    )


def crash_reason(log_path: Path, returncode: int) -> str:
    if not log_path.exists():
        return f"process exited with code {returncode}"

    text = log_path.read_text(encoding="utf-8", errors="replace")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return f"process exited with code {returncode}"
    return lines[-1]


def run_solution(args: list[str], log_path: Path) -> tuple[int, Path]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(SOLUTION_PATH), *args]
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            text=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    return completed.returncode, log_path


def replace_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(destination))


def format_row(
    *,
    run_name: str,
    status: str,
    score: str,
    best_iteration: str,
    description: str,
    model_config: str,
    feature_config: str,
) -> dict[str, str]:
    return {
        "timestamp": current_timestamp(),
        "run": run_name,
        "status": status,
        "val_balanced_accuracy_score": score,
        "best_iteration": best_iteration,
        "commit": current_commit(),
        "branch": current_branch(),
        "description": description,
        "model_config": model_config,
        "feature_config": feature_config,
    }


def full_run_description(
    *,
    base_description: str,
    champion_before: float | None,
    score: float,
    weakest_class_recall: str,
) -> str:
    weakest_text = ""
    if weakest_class_recall:
        weakest_text = f" Weakest recall: {weakest_class_recall}."

    if champion_before is None:
        return (
            f"{base_description} Initial champion at {score:.10f}."
            f"{weakest_text}"
        )
    return (
        f"{base_description} Improved from {champion_before:.10f} to {score:.10f}."
        f"{weakest_text}"
    )


def discard_description(
    *,
    base_description: str,
    champion_before: float,
    score: float,
    weakest_class_recall: str,
) -> str:
    weakest_text = ""
    if weakest_class_recall:
        weakest_text = f" Weakest recall: {weakest_class_recall}."
    return (
        f"{base_description} No lift versus champion {champion_before:.10f}; "
        f"finished at {score:.10f}.{weakest_text}"
    )


def crash_description(base_description: str, reason: str) -> str:
    return f"{base_description} Crash: {reason}"


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Expected artifact was not created: {path}")


def main() -> None:
    args = parse_args()
    ensure_results_file()
    champion_before = champion_score()

    with tempfile.TemporaryDirectory() as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        validation_metadata = temp_dir / "validation_metadata.json"
        validation_args = [
            *args.solution_args,
            "--skip-refit",
            "--metadata-path",
            str(validation_metadata),
        ]

        validation_returncode, validation_log_path = run_solution(
            validation_args,
            args.log_path,
        )
        if validation_returncode != 0:
            reason = crash_reason(validation_log_path, validation_returncode)
            append_result(
                format_row(
                    run_name=args.run_name,
                    status="crash",
                    score="",
                    best_iteration="",
                    description=crash_description(args.description, reason),
                    model_config=args.model_config,
                    feature_config=args.feature_config,
                )
            )
            print(f"status: crash")
            print(f"reason: {reason}")
            print(f"run_log_path: {validation_log_path}")
            return

        try:
            validation_result = parse_run_log(validation_log_path)
        except ValueError as exc:
            append_result(
                format_row(
                    run_name=args.run_name,
                    status="crash",
                    score="",
                    best_iteration="",
                    description=crash_description(args.description, str(exc)),
                    model_config=args.model_config,
                    feature_config=args.feature_config,
                )
            )
            print("status: crash")
            print(f"reason: {exc}")
            print(f"run_log_path: {validation_log_path}")
            return

        if (
            champion_before is not None
            and validation_result.score <= champion_before + SCORE_EPSILON
        ):
            append_result(
                format_row(
                    run_name=args.run_name,
                    status="discard",
                    score=f"{validation_result.score:.10f}",
                    best_iteration=validation_result.best_iteration,
                    description=discard_description(
                        base_description=args.description,
                        champion_before=champion_before,
                        score=validation_result.score,
                        weakest_class_recall=validation_result.weakest_class_recall,
                    ),
                    model_config=args.model_config,
                    feature_config=args.feature_config,
                )
            )
            print("status: discard")
            print(
                f"val_balanced_accuracy_score: {validation_result.score:.10f}"
            )
            print(f"best_iteration: {validation_result.best_iteration}")
            if validation_result.weakest_class_recall:
                print(
                    f"weakest_class_recall: {validation_result.weakest_class_recall}"
                )
            print(f"run_log_path: {validation_log_path}")
            return

        submission_temp = temp_dir / "prediction_irr_need.csv"
        model_temp = temp_dir / "irrigation_need_catboost.cbm"
        metadata_temp = temp_dir / "irrigation_need_metadata.json"
        full_run_args = [
            *args.solution_args,
            "--submission-path",
            str(submission_temp),
            "--model-path",
            str(model_temp),
            "--metadata-path",
            str(metadata_temp),
        ]
        full_returncode, full_log_path = run_solution(full_run_args, args.log_path)
        if full_returncode != 0:
            reason = crash_reason(full_log_path, full_returncode)
            append_result(
                format_row(
                    run_name=args.run_name,
                    status="crash",
                    score=f"{validation_result.score:.10f}",
                    best_iteration=validation_result.best_iteration,
                    description=crash_description(args.description, reason),
                    model_config=args.model_config,
                    feature_config=args.feature_config,
                )
            )
            print("status: crash")
            print(f"reason: {reason}")
            print(f"run_log_path: {full_log_path}")
            return

        try:
            full_result = parse_run_log(full_log_path)
        except ValueError as exc:
            append_result(
                format_row(
                    run_name=args.run_name,
                    status="crash",
                    score=f"{validation_result.score:.10f}",
                    best_iteration=validation_result.best_iteration,
                    description=crash_description(args.description, str(exc)),
                    model_config=args.model_config,
                    feature_config=args.feature_config,
                )
            )
            print("status: crash")
            print(f"reason: {exc}")
            print(f"run_log_path: {full_log_path}")
            return

        if champion_before is not None and full_result.score <= champion_before + SCORE_EPSILON:
            append_result(
                format_row(
                    run_name=args.run_name,
                    status="discard",
                    score=f"{full_result.score:.10f}",
                    best_iteration=full_result.best_iteration,
                    description=discard_description(
                        base_description=args.description,
                        champion_before=champion_before,
                        score=full_result.score,
                        weakest_class_recall=full_result.weakest_class_recall,
                    ),
                    model_config=args.model_config,
                    feature_config=args.feature_config,
                )
            )
            print("status: discard")
            print(f"val_balanced_accuracy_score: {full_result.score:.10f}")
            print(f"best_iteration: {full_result.best_iteration}")
            if full_result.weakest_class_recall:
                print(f"weakest_class_recall: {full_result.weakest_class_recall}")
            print(f"run_log_path: {full_log_path}")
            return

        try:
            require_file(submission_temp)
            require_file(model_temp)
            require_file(metadata_temp)
            replace_file(submission_temp, PREDICTION_PATH)
            replace_file(model_temp, MODEL_PATH)
            replace_file(metadata_temp, METADATA_PATH)
        except (FileNotFoundError, OSError) as exc:
            append_result(
                format_row(
                    run_name=args.run_name,
                    status="crash",
                    score=f"{full_result.score:.10f}",
                    best_iteration=full_result.best_iteration,
                    description=crash_description(args.description, str(exc)),
                    model_config=args.model_config,
                    feature_config=args.feature_config,
                )
            )
            print("status: crash")
            print(f"reason: {exc}")
            print(f"run_log_path: {full_log_path}")
            return

        append_result(
            format_row(
                run_name=args.run_name,
                status="champion",
                score=f"{full_result.score:.10f}",
                best_iteration=full_result.best_iteration,
                description=full_run_description(
                    base_description=args.description,
                    champion_before=champion_before,
                    score=full_result.score,
                    weakest_class_recall=full_result.weakest_class_recall,
                ),
                model_config=args.model_config,
                feature_config=args.feature_config,
            )
        )
        print("status: champion")
        print(f"val_balanced_accuracy_score: {full_result.score:.10f}")
        print(f"best_iteration: {full_result.best_iteration}")
        if full_result.weakest_class_recall:
            print(f"weakest_class_recall: {full_result.weakest_class_recall}")
        print(f"run_log_path: {full_log_path}")
        print(f"submission_path: {PREDICTION_PATH}")
        print(f"model_path: {MODEL_PATH}")
        print(f"metadata_path: {METADATA_PATH}")
        print(f"results_path: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
