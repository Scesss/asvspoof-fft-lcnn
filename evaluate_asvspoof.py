import argparse
import os
from pathlib import Path

from src.metrics.calculate_eer import calculate_tDCF_EER


def main():
    parser = argparse.ArgumentParser(description="Calculate ASVspoof EER and t-DCF")
    parser.add_argument(
        "--cm-scores",
        type=Path,
        default=Path("data/saved/asvspoof/eval/cm_scores.txt"),
    )
    parser.add_argument("--asv-scores", type=Path, default=None)
    parser.add_argument(
        "--output", type=Path, default=Path("data/saved/asvspoof/metrics.txt")
    )
    args = parser.parse_args()

    dataset_root = Path(
        os.environ.get("ASVSPOOF_ROOT", "/kaggle/input/asvpoof-2019-dataset/LA")
    )
    asv_scores = args.asv_scores or (
        dataset_root
        / "ASVspoof2019_LA_asv_scores"
        / "ASVspoof2019.LA.asv.eval.gi.trl.scores.txt"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    eer, min_tdcf = calculate_tDCF_EER(
        cm_scores_file=args.cm_scores,
        asv_score_file=asv_scores,
        output_file=args.output,
        printout=True,
    )
    print(f"EER: {eer:.6f}%")
    print(f"min-tDCF: {min_tdcf:.6f}")


if __name__ == "__main__":
    main()
