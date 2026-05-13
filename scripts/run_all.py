from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

def run_script(script_name: str):
    print(f"\n>>> Running {script_name}")
    runpy.run_path(str(ROOT / "scripts" / script_name), run_name="__main__")

def main():
    run_script("run_synthetic_experiment.py")
    run_script("run_ablation.py")
    run_script("run_sensitivity.py")
    run_script("generate_all_figures.py")

    if (ROOT / "data/public/creditcard.csv").exists():
        run_script("run_creditcard_experiment.py")
    else:
        print("\nSkipping credit card experiment because data/public/creditcard.csv was not found.")

    print("\nAll available workflows completed successfully.")

if __name__ == "__main__":
    main()
