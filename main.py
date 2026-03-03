import subprocess
import sys
import os
import time

def run_step(script_name):
    print(f"\n{'='*60}")
    print(f"RUNNING: {script_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    # Use the same python executable that is running main.py
    result = subprocess.run([sys.executable, script_name], capture_output=False, text=True)
    
    if result.returncode == 0:
        elapsed = time.time() - start_time
        print(f"\nSUCCESS: {script_name} completed in {elapsed:.2f} seconds.")
        return True
    else:
        print(f"\nFAILED: {script_name} exited with code {result.returncode}")
        return False

def main():
    # Ensure we are in the application directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir)
    
    steps = [
        "Step1_feature_measuring.py",
        "Step2_w1_score.py",
        "Step3_component_selection.py",
        "Step4_market_health_index.py",
        "Step5_Predictive_modelling.py"
    ]
    
    print("Starting Market Health Index Pipeline...")
    overall_start = time.time()
    
    for step in steps:
        if not run_step(step):
            print("\nPipeline aborted due to error in step.")
            sys.exit(1)
            
    overall_elapsed = time.time() - overall_start
    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE: Total time {overall_elapsed:.2f} seconds.")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
