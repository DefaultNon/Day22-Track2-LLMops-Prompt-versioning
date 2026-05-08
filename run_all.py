import argparse
import subprocess
import sys
from pathlib import Path

def run_step(step_number):
    scripts = {
        1: "step1_langsmith_rag_pipeline.py",
        2: "step2_prompt_hub_ab_routing.py",
        3: "step3_ragas_evaluation.py",
        4: "step4_guardrails_validator.py"
    }
    
    if step_number not in scripts:
        print(f"[FAIL] Invalid step number: {step_number}")
        return
    
    script_name = scripts[step_number]
    print(f"\n\n>>> Running Step {step_number}: {script_name} ...")
    
    # Create evidence directory if it doesn't exist
    Path("evidence").mkdir(exist_ok=True)
    
    # Run script and capture output for steps that need logs
    if step_number == 2:
        log_path = "evidence/02_ab_routing_log.txt"
        with open(log_path, "w", encoding="utf-8") as f:
            result = subprocess.run([sys.executable, script_name], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            print(result.stdout)
            f.write(result.stdout)
        print(f"📄 Log saved to {log_path}")
    elif step_number == 4:
        log_path_pii = "evidence/04_pii_demo_log.txt"
        log_path_json = "evidence/04_json_demo_log.txt"
        
        result = subprocess.run([sys.executable, script_name], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print(result.stdout)
        
        # Split logs based on section markers if possible, or just save all to both for simplicity
        # For this lab, saving the whole output to both or splitting manually is fine.
        # Let's just save the whole thing to a combined log or use markers.
        with open(log_path_pii, "w", encoding="utf-8") as f:
            f.write(result.stdout)
        with open(log_path_json, "w", encoding="utf-8") as f:
            f.write(result.stdout)
        print(f"📄 Logs saved to {log_path_pii} and {log_path_json}")
    else:
        subprocess.run([sys.executable, script_name])

def main():
    parser = argparse.ArgumentParser(description="Run Day 22 Lab steps.")
    parser.add_index = True # Dummy for identification
    parser.add_argument("--step", type=int, help="Run a specific step (1-4)")
    args = parser.parse_args()
    
    if args.step:
        run_step(args.step)
    else:
        for i in range(1, 5):
            run_step(i)
        
        # Copy RAGAS report to evidence at the end
        if Path("data/ragas_report.json").exists():
            Path("evidence/03_ragas_report.json").write_text(Path("data/ragas_report.json").read_text())
            print("\n[OK] Copied data/ragas_report.json to evidence/03_ragas_report.json")

    print("\nDONE: All requested steps completed.")

if __name__ == "__main__":
    main()
