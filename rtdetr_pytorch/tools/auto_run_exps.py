"""
by HO TZU CHUN
"""

import subprocess
import sys
import os
import time
from datetime import datetime

def run_training(config_path, experiment_name):
    """
    run single training experiment
    
    Args:
        config_path (str): config file path
        experiment_name (str): experiment name
    """
    print(f"\n{'='*60}")
    print(f"start running experiment: {experiment_name}")
    print(f"config file: {config_path}")
    print(f"start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    # build training command
    train_script = "train.py"
    cmd = [
        sys.executable,  # use current python interpreter
        train_script,
        "-c", config_path
    ]
    
    try:
        # run training command
        print(f"execute command: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=False)
        subprocess.kill(result.pid)
        
        print(f"\n{'='*60}")
        print(f"experiment {experiment_name} completed!")
        print(f"completed time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        
    except subprocess.CalledProcessError as e:
        print(f"\n experiment {experiment_name} failed!")
        print(f"error code: {e.returncode}")
        print(f"error info: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n experiment {experiment_name} interrupted by user")
        return False
    except Exception as e:
        print(f"\n experiment {experiment_name} unknown error: {e}")
        return False
    
    return True

def main():
    """main function"""
    print("🚀 RT-DETR training experiment automation script")
    print(f"start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # define experiment configurations
    # TODO: add your experiment configurations here
    experiments = [
        {
            "config": "",
            "name": ""
        },
        {
            "config": "",
            "name": ""
        },
        {
            "config": "",
            "name": ""
        },
        {
            "config": "",
            "name": ""
        }
    ]
    
    # check if config file exists
    print("\ncheck config file...")
    for exp in experiments:
        if not os.path.exists(exp["config"]):
            print(f"config file not found: {exp['config']}")
            return
        else:
            print(f"config file found: {exp['name']}: {exp['config']}")
    
    # run all experiments
    successful_experiments = []
    failed_experiments = []
    
    for i, exp in enumerate(experiments, 1):
        print(f"\nprogress: {i}/{len(experiments)}")
        
        success = run_training(exp["config"], exp["name"])
        
        if success:
            successful_experiments.append(exp["name"])
        else:
            failed_experiments.append(exp["name"])
        
        # add short break between experiments
        if i < len(experiments):
            print(f"\nwait 5 seconds before next experiment...")
            time.sleep(5)
    
    # print summary
    print(f"\n{'='*60}")
    print("experiment summary")
    print(f"{'='*60}")
    print(f"total experiments: {len(experiments)}")
    print(f"successful experiments: {len(successful_experiments)}")
    print(f"failed experiments: {len(failed_experiments)}")
    
    if successful_experiments:
        print(f"\nsuccessful experiments:")
        for exp in successful_experiments:
            print(f"  - {exp}")
    
    if failed_experiments:
        print(f"\nfailed experiments:")
        for exp in failed_experiments:
            print(f"  - {exp}")
    
    print(f"\nend time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 