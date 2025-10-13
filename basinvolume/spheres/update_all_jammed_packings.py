#!/usr/bin/env python
"""
Script to update energy and pressure in all jammed_packings subdirectories
within a given parent directory.

Usage: python update_all_jammed_packings.py /path/to/parent/directory
"""

import os
import sys
import subprocess
import glob
from pathlib import Path
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

def process_config_file(config_path, update_script_path):
    """Process a single config file"""
    try:
        result = subprocess.run(
            [sys.executable, update_script_path, config_path],
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode == 0:
            return f"SUCCESS: {config_path}"
        else:
            return f"FAILED: {config_path} - {result.stderr.strip()}"
    except subprocess.TimeoutExpired:
        return f"TIMEOUT: {config_path}"
    except Exception as e:
        return f"ERROR: {config_path} - {str(e)}"

def process_directory(jammed_dir, update_script_path, max_workers=4):
    """Process all config files in a jammed_packings directory"""
    config_files = glob.glob(os.path.join(jammed_dir, "jammed_packing*.config"))
    if not config_files:
        logging.warning(f"No config files found in {jammed_dir}")
        return
    
    logging.info(f"Processing {len(config_files)} config files in {jammed_dir}")
    
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_config_file, cf, update_script_path): cf 
                  for cf in config_files}
        
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if result.startswith("SUCCESS"):
                logging.debug(result)
            else:
                logging.warning(result)
    
    successful = sum(1 for r in results if r.startswith("SUCCESS"))
    logging.info(f"Completed {jammed_dir}: {successful}/{len(config_files)} successful")

def main():
    if len(sys.argv) != 2:
        print("Usage: python update_all_jammed_packings.py /path/to/parent/directory")
        print("Example: python update_all_jammed_packings.py /scratch/ps4586/FIRE_RUNS_3d_32_new")
        sys.exit(1)
    
    parent_dir = sys.argv[1]
    if not os.path.exists(parent_dir):
        logging.error(f"Directory {parent_dir} does not exist")
        sys.exit(1)
    
    # Find the update_config_energy_pressure.py script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    update_script_path = os.path.join(script_dir, "update_config_energy_pressure.py")
    
    if not os.path.exists(update_script_path):
        logging.error(f"Cannot find update_config_energy_pressure.py in {script_dir}")
        sys.exit(1)
    
    # Find all jammed_packings directories
    jammed_dirs = []
    for root, dirs, files in os.walk(parent_dir):
        if "jammed_packings" in dirs:
            jammed_dir = os.path.join(root, "jammed_packings")
            jammed_dirs.append(jammed_dir)
    
    if not jammed_dirs:
        logging.error(f"No jammed_packings directories found in {parent_dir}")
        sys.exit(1)
    
    logging.info(f"Found {len(jammed_dirs)} jammed_packings directories to process")
    
    # Process each directory
    start_time = time.time()
    for jammed_dir in sorted(jammed_dirs):
        process_directory(jammed_dir, update_script_path)
    
    elapsed = time.time() - start_time
    logging.info(f"All directories processed in {elapsed:.2f} seconds")

if __name__ == "__main__":
    main()