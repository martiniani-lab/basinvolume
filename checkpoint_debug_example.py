#!/usr/bin/env python3
"""
Example script demonstrating the improved checkpoint system for PT_Master
This script shows how to use different checkpoint formats and debug checkpoint issues.
"""

import os
import logging
import numpy as np
from basinvolume.spheres._pt_master import PT_Master

# Set up logging to see detailed checkpoint information
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_mock_mcrunner():
    """Create a mock MCRunner for testing purposes"""
    class MockMCRunner:
        def __init__(self):
            self.nparticles = 10
            self.bdim = 3
            self.niter = 1000
            self.equilibration_steps = 500
            self.potential = None
            
        def get_complete_state(self):
            """Return a mock state"""
            class MockState:
                def __init__(self):
                    self.bias_params = 1.0
                    self.stepsize = 0.1
                    self.takestep_count = 0
                    self.energy = -10.0
                    self.coords = np.random.random(30)  # 10 particles * 3 dimensions
                    self.counters = np.array([0, 0, 0, 0, 0], dtype='uintp')
                    self.step_adaptation_counters = np.array([0, 0], dtype='uintp')
            return MockState()
    
    return MockMCRunner()

def test_checkpoint_formats():
    """Test different checkpoint formats"""
    print("Testing different checkpoint formats...")
    
    example_mcrunner = create_mock_mcrunner()
    
    # Test different formats
    formats_to_test = [
        ("checkpoint_pickle.pkl", "pickle"),
        ("checkpoint_json.json", "json"),
    ]
    
    # Add joblib and dill if available
    try:
        import joblib
        formats_to_test.append(("checkpoint_joblib.joblib", "joblib"))
    except ImportError:
        print("joblib not available - skipping joblib test")
    
    try:
        import dill
        formats_to_test.append(("checkpoint_dill.dill", "dill"))
    except ImportError:
        print("dill not available - skipping dill test")
    
    for checkpoint_file, format_type in formats_to_test:
        print(f"\n--- Testing {format_type} format ---")
        
        try:
            # Create PT_Master instance
            pt_master = PT_Master(
                nreplicas=4,
                example_mcrunner=example_mcrunner,
                kmax=10.0,
                kmin=0.1,
                u2meank0=1.0,
                max_ptiter=2,  # Very short for testing
                checkpoint_file=checkpoint_file,
                checkpoint_format=format_type,
                base_directory="test_checkpoints"
            )
            
            # Simulate some progress
            pt_master.ptiter = 1
            pt_master.anyswap = True
            pt_master.eq_time = 100
            
            # Create checkpoint
            print(f"Creating checkpoint with {format_type}...")
            pt_master._create_checkpoint()
            
            # Verify file exists and has content
            checkpoint_path = os.path.join(pt_master.base_directory, checkpoint_file)
            if os.path.exists(checkpoint_path):
                size = os.path.getsize(checkpoint_path)
                print(f"✓ Checkpoint created successfully: {size} bytes")
                
                # Test loading
                print(f"Loading checkpoint with {format_type}...")
                loaded_pt = PT_Master.load_checkpoint(checkpoint_path, example_mcrunner)
                print(f"✓ Checkpoint loaded successfully")
                print(f"  - ptiter: {loaded_pt.ptiter}")
                print(f"  - nreplicas: {loaded_pt.nreplicas}")
                print(f"  - eq_time: {loaded_pt.eq_time}")
                
            else:
                print(f"✗ Checkpoint file not created: {checkpoint_path}")
                
        except Exception as e:
            print(f"✗ Error with {format_type}: {e}")
            logging.exception(f"Detailed error for {format_type}")

def debug_corrupted_checkpoint(checkpoint_path):
    """Debug a potentially corrupted checkpoint file"""
    print(f"\n--- Debugging checkpoint: {checkpoint_path} ---")
    
    if not os.path.exists(checkpoint_path):
        print(f"✗ Checkpoint file does not exist: {checkpoint_path}")
        return
    
    file_size = os.path.getsize(checkpoint_path)
    print(f"File size: {file_size} bytes")
    
    if file_size == 0:
        print("✗ File is empty - this indicates a write failure")
        return
    
    # Try to read first few bytes to check format
    try:
        with open(checkpoint_path, 'rb') as f:
            first_bytes = f.read(10)
            print(f"First 10 bytes (hex): {first_bytes.hex()}")
            print(f"First 10 bytes (ascii): {first_bytes}")
    except Exception as e:
        print(f"✗ Cannot read file: {e}")
        return
    
    # Try to determine format and load
    ext = os.path.splitext(checkpoint_path)[1].lower()
    
    if ext == '.json':
        try:
            import json
            with open(checkpoint_path, 'r') as f:
                data = json.load(f)
            print("✓ JSON checkpoint is valid")
        except json.JSONDecodeError as e:
            print(f"✗ JSON checkpoint is corrupted: {e}")
        except Exception as e:
            print(f"✗ Error reading JSON: {e}")
    
    elif ext == '.joblib':
        try:
            import joblib
            data = joblib.load(checkpoint_path)
            print("✓ Joblib checkpoint is valid")
        except Exception as e:
            print(f"✗ Joblib checkpoint is corrupted: {e}")
    
    elif ext == '.dill':
        try:
            import dill
            with open(checkpoint_path, 'rb') as f:
                data = dill.load(f)
            print("✓ Dill checkpoint is valid")
        except Exception as e:
            print(f"✗ Dill checkpoint is corrupted: {e}")
    
    else:
        # Try pickle
        try:
            import pickle
            with open(checkpoint_path, 'rb') as f:
                data = pickle.load(f)
            print("✓ Pickle checkpoint is valid")
        except Exception as e:
            print(f"✗ Pickle checkpoint is corrupted: {e}")

def demonstrate_auto_format_detection():
    """Demonstrate automatic format detection"""
    print("\n--- Demonstrating auto format detection ---")
    
    example_mcrunner = create_mock_mcrunner()
    
    # Test auto detection with different file extensions
    test_files = [
        "auto_test.pkl",     # Should use pickle
        "auto_test.json",    # Should use json
        "auto_test.joblib",  # Should use joblib (if available)
        "auto_test.dill",    # Should use dill (if available)
        "auto_test.xyz",     # Should default to pickle
    ]
    
    for test_file in test_files:
        try:
            pt_master = PT_Master(
                nreplicas=3,
                example_mcrunner=example_mcrunner,
                kmax=5.0,
                kmin=0.1,
                u2meank0=1.0,
                max_ptiter=1,
                checkpoint_file=test_file,
                checkpoint_format="auto",  # This triggers auto-detection
                base_directory="auto_test_checkpoints"
            )
            
            detected_format = pt_master._get_checkpoint_format(test_file)
            print(f"{test_file} -> detected format: {detected_format}")
            
        except Exception as e:
            print(f"{test_file} -> error: {e}")

if __name__ == "__main__":
    print("Checkpoint Debug and Testing Script")
    print("=" * 50)
    
    # Test different checkpoint formats
    test_checkpoint_formats()
    
    # Demonstrate auto format detection
    demonstrate_auto_format_detection()
    
    # If you have a corrupted checkpoint, uncomment and modify this line:
    # debug_corrupted_checkpoint("/path/to/your/corrupted/checkpoint.dmp")
    
    print("\n" + "=" * 50)
    print("Testing complete!")
    print("\nUsage recommendations:")
    print("1. Use JSON format for debugging and human-readable checkpoints")
    print("2. Use pickle for maximum compatibility")
    print("3. Use joblib for better compression and performance")
    print("4. Use dill for complex objects that pickle can't handle")
    print("5. Set checkpoint_format='auto' to auto-detect from file extension") 