#!/usr/bin/env python3
"""
Wrapper script to run Main_File.py with proper error handling
"""
import sys
import os

# Change to notebooks directory
os.chdir(os.path.join(os.path.dirname(__file__), 'notebooks'))

print("=" * 80)
print("Running Main_File.py - mAInXance Assistant")
print("=" * 80)
print("\nNote: This will take several minutes to complete.")
print("The script will:")
print("  1. Load and clean maintenance data")
print("  2. Label maintenance actions")
print("  3. Create visualizations")
print("  4. Train multiple ML models")
print("  5. Generate predictions and narratives")
print("\n" + "=" * 80 + "\n")

try:
    # Execute Main_File.py
    with open('Main_File.py', 'r') as f:
        code = f.read()
    
    exec(code, {'__file__': 'Main_File.py'})
    
    print("\n" + "=" * 80)
    print("✓ Main_File.py completed successfully!")
    print("=" * 80)
    
except Exception as e:
    print("\n" + "=" * 80)
    print(f"✗ Error during execution: {type(e).__name__}")
    print(f"  {str(e)}")
    print("=" * 80)
    import traceback
    traceback.print_exc()
    sys.exit(1)
