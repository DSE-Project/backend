#!/usr/bin/env python3
"""
Convenience script to run tests from backend directory
Delegates to the consolidated test runner in tests/
"""

import sys
import subprocess
from pathlib import Path

def main():
    """Run tests using the consolidated runner"""
    backend_dir = Path(__file__).parent
    test_runner = backend_dir / "tests" / "run_tests.py"
    
    # Pass all arguments to the consolidated runner
    cmd = [sys.executable, str(test_runner)] + sys.argv[1:]
    
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())