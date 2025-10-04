#!/usr/bin/env python3
"""
RecessionScope Test Runner
Consolidated test runner for all test types with clean output
"""

import subprocess
import sys
import os
import re
from pathlib import Path

def run_tests(test_type="all"):
    """Run tests with clean output format"""
    
    # Ensure we're in the backend directory
    backend_dir = Path(__file__).parent.parent
    os.chdir(backend_dir)
    
    # Define test configurations
    test_configs = {
        "failover": {
            "marker": "failover",
            "title": "RECESSIONSCOPE FAILOVER TESTS"
        },
        "config": {
            "marker": "config", 
            "title": "RECESSIONSCOPE CONFIGURATION TESTS"
        },
        "all": {
            "marker": None,
            "title": "RECESSIONSCOPE ALL TESTS"
        }
    }
    
    if test_type not in test_configs:
        print(f"❌ Unknown test type: {test_type}")
        print(f"Available types: {', '.join(test_configs.keys())}")
        return 1
    
    config = test_configs[test_type]
    
    # Display header
    print(config["title"])
    print("=" * len(config["title"]))
    
    # Build pytest command
    cmd = [sys.executable, "-m", "pytest", "-q", "--tb=no", "--disable-warnings", "--no-header"]
    
    if config["marker"]:
        cmd.extend(["-m", config["marker"]])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        
        # Extract test counts from output
        passed_count = len(re.findall(r'\.', result.stdout))
        failed_count = len(re.findall(r'F', result.stdout))
        
        # Alternative parsing from summary line
        for line in result.stdout.split('\n'):
            if 'passed' in line or 'failed' in line:
                pass_match = re.search(r'(\d+) passed', line)
                fail_match = re.search(r'(\d+) failed', line)
                if pass_match:
                    passed_count = int(pass_match.group(1))
                if fail_match:
                    failed_count = int(fail_match.group(1))
        
        total_count = passed_count + failed_count
        
        # Display results
        if total_count > 0:
            print(f"\nTest Results:")
            print(f"✅ {passed_count} passed")
            if failed_count > 0:
                print(f"❌ {failed_count} failed")
            print(f"Total: {total_count}")
            
            # Overall status
            status = "PASSED" if failed_count == 0 else "FAILED"
            icon = "✅" if failed_count == 0 else "❌"
            print(f"\n{icon} ALL {test_type.upper()} TESTS {status}")
        else:
            print("\n❌ No tests found or executed")
        
        return result.returncode
        
    except KeyboardInterrupt:
        print("\n❌ Tests interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return 1

def main():
    """Main entry point"""
    test_type = sys.argv[1].lower() if len(sys.argv) > 1 else "all"
    return run_tests(test_type)

if __name__ == "__main__":
    sys.exit(main())