"""
Simple test runner to validate modular services.
"""
import sys
import importlib
import inspect
import pytest
import traceback

def run_test(module_name, test_function_name):
    """Run an individual test function and report its status."""
    try:
        # Import the module
        module = importlib.import_module(module_name)
        
        # Get the test function
        test_function = getattr(module, test_function_name)
        
        # Configure pytest for the test
        args = []
        if 'monkeypatch' in inspect.signature(test_function).parameters:
            monkeypatch = pytest.MonkeyPatch()
            args.append(monkeypatch)
        if 'mock_config' in inspect.signature(test_function).parameters:
            # Get the fixture function
            fixture_func = getattr(module, 'mock_config')
            # Create a monkeypatch for the fixture
            mp = pytest.MonkeyPatch()
            # Call the fixture function with the monkeypatch
            fixture_result = fixture_func(mp)
            # If the fixture returns a value, add it to args
            if fixture_result is not None:
                args.append(fixture_result)
        
        # Run the test
        test_function(*args)
        return True, None
    except Exception as e:
        return False, traceback.format_exc()

def get_test_functions(module_name):
    """Get all test functions from a module."""
    module = importlib.import_module(module_name)
    test_functions = []
    
    for name, obj in inspect.getmembers(module):
        if name.startswith('test_') and callable(obj):
            test_functions.append(name)
    
    return test_functions

def main():
    """Run all tests from test_modular_services."""
    module_name = 'tests.test_modular_services'
    test_functions = get_test_functions(module_name)
    
    # Run all tests and track results
    results = []
    
    print(f"\nRunning {len(test_functions)} tests from {module_name}...\n")
    
    for test_name in test_functions:
        print(f"Running {test_name}...", end=" ")
        success, error_msg = run_test(module_name, test_name)
        
        if success:
            print("PASSED ✓")
        else:
            print("FAILED ✗")
            print("-" * 80)
            print(error_msg)
            print("-" * 80)
        
        results.append((test_name, success))
    
    # Summarize results
    passed = sum(1 for _, success in results if success)
    failed = len(results) - passed
    
    print(f"\nTest Summary: {passed} passed, {failed} failed")
    
    if failed > 0:
        print("\nFailed tests:")
        for test_name, success in results:
            if not success:
                print(f"  - {test_name}")
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
