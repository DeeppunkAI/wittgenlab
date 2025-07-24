#!/usr/bin/env python3
"""
Simple test script to verify basic functionality of WittgenLab.

This script tests the core components without requiring external dependencies.
"""

import sys
import os

# Add src to path so we can import wittgenlab
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all core modules can be imported."""
    print("Testing imports...")
    
    try:
        from wittgenlab import EvalHub, EvalConfig, EvalResults, BenchmarkResults
        print("✓ Core imports successful")
        
        from wittgenlab.metrics.base import BaseMetric, ReferenceBasedMetric
        print("✓ Metrics base classes imported")
        
        from wittgenlab.benchmarks.base import BaseBenchmark
        print("✓ Benchmarks base classes imported")
        
        from wittgenlab.core.evaluator import EvalHub
        print("✓ EvalHub imported")
        
        return True
        
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_basic_initialization():
    """Test basic initialization of core components."""
    print("\nTesting initialization...")
    
    try:
        from wittgenlab import EvalHub, EvalConfig
        
        # Test config creation
        config = EvalConfig()
        print("✓ EvalConfig created")
        
        # Test EvalHub creation
        evaluator = EvalHub(config=config)
        print("✓ EvalHub created")
        
        # Test registries
        metrics = evaluator.metrics_registry.list_metrics()
        benchmarks = evaluator.benchmarks_registry.list_benchmarks()
        
        print(f"✓ Found {len(metrics)} metrics")
        print(f"✓ Found {len(benchmarks)} benchmarks")
        
        return True
        
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        return False


def test_metric_evaluation():
    """Test basic metric evaluation functionality."""
    print("\nTesting metric evaluation...")
    
    try:
        from wittgenlab import EvalHub
        
        evaluator = EvalHub()
        
        # Sample data
        predictions = [
            "The cat sat on the mat.",
            "Python is a programming language."
        ]
        
        references = [
            "A cat was sitting on the mat.",
            "Python is a popular programming language."
        ]
        
        # Test BLEU metric (should work with simplified implementation)
        available_metrics = evaluator.metrics_registry.list_metrics()
        
        if 'bleu' in available_metrics:
            results = evaluator.evaluate(
                predictions=predictions,
                references=references,
                metrics=['bleu'],
                task='test'
            )
            
            print(f"✓ BLEU evaluation successful: {results.get_score('bleu'):.4f}")
        else:
            print("⚠ BLEU metric not available")
            
        if 'rouge' in available_metrics:
            results = evaluator.evaluate(
                predictions=predictions,
                references=references,
                metrics=['rouge'],
                task='test'
            )
            
            print(f"✓ ROUGE evaluation successful: {results.get_score('rouge'):.4f}")
        else:
            print("⚠ ROUGE metric not available")
        
        return True
        
    except Exception as e:
        print(f"✗ Metric evaluation failed: {e}")
        return False


def test_results_functionality():
    """Test results saving and loading."""
    print("\nTesting results functionality...")
    
    try:
        from wittgenlab.core.results import EvalResults
        
        # Create sample results
        results = EvalResults(
            scores={'bleu': 0.85, 'rouge': 0.78},
            task='test',
            num_samples=10
        )
        
        print("✓ EvalResults created")
        
        # Test summary
        summary = results.summary
        print(f"✓ Summary generated: {summary}")
        
        # Test JSON conversion
        json_str = results.to_json()
        print("✓ JSON conversion successful")
        
        # Test score access
        bleu_score = results.get_score('bleu')
        print(f"✓ Score access: BLEU = {bleu_score}")
        
        return True
        
    except Exception as e:
        print(f"✗ Results functionality failed: {e}")
        return False


def main():
    """Run all tests."""
    print("WittgenLab Basic Functionality Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_basic_initialization,
        test_metric_evaluation,
        test_results_functionality
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"Tests completed: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! WittgenLab is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Check the output above for details.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 