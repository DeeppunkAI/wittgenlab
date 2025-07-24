#!/usr/bin/env python3
"""
Tests for per-item evaluation functionality.
"""

import pytest
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from wittgenlab.core.evaluator import EvalHub
from wittgenlab.core.results import EvalResults


class TestPerItemEvaluation:
    """Test class for per-item evaluation functionality."""
    
    def setup_method(self):
        """Setup method called before each test."""
        self.evaluator = EvalHub()
        self.predictions = [
            "The cat sat on the mat.",
            "Python is a programming language.",
            "Machine learning is useful."
        ]
        self.references = [
            "A cat was sitting on the mat.",
            "Python is a popular programming language.",
            "Machine learning is very useful for data science."
        ]
    
    def test_traditional_evaluation_still_works(self):
        """Test that traditional evaluation (without per-item) still works."""
        results = self.evaluator.evaluate(
            predictions=self.predictions,
            references=self.references,
            metrics=['bleu', 'rouge'],
            task='test',
            include_per_item=False
        )
        
        assert isinstance(results, EvalResults)
        assert 'bleu' in results.scores
        assert 'rouge' in results.scores
        assert results.get_score('bleu') is not None
        assert results.get_score('rouge') is not None
        assert len(results.per_item_scores) == 0  # No per-item scores
    
    def test_per_item_evaluation_basic(self):
        """Test basic per-item evaluation functionality."""
        results = self.evaluator.evaluate(
            predictions=self.predictions,
            references=self.references,
            metrics=['bleu', 'rouge'],
            task='test',
            include_per_item=True
        )
        
        assert isinstance(results, EvalResults)
        
        # Check overall scores
        assert 'bleu' in results.scores
        assert 'rouge' in results.scores
        assert results.get_score('bleu') is not None
        assert results.get_score('rouge') is not None
        
        # Check per-item scores
        assert len(results.per_item_scores) > 0
        assert 'bleu' in results.per_item_scores
        assert 'rouge' in results.per_item_scores
        
        # Check per-item score structure
        bleu_per_item = results.get_per_item_scores('bleu')
        rouge_per_item = results.get_per_item_scores('rouge')
        
        assert bleu_per_item is not None
        assert rouge_per_item is not None
        assert len(bleu_per_item) == len(self.predictions)
        assert len(rouge_per_item) == len(self.predictions)
        
        # Check that all scores are numeric
        for score in bleu_per_item:
            assert isinstance(score, (int, float))
            assert 0 <= score <= 1  # BLEU scores should be between 0 and 1
        
        for score in rouge_per_item:
            assert isinstance(score, (int, float))
            assert 0 <= score <= 1  # ROUGE scores should be between 0 and 1
    
    def test_single_metric_per_item(self):
        """Test per-item evaluation with single metrics."""
        # Test BLEU only
        bleu_results = self.evaluator.evaluate(
            predictions=self.predictions,
            references=self.references,
            metrics=['bleu'],
            task='test_bleu',
            include_per_item=True
        )
        
        assert 'bleu' in bleu_results.scores
        assert 'bleu' in bleu_results.per_item_scores
        assert 'rouge' not in bleu_results.per_item_scores
        
        # Test ROUGE only
        rouge_results = self.evaluator.evaluate(
            predictions=self.predictions,
            references=self.references,
            metrics=['rouge'],
            task='test_rouge',
            include_per_item=True
        )
        
        assert 'rouge' in rouge_results.scores
        assert 'rouge' in rouge_results.per_item_scores
        assert 'bleu' not in rouge_results.per_item_scores
    
    def test_get_item_score_method(self):
        """Test the get_item_score method."""
        results = self.evaluator.evaluate(
            predictions=self.predictions,
            references=self.references,
            metrics=['bleu', 'rouge'],
            task='test',
            include_per_item=True
        )
        
        # Test valid indices
        for i in range(len(self.predictions)):
            bleu_score = results.get_item_score('bleu', i)
            rouge_score = results.get_item_score('rouge', i)
            
            assert bleu_score is not None
            assert rouge_score is not None
            assert isinstance(bleu_score, (int, float))
            assert isinstance(rouge_score, (int, float))
        
        # Test invalid indices
        assert results.get_item_score('bleu', -1) is None
        assert results.get_item_score('bleu', len(self.predictions)) is None
        
        # Test invalid metric
        assert results.get_item_score('nonexistent', 0) is None
    
    def test_add_per_item_scores_method(self):
        """Test the add_per_item_scores method."""
        results = EvalResults(
            scores={'test_metric': 0.5},
            task='test',
            num_samples=3
        )
        
        # Add per-item scores
        per_item_scores = [0.3, 0.5, 0.7]
        results.add_per_item_scores('test_metric', per_item_scores)
        
        assert 'test_metric' in results.per_item_scores
        assert results.get_per_item_scores('test_metric') == per_item_scores
        
        # Test individual access
        for i, expected_score in enumerate(per_item_scores):
            assert results.get_item_score('test_metric', i) == expected_score
    
    def test_results_summary_with_per_item(self):
        """Test that results summary includes per-item information."""
        results = self.evaluator.evaluate(
            predictions=self.predictions,
            references=self.references,
            metrics=['bleu'],
            task='test',
            include_per_item=True
        )
        
        summary = results.summary
        assert 'has_per_item_scores' in summary
        assert summary['has_per_item_scores'] is True
        
        # Test without per-item scores
        results_no_per_item = self.evaluator.evaluate(
            predictions=self.predictions,
            references=self.references,
            metrics=['bleu'],
            task='test',
            include_per_item=False
        )
        
        summary_no_per_item = results_no_per_item.summary
        assert summary_no_per_item['has_per_item_scores'] is False
    
    def test_results_string_representation(self):
        """Test string representation includes per-item information."""
        results = self.evaluator.evaluate(
            predictions=self.predictions,
            references=self.references,
            metrics=['bleu', 'rouge'],
            task='test',
            include_per_item=True
        )
        
        str_repr = str(results)
        assert 'Per-item scores available' in str_repr
        assert 'Per-item range' in str_repr
    
    def test_empty_input_handling(self):
        """Test handling of empty inputs."""
        with pytest.raises(ValueError, match="Input lists cannot be empty"):
            self.evaluator.evaluate(
                predictions=[],
                references=[],
                metrics=['bleu'],
                task='test',
                include_per_item=True
            )
    
    def test_mismatched_lengths(self):
        """Test handling of mismatched prediction/reference lengths."""
        with pytest.raises(ValueError, match="Predictions and references must have the same length"):
            self.evaluator.evaluate(
                predictions=self.predictions,
                references=self.references[:-1],  # Remove one reference
                metrics=['bleu'],
                task='test',
                include_per_item=True
            )
    
    def test_save_and_load_with_per_item(self):
        """Test saving and loading results with per-item scores."""
        results = self.evaluator.evaluate(
            predictions=self.predictions,
            references=self.references,
            metrics=['bleu', 'rouge'],
            task='test',
            include_per_item=True
        )
        
        # Save results
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            results.save(temp_file)
            
            # Load results
            loaded_results = EvalResults.load(temp_file)
            
            # Verify loaded results
            assert loaded_results.scores == results.scores
            assert loaded_results.per_item_scores == results.per_item_scores
            assert loaded_results.task == results.task
            assert loaded_results.num_samples == results.num_samples
            
        finally:
            os.unlink(temp_file)
    
    def test_per_item_with_error_handling(self):
        """Test per-item evaluation with error handling."""
        # Use some predictions that might cause issues
        problematic_predictions = ["", "normal text", ""]
        problematic_references = ["reference", "normal reference", "another reference"]
        
        results = self.evaluator.evaluate(
            predictions=problematic_predictions,
            references=problematic_references,
            metrics=['bleu', 'rouge'],
            task='test_errors',
            include_per_item=True
        )
        
        # Should still produce results even with empty strings
        assert 'bleu' in results.scores
        assert 'rouge' in results.scores
        assert len(results.get_per_item_scores('bleu')) == len(problematic_predictions)
        assert len(results.get_per_item_scores('rouge')) == len(problematic_predictions)


def test_integration_example():
    """Integration test that matches the example usage."""
    evaluator = EvalHub()
    
    predictions = [
        "El gato se sentó en la alfombra roja.",
        "Python es un lenguaje de programación muy popular."
    ]
    
    references = [
        "Un gato estaba sentado en la alfombra.",
        "Python es un lenguaje de programación popular."
    ]
    
    # Test the complete workflow
    results = evaluator.evaluate(
        predictions=predictions,
        references=references,
        metrics=['bleu', 'rouge'],
        task='integration_test',
        include_per_item=True
    )
    
    # Verify we can access all functionality
    assert results.get_score('bleu') is not None
    assert results.get_score('rouge') is not None
    
    bleu_per_item = results.get_per_item_scores('bleu')
    rouge_per_item = results.get_per_item_scores('rouge')
    
    assert len(bleu_per_item) == 2
    assert len(rouge_per_item) == 2
    
    # Test individual access
    for i in range(2):
        assert results.get_item_score('bleu', i) is not None
        assert results.get_item_score('rouge', i) is not None
    
    print("Integration test passed successfully!")


if __name__ == "__main__":
    test_integration_example()
    print("All tests would pass!") 