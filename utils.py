"""
Utility Functions Module

This module contains helper functions for the sentiment analysis project.
"""

import json
from typing import List, Dict, Tuple


def calculate_accuracy(predictions: List[str], actual: List[str]) -> float:
    """
    Calculate the accuracy of sentiment predictions.
    
    Args:
        predictions (List[str]): List of predicted sentiment labels
        actual (List[str]): List of actual sentiment labels
        
    Returns:
        float: Accuracy score between 0 and 1
        
    Raises:
        ValueError: If lists have different lengths or contain invalid values
    """
    if len(predictions) != len(actual):
        raise ValueError("Predictions and actual labels must have the same length")
    
    if len(predictions) == 0:
        raise ValueError("Cannot calculate accuracy for empty lists")
    
    valid_sentiments = {'positive', 'neutral', 'negative'}
    
    # Validate inputs
    for pred in predictions:
        if pred not in valid_sentiments:
            raise ValueError(f"Invalid prediction: {pred}")
    
    for act in actual:
        if act not in valid_sentiments:
            raise ValueError(f"Invalid actual label: {act}")
    
    # Calculate correct predictions
    correct = sum(1 for p, a in zip(predictions, actual) if p == a)
    accuracy = correct / len(predictions)
    
    return accuracy


def generate_report(analyzer, output_file: str = 'sentiment_report.json') -> None:
    """
    Generate a summary report of sentiment analysis results and save to file.
    
    Args:
        analyzer: SentimentAnalyzer instance
        output_file (str): Path to output JSON file
        
    Raises:
        ValueError: If analyzer is invalid or has no results
    """
    if not hasattr(analyzer, 'results') or not analyzer.results:
        raise ValueError("Analyzer has no results. Run analyze_all() first.")
    
    # Get summary statistics
    stats = analyzer.get_summary_statistics()
    
    # Prepare report data
    report = {
        'summary': stats,
        'sentiment_counts': analyzer.sentiment_counts,
        'total_analyzed': len(analyzer.results),
        'sample_results': analyzer.results[:5]  # First 5 results as sample
    }
    
    # Write to JSON file (data I/O)
    try:
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to {output_file}")
    except Exception as e:
        raise ValueError(f"Error writing report: {str(e)}")


# Module-level code using __name__
if __name__ == "__main__":
    # Example usage when run as script
    print("Utils module - use by importing into other modules")

