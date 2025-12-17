"""
Pytest test file for sentiment analysis project.

Tests exception handling and core functionality.
"""

import pytest
import os
import tempfile
import pandas as pd
from text_dataset import TextDataset
from sentiment_analyzer import SentimentAnalyzer
from utils import calculate_accuracy, generate_report


def test_file_not_found_error():
    """
    Test that FileNotFoundError is raised when file doesn't exist.
    """
    with pytest.raises(FileNotFoundError):
        dataset = TextDataset("nonexistent_file.csv")


def test_text_dataset_initialization():
    """
    Test TextDataset initialization with valid file.
    """
    # Create temporary CSV file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("text\n")
        f.write("This is a test sentence\n")
        f.write("Another test sentence\n")
        temp_file = f.name
    
    try:
        dataset = TextDataset(temp_file)
        assert dataset is not None
        assert len(dataset.raw_texts) == 2
        assert dataset.raw_texts[0] == "This is a test sentence"
    finally:
        os.unlink(temp_file)


def test_sentiment_analyzer_composition():
    """
    Test that SentimentAnalyzer uses composition with TextDataset.
    """
    # Create temporary CSV file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("text\n")
        f.write("I feel great and happy\n")
        f.write("I feel terrible and sad\n")
        temp_file = f.name
    
    try:
        dataset = TextDataset(temp_file)
        analyzer = SentimentAnalyzer(dataset)
        
        # Verify composition relationship
        assert isinstance(analyzer.dataset, TextDataset)
        assert analyzer.dataset == dataset
    finally:
        os.unlink(temp_file)


def test_sentiment_analysis_logic():
    """
    Test sentiment analysis classification logic.
    """
    # Create temporary CSV file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("text\n")
        f.write("I feel great and happy\n")
        f.write("I feel terrible and sad\n")
        f.write("This is neutral\n")
        temp_file = f.name
    
    try:
        dataset = TextDataset(temp_file)
        dataset.preprocess()
        analyzer = SentimentAnalyzer(dataset)
        analyzer.analyze_all()
        
        # Check that results were generated
        assert len(analyzer.results) == 3
        assert analyzer.sentiment_counts['positive'] > 0 or analyzer.sentiment_counts['negative'] > 0
    finally:
        os.unlink(temp_file)


def test_calculate_accuracy():
    """
    Test accuracy calculation function.
    """
    predictions = ['positive', 'negative', 'neutral']
    actual = ['positive', 'negative', 'neutral']
    
    accuracy = calculate_accuracy(predictions, actual)
    assert accuracy == 1.0
    
    # Test with mismatched predictions
    predictions2 = ['positive', 'negative', 'positive']
    actual2 = ['positive', 'negative', 'neutral']
    accuracy2 = calculate_accuracy(predictions2, actual2)
    assert accuracy2 == pytest.approx(0.666, abs=0.01)


def test_calculate_accuracy_value_error():
    """
    Test that ValueError is raised for invalid inputs.
    """
    predictions = ['positive', 'negative']
    actual = ['positive', 'negative', 'neutral']  # Different length
    
    with pytest.raises(ValueError):
        calculate_accuracy(predictions, actual)


def test_operator_overloading():
    """
    Test operator overloading (__add__) in TextDataset.
    """
    # Create two temporary CSV files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f1:
        f1.write("text\n")
        f1.write("First dataset text\n")
        temp_file1 = f1.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f2:
        f2.write("text\n")
        f2.write("Second dataset text\n")
        temp_file2 = f2.name
    
    try:
        dataset1 = TextDataset(temp_file1)
        dataset2 = TextDataset(temp_file2)
        
        # Test operator overloading
        merged = dataset1 + dataset2
        assert len(merged.raw_texts) == 2
    finally:
        os.unlink(temp_file1)
        os.unlink(temp_file2)


def test_generator_function():
    """
    Test generator function in TextDataset.
    """
    # Create temporary CSV file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("text\n")
        for i in range(25):
            f.write(f"Text number {i}\n")
        temp_file = f.name
    
    try:
        dataset = TextDataset(temp_file)
        dataset.preprocess()
        
        # Test generator
        batches = list(dataset.batch_generator(batch_size=10))
        assert len(batches) == 3  # 25 texts / 10 batch size = 3 batches
        assert len(batches[0]) == 10
    finally:
        os.unlink(temp_file)

