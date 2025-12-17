"""
SentimentAnalyzer Module

This module contains the SentimentAnalyzer class which performs
sentiment analysis on text datasets using composition with TextDataset.
"""

from text_dataset import TextDataset
from typing import List, Dict
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter  # Built-in library module


class SentimentAnalyzer:
    """
    A class that performs sentiment analysis on text data.
    Uses composition by containing a TextDataset instance.
    
    Attributes:
        dataset (TextDataset): The text dataset to analyze (composition)
        results (List[Dict]): List of sentiment analysis results
        sentiment_counts (Dict[str, int]): Count of each sentiment category
    """
    
    def __init__(self, dataset: TextDataset):
        """
        Initialize SentimentAnalyzer with a TextDataset instance.
        
        Args:
            dataset (TextDataset): TextDataset object to analyze
            
        Raises:
            ValueError: If dataset is invalid
        """
        if not isinstance(dataset, TextDataset):
            raise ValueError("dataset must be a TextDataset instance")
        
        # Composition: SentimentAnalyzer contains TextDataset
        self.dataset = dataset
        self.results = []  # Mutable list
        self.sentiment_counts = {'positive': 0, 'neutral': 0, 'negative': 0}  # Mutable dict
        
        # Ensure dataset is preprocessed
        if not self.dataset.cleaned_texts:
            self.dataset.preprocess()
    
    def analyze_sentiment(self, text: str) -> str:
        """
        Analyze sentiment of a single text using enhanced rule-based approach.
        Uses word frequency analysis and intensity scoring for more accurate results.
        
        Args:
            text (str): Text to analyze (immutable string)
            
        Returns:
            str: Sentiment label ('positive', 'neutral', or 'negative')
        """
        if not isinstance(text, str):
            text = str(text)
        
        text_lower = text.lower()
        
        # Enhanced sentiment word lists with intensity weights
        # Strong positive words (weight 2)
        strong_positive = ['excellent', 'fantastic', 'amazing', 'wonderful', 'love', 
                          'joy', 'delighted', 'ecstatic', 'brilliant', 'perfect']
        # Moderate positive words (weight 1)
        moderate_positive = ['good', 'great', 'happy', 'positive', 'nice', 'fine', 
                           'well', 'pleased', 'satisfied', 'content']
        
        # Strong negative words (weight 2)
        strong_negative = ['terrible', 'awful', 'hate', 'depressed', 'anxiety', 
                         'horrible', 'dreadful', 'miserable', 'devastated', 'despair']
        # Moderate negative words (weight 1)
        moderate_negative = ['bad', 'sad', 'angry', 'stress', 'worried', 'negative', 
                           'upset', 'frustrated', 'disappointed', 'concerned']
        
        # Negation words that flip sentiment
        negations = ['not', 'no', 'never', 'none', 'nobody', 'nothing', 'nowhere']
        
        # Split text into words for better matching
        words = text_lower.split()
        
        # Calculate sentiment scores with intensity weighting
        pos_score = 0
        neg_score = 0
        
        for i, word in enumerate(words):
            # Check for negations (simple: if previous word is negation, flip sentiment)
            is_negated = i > 0 and words[i-1] in negations
            
            # Check strong positive
            if word in strong_positive:
                pos_score += 2 if not is_negated else -2
            # Check moderate positive
            elif word in moderate_positive:
                pos_score += 1 if not is_negated else -1
            # Check strong negative
            elif word in strong_negative:
                neg_score += 2 if not is_negated else -2
            # Check moderate negative
            elif word in moderate_negative:
                neg_score += 1 if not is_negated else -1
        
        # Determine sentiment with threshold to avoid ties
        if pos_score > neg_score and pos_score > 0:
            return 'positive'
        elif neg_score > pos_score and neg_score > 0:
            return 'negative'
        else:
            return 'neutral'
    
    def analyze_all(self) -> None:
        """
        Analyze sentiment for all texts in the dataset.
        Stores results in the results list.
        """
        self.results = []
        self.sentiment_counts = {'positive': 0, 'neutral': 0, 'negative': 0}
        
        # For loop to iterate through all texts
        for text in self.dataset.cleaned_texts:
            sentiment = self.analyze_sentiment(text)
            result = {
                'text': text,
                'sentiment': sentiment
            }
            self.results.append(result)
            self.sentiment_counts[sentiment] += 1
    
    def visualize_results(self, save_path: str = 'sentiment_distribution.png') -> None:
        """
        Create visualizations of sentiment analysis results using matplotlib.
        
        Args:
            save_path (str): Path to save the visualization image
        """
        if not self.results:
            self.analyze_all()
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Bar chart
        sentiments = list(self.sentiment_counts.keys())
        counts = list(self.sentiment_counts.values())
        ax1.bar(sentiments, counts, color=['green', 'gray', 'red'])
        ax1.set_title('Sentiment Distribution (Bar Chart)')
        ax1.set_xlabel('Sentiment')
        ax1.set_ylabel('Count')
        
        # Pie chart
        ax2.pie(counts, labels=sentiments, autopct='%1.1f%%', 
                colors=['green', 'gray', 'red'])
        ax2.set_title('Sentiment Distribution (Pie Chart)')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Visualization saved to {save_path}")
    
    def get_summary_statistics(self) -> Dict[str, float]:
        """
        Calculate summary statistics for sentiment analysis.
        
        Returns:
            Dict[str, float]: Dictionary with statistics
        """
        if not self.results:
            self.analyze_all()
        
        total = len(self.results)
        if total == 0:
            return {}
        
        stats = {
            'total_texts': total,
            'positive_percentage': (self.sentiment_counts['positive'] / total) * 100,
            'neutral_percentage': (self.sentiment_counts['neutral'] / total) * 100,
            'negative_percentage': (self.sentiment_counts['negative'] / total) * 100
        }
        return stats
    
    def __str__(self) -> str:
        """
        String representation of SentimentAnalyzer.
        
        Returns:
            str: Human-readable description
        """
        total = len(self.results) if self.results else 0
        return f"SentimentAnalyzer(dataset={self.dataset}, analyzed_texts={total})"
    
    def __repr__(self) -> str:
        """Return string representation."""
        return self.__str__()


# Module-level code using __name__
if __name__ == "__main__":
    # Example usage when run as script
    print("SentimentAnalyzer module - use by importing into other modules")

