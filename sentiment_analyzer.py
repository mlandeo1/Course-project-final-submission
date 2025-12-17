"""
SentimentAnalyzer Module

This module contains the SentimentAnalyzer class which performs
sentiment analysis on text datasets using composition with TextDataset.
"""

from text_dataset import TextDataset
from typing import List, Dict
import matplotlib.pyplot as plt
import pandas as pd


class SentimentAnalyzer:
    """
    A class that performs sentiment analysis on text data.
    Uses composition by containing a TextDataset instance.
    
    Attributes:
        dataset (TextDataset): The text dataset to analyze
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
        
        self.dataset = dataset
        self.results = []
        self.sentiment_counts = {'positive': 0, 'neutral': 0, 'negative': 0}
        
        # Ensure dataset is preprocessed
        if not self.dataset.cleaned_texts:
            self.dataset.preprocess()
    
    def analyze_sentiment(self, text: str) -> str:
        """
        Analyze sentiment of a single text using enhanced rule-based approach.
        Uses word frequency analysis and intensity scoring for more accurate results.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            str: Sentiment label ('positive', 'neutral', or 'negative')
        """
        if not isinstance(text, str):
            text = str(text)
        
        text_lower = text.lower()
        
        strong_positive = ['excellent', 'fantastic', 'amazing', 'wonderful', 'love', 
                          'joy', 'delighted', 'ecstatic', 'brilliant', 'perfect']
        moderate_positive = ['good', 'great', 'happy', 'positive', 'nice', 'fine', 
                           'well', 'pleased', 'satisfied', 'content']
        
        strong_negative = ['terrible', 'awful', 'hate', 'depressed', 'anxiety', 
                         'horrible', 'dreadful', 'miserable', 'devastated', 'despair']
        moderate_negative = ['bad', 'sad', 'angry', 'stress', 'worried', 'negative', 
                           'upset', 'frustrated', 'disappointed', 'concerned']
        
        negations = ['not', 'no', 'never', 'none', 'nobody', 'nothing', 'nowhere']
        
        words = text_lower.split()
        pos_score = 0
        neg_score = 0
        
        for i, word in enumerate(words):
            is_negated = i > 0 and words[i-1] in negations
            
            if word in strong_positive:
                pos_score += 2 if not is_negated else -2
            elif word in moderate_positive:
                pos_score += 1 if not is_negated else -1
            elif word in strong_negative:
                neg_score += 2 if not is_negated else -2
            elif word in moderate_negative:
                neg_score += 1 if not is_negated else -1
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
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        sentiments = list(self.sentiment_counts.keys())
        counts = list(self.sentiment_counts.values())
        ax1.bar(sentiments, counts, color=['green', 'gray', 'red'])
        ax1.set_title('Sentiment Distribution (Bar Chart)')
        ax1.set_xlabel('Sentiment')
        ax1.set_ylabel('Count')
        
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

