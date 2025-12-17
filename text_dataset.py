"""
TextDataset Module

This module contains the TextDataset class which handles reading,
cleaning, and preprocessing text data from CSV files.
"""

import pandas as pd
from typing import List, Generator, Tuple
import os


class TextDataset:
    """
    A class to handle text dataset operations including loading,
    cleaning, and preprocessing text data.
    
    Attributes:
        file_path (str): Path to the CSV file containing text data
        data (pd.DataFrame): The loaded and processed dataset
        raw_texts (List[str]): List of raw text entries
        cleaned_texts (List[str]): List of cleaned text entries
    """
    
    def __init__(self, file_path: str):
        """
        Initialize TextDataset with a file path.
        
        Args:
            file_path (str): Path to the CSV file
            
        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the file is empty or invalid
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            # Read CSV file using pandas
            self.data = pd.read_csv(file_path)
            
            if self.data.empty:
                raise ValueError("Dataset is empty")
            
            # Extract text column (assuming 'text' column exists)
            if 'text' not in self.data.columns:
                # Try common alternatives
                text_col = None
                for col in ['Text', 'TEXT', 'content', 'Content', 'message', 'Message']:
                    if col in self.data.columns:
                        text_col = col
                        break
                
                if text_col is None:
                    raise ValueError("No text column found in dataset")
                self.raw_texts = self.data[text_col].tolist()
            else:
                self.raw_texts = self.data['text'].tolist()
            
            self.cleaned_texts = []
            self.file_path = file_path
            self._temp_file_path = None
            
        except pd.errors.EmptyDataError:
            raise ValueError("File is empty or corrupted")
        except Exception as e:
            raise ValueError(f"Error reading file: {str(e)}")
    
    def clean_text(self, text: str) -> str:
        """
        Clean a single text string by removing extra whitespace
        and converting to lowercase.
        
        Args:
            text (str): Raw text string
            
        Returns:
            str: Cleaned text string
        """
        if not isinstance(text, str):
            text = str(text)
        # Remove extra whitespace and convert to lowercase
        cleaned = ' '.join(text.split()).lower()
        return cleaned
    
    def preprocess(self) -> None:
        """
        Preprocess all texts in the dataset using map and lambda.
        Stores cleaned texts in the cleaned_texts list.
        """
        self.cleaned_texts = list(map(lambda x: self.clean_text(x), self.raw_texts))
    
    def filter_tokens(self, min_length: int = 3) -> List[str]:
        """
        Filter tokens from cleaned texts using list comprehension.
        
        Args:
            min_length (int): Minimum token length to keep
            
        Returns:
            List[str]: List of filtered tokens
        """
        all_tokens = []
        for text in self.cleaned_texts:
            tokens = [token for token in text.split() if len(token) >= min_length]
            all_tokens.extend(tokens)
        return all_tokens
    
    def batch_generator(self, batch_size: int = 10) -> Generator[List[str], None, None]:
        """
        Generator function that yields batches of text data.
        Useful for processing large datasets efficiently.
        
        Args:
            batch_size (int): Number of texts per batch
            
        Yields:
            List[str]: Batch of text strings
        """
        if not self.cleaned_texts:
            self.preprocess()
        
        for i in range(0, len(self.cleaned_texts), batch_size):
            yield self.cleaned_texts[i:i + batch_size]
    
    def get_word_frequencies(self) -> dict:
        """
        Calculate word frequencies from cleaned texts.
        
        Returns:
            dict: Dictionary mapping words to their frequencies
        """
        if not self.cleaned_texts:
            self.preprocess()
        
        freq_dict = {}
        for text in self.cleaned_texts:
            for word in text.split():
                freq_dict[word] = freq_dict.get(word, 0) + 1
        return freq_dict
    
    def get_sentiment_categories(self) -> Tuple[str, ...]:
        """
        Return sentiment categories as a tuple.
        
        Returns:
            Tuple[str, ...]: Tuple of sentiment category names
        """
        categories = ('positive', 'neutral', 'negative')
        return categories
    
    def __add__(self, other: 'TextDataset') -> 'TextDataset':
        """
        Operator overloading: merge two TextDataset objects.
        Creates a new dataset combining texts from both.
        
        Args:
            other (TextDataset): Another TextDataset instance
            
        Returns:
            TextDataset: New merged dataset
        """
        if not isinstance(other, TextDataset):
            raise ValueError("Can only merge with another TextDataset")
        
        import tempfile
        import pandas as pd
        
        # Combine dataframes
        combined_data = pd.concat([self.data, other.data], ignore_index=True)
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        temp_file_path = temp_file.name
        combined_data.to_csv(temp_file_path, index=False)
        temp_file.close()
        
        # Create new merged dataset
        merged = TextDataset(temp_file_path)
        # Store temp file path for potential cleanup
        merged._temp_file_path = temp_file_path
        return merged
    
    def __str__(self) -> str:
        """
        String representation of TextDataset.
        
        Returns:
            str: Human-readable description
        """
        num_texts = len(self.raw_texts) if self.raw_texts else 0
        return f"TextDataset(file_path='{self.file_path}', num_texts={num_texts})"
    
    def __repr__(self) -> str:
        """Return string representation."""
        return self.__str__()
    
    def cleanup_temp_file(self) -> None:
        """
        Clean up temporary file if this dataset was created from a merge operation.
        Only call this when you're done using the merged dataset.
        """
        if hasattr(self, '_temp_file_path') and self._temp_file_path and os.path.exists(self._temp_file_path):
            try:
                os.unlink(self._temp_file_path)
                self._temp_file_path = None
            except Exception:
                pass  # Ignore cleanup errors


# Module-level code using __name__
if __name__ == "__main__":
    # Example usage when run as script
    print("TextDataset module - use by importing into other modules")

