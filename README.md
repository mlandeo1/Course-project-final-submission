# Mental Health Sentiment Analyzer

## Project Information

**Project Title:** Mental Health Sentiment Analyzer

**Student:** Marco Landeo  
**Email:** mlandeo@stevens.edu  
**Status:** Working alone

## Problem Description

Mental health awareness has become increasingly important in our digital age, where people frequently express their emotions through online platforms like social media. This project addresses the challenge of automatically analyzing large collections of text data to detect sentiment trends related to mental health.

The program processes text data from CSV files, performs sentiment analysis to classify posts as positive, neutral, or negative, and generates visualizations and reports. While not intended for clinical diagnosis, the system can help identify emotional patterns and trends that may warrant attention.

## Solution Approach

The project uses a rule-based sentiment analysis approach combined with data processing and visualization:

1. **Data Loading**: Reads text data from CSV files using pandas
2. **Preprocessing**: Cleans and normalizes text data
3. **Sentiment Analysis**: Classifies text into positive, neutral, or negative categories
4. **Visualization**: Creates bar charts and pie charts using matplotlib
5. **Reporting**: Generates JSON reports with summary statistics

## Program Structure

```
.
├── main.ipynb                    # Main Jupyter notebook (entry point)
├── text_dataset.py               # TextDataset class module
├── sentiment_analyzer.py         # SentimentAnalyzer class module
├── utils.py                      # Utility functions (calculate_accuracy, generate_report)
├── test_sentiment_analyzer.py   # Pytest test file
├── sample_data.csv               # Sample dataset for testing
├── README.md                     # This file
└── requirements.txt              # Python dependencies
```

### Class Structure

- **TextDataset**: Handles reading, cleaning, and preprocessing text data from CSV files
  - Attributes: `file_path`, `data`, `raw_texts`, `cleaned_texts`
  - Methods: `clean_text()`, `preprocess()`, `filter_tokens()`, `batch_generator()`, `get_word_frequencies()`
  - Special methods: `__add__()` (operator overloading), `__str__()`

- **SentimentAnalyzer**: Performs sentiment analysis on text datasets
  - Uses composition: contains a `TextDataset` instance
  - Attributes: `dataset`, `results`, `sentiment_counts`
  - Methods: `analyze_sentiment()`, `analyze_all()`, `visualize_results()`, `get_summary_statistics()`
  - Special methods: `__str__()`

### Function Structure

- **calculate_accuracy()**: Calculates accuracy of sentiment predictions
- **generate_report()**: Generates and saves sentiment analysis reports to JSON files

## How to Use the Program

### Prerequisites

- Python 3.12 or 3.13
- Required packages (install using pip):
  - pandas
  - matplotlib
  - pytest (for testing)

### Installation

1. Clone or download this repository
2. Install required dependencies:
```bash
pip install pandas matplotlib pytest
```

Or use the requirements file:
```bash
pip install -r requirements.txt
```

### Running the Program

1. **Open the Jupyter Notebook:**
   ```bash
   jupyter notebook main.ipynb
   ```

2. **Run all cells** in order, or run them individually to see step-by-step execution

3. **Expected Outputs:**
   - Console output showing sentiment analysis results
   - `sentiment_distribution.png` - Visualization file
   - `sentiment_report.json` - JSON report file

### Running Tests

To run the pytest tests:
```bash
pytest test_sentiment_analyzer.py -v
```

This will verify:
- Exception handling (FileNotFoundError, ValueError)
- Class initialization and composition
- Sentiment analysis logic
- Accuracy calculation
- Operator overloading
- Generator functions

### Using Your Own Data

1. Create a CSV file with a `text` column (or one of: `Text`, `TEXT`, `content`, `Content`, `message`, `Message`)
2. Place the file in the project directory
3. Update the file path in `main.ipynb`:
   ```python
   dataset = TextDataset('your_file.csv')
   ```

## Main Contributions

**Marco Landeo:**
- Implemented TextDataset and SentimentAnalyzer classes
- Created utility functions for accuracy calculation and reporting
- Developed the Jupyter notebook workflow
- Added pytest tests for validation

## Expected Outputs

When running the notebook, you should see:
1. Dataset loading confirmation
2. Preprocessing status
3. Sentiment analysis results with counts
4. Visualization saved as PNG file
5. JSON report with summary statistics
6. Accuracy calculation (if applicable)
7. Interactive mode for testing custom sentences

## Notes

- The sentiment analysis uses a simple rule-based approach suitable for demonstration
- For production use, consider using more sophisticated NLP libraries (e.g., TextBlob, VADER, or transformer models)
- The sample dataset (`sample_data.csv`) contains 10 example texts for testing
- All code follows PEP 8 style guidelines with clear variable names and documentation

## License

This project is created for educational purposes as part of a course assignment.

