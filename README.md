# ML Enhanced Smart Fact Checker

A powerful fact-checking application that combines machine learning, knowledge base verification, and linguistic analysis to detect misinformation and fake news.

## Features

- **ML-Powered Detection System**: Utilizes a trained machine learning model to predict the veracity of claims
- **Knowledge Base Verification**: Checks claims against a comprehensive database of verified facts
- **Fake News Pattern Recognition**: Identifies common fake news indicators and sensationalized language
- **Linguistic Analysis**: Analyzes text for emotional manipulation and credibility markers
- **External Source Verification**: Cross-references claims with Wikipedia, news sources, and web search results
- **Interactive GUI**: User-friendly interface with visual results and evidence presentation

## Detection Workflow

1. **Fake News Indicator Detection** (First Priority)
   - Scans for sensationalized language patterns commonly used in fake news

2. **Trained Fake Pattern Matching** (Second Priority)
   - Compares claims against known false patterns in the knowledge base

3. **Machine Learning Model Prediction** (Third Priority)
   - Uses a trained Random Forest classifier to predict claim veracity

4. **Linguistic Analysis** (Fourth Priority)
   - Analyzes text for emotional manipulation and credibility markers

5. **Knowledge Base Verification** (Fifth Priority)
   - Checks claims against verified facts in the database

6. **External Source Verification** (Final Step)
   - Cross-references with Wikipedia, news sources, and web search

7. **Evidence-Based Results**
   - Presents findings with clear sources and confidence levels

## Installation

### Prerequisites
- Python 3.7 or higher
- PyQt5
- scikit-learn
- pandas
- numpy
- requests
- wikipedia (optional)
## Usage

1. Launch the application to see the welcome screen
2. Click "Start Analysis" to open the fact-checking interface
3. Enter a headline or claim in the text field
4. Click "Analyze" to begin the fact-checking process
5. Review the results, which include:
   - Overall verdict (VERIFIED, FALSE, or UNCERTAIN)
   - Confidence percentage
   - Evidence from various sources
   - Visual representation of the analysis

## Configuration
The application uses a configuration file (`fact_checker_config.json`) for API keys and settings. You can customize:
- API keys for external services (SerpAPI, NewsAPI)
- Request timeouts and delays
- Model and vectorizer file paths
- Maximum number of sources to check

## Knowledge Base
The application includes a comprehensive knowledge base with:
- Verified political leadership information
- CEO and company ownership data
- Capital cities and countries
- Common fake news patterns
- Verified facts for model training

## Machine Learning Model
The application uses a Random Forest classifier trained on:

- Verified facts from the knowledge base
- Known false claims and patterns
- Linguistic features from credible and non-credible sources

The model is automatically trained on first run and saved for future use.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.
## Acknowledgments
- PyQt5 for the GUI framework
- scikit-learn for machine learning capabilities
- Wikipedia API for fact verification
- NewsAPI and SerpAPI for external source verification
## Disclaimer
This tool is designed for educational and research purposes. While it strives for accuracy, it should not be used as the sole source for fact-checking important information. Always verify critical information through multiple reliable sources.
