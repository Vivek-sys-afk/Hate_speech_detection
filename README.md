# Hate Speech Detection

This repository contains a machine learning pipeline for classifying social media text into hate speech, offensive language, or neutral categories. The system utilizes natural language processing (NLP) to clean and prepare text data, extracts features using a Bag of Words representation, and evaluates classifiers to identify the model for the task.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset Details](#dataset-details)
- [NLP Pipeline](#nlp-pipeline)
- [Feature Extraction](#feature-extraction)
- [Model Performance](#model-performance)
  - [Decision Tree Classifier](#decision-tree-classifier)
  - [Random Forest Classifier](#random-forest-classifier)
- [Environment Setup](#environment-setup)
- [How to Run](#how-to-run)

## Project Overview
Social media platforms generate vast quantities of user content, necessitating automated moderation tools. This project builds and compares supervised classification models to categorize tweets. 

Each text sample is classified into one of three classes:
- **Class 0**: Hate Speech
- **Class 1**: Offensive Language
- **Class 2**: Neither (Neutral)

## Dataset Details
The dataset used for this project is stored in `labeled_data.csv` and contains 24,783 rows. The key columns used are:
- `tweet`: The raw text of the social media post.
- `class`: The target variable containing the category labels (0, 1, or 2).

### Class Distribution
The dataset is imbalanced, with a distribution of labels as follows:
- **Class 1 (Offensive Language)**: 19,190 tweets
- **Class 2 (Neither)**: 4,163 tweets
- **Class 0 (Hate Speech)**: 1,430 tweets

## NLP Pipeline
The text cleaning and tokenization process consists of:
1. **Lowercasing**: Converting all text to lowercase.
2. **URL Removal**: Removing web links starting with HTTP/HTTPS or WWW.
3. **Punctuation & HTML Stripping**: Removing HTML tags, brackets, and general punctuation characters.
4. **Digit Removal**: Stripping words that contain numeric values.
5. **Stopwords Filtering**: Removing English stopwords using the NLTK corpus, along with the common retweet abbreviation "rt".
6. **Stemming**: Reducing words to their base or root forms using the NLTK `SnowballStemmer` for the English language.

## Feature Extraction
The processed text tokens are converted into a numerical format using a Bag of Words representation:
- **Method**: `CountVectorizer` from `scikit-learn`.
- **Outcome**: The text corpus is transformed into a compressed sparse row (CSR) matrix of shape `(24783, 27804)`.

## Model Performance
The models are evaluated using a 67/33 train-test split (16,604 training samples and 8,179 testing samples).

### Decision Tree Classifier
- **Accuracy**: 86.55%
- **Confusion Matrix**:
  ```
  [[ 163,  272,   30],   # Hate Speech (Class 0)
   [ 264, 5836,  235],   # Offensive Language (Class 1)
   [  42,  257, 1080]]   # Neither (Class 2)
  ```

### Random Forest Classifier
- **Accuracy**: 88.84%
- **Confusion Matrix**:
  ```
  [[ 121,  313,   31],   # Hate Speech (Class 0)
   [ 116, 6064,  155],   # Offensive Language (Class 1)
   [  12,  286, 1081]]   # Neither (Class 2)
  ```

### Key Observation
The Random Forest Classifier achieves a higher overall accuracy (88.84% compared to 86.55%). Both models face challenges classifying Class 0 (Hate Speech) due to the sample size of Class 0 in the training dataset.

## Environment Setup
To run the classification pipeline locally, set up a Python environment and install the required dependencies.

### Prerequisites
- Python 3.8 or higher
- `pip` package manager

### Installation Steps
1. Clone this repository to your local machine:
   ```bash
   git clone <repository-url>
   cd Hate_speech_detection
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:
   - **Windows (PowerShell)**:
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
   - **macOS/Linux**:
     ```bash
     source venv/bin/activate
     ```

4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run
After setting up the environment, you can run the Jupyter Notebook to train the models and visualize the classification results:

1. Launch the Jupyter Notebook server:
   ```bash
   jupyter notebook
   ```

2. Open the file `HateSpeechDetection.ipynb` from the Jupyter interface.

3. Run the cells in order to load the dataset, preprocess the text, train the models, and evaluate their performances.
