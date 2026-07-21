# Phishing URL Detector (AI + ML)

This project is an AI-powered phishing URL detector using **Python + scikit-learn**.  
Dataset: Kaggle Phishing URL Dataset.

## Features
- Extracts 10+ phishing-related features from URLs.
- Trains Logistic Regression and Random Forest models.
- Saves the best model (`phishing_model.pkl`).
- Predicts new URLs with confidence scores.
- Interactive mode for testing your own URLs.

- ### Added comment for feature extraction
- ### Added Random Forest feature importance output
- ### Added code to save best model as phishing_model.pkl
- ### Added dataset.csv placeholder and Kaggle dataset instructions
- ### Implemented URL feature extraction functions (length, symbols, IP, HTTPS, etc.)

## Requirements
- Python 3.8+
- pip package manager

### Install dependencies
```bash
pip install pandas numpy scikit-learn

Run Instructions for Windows:

1: Clone the repository
git clone https://github.com/Aisha031/phishing-url-detector.git
cd phishing-url-detector

2:Place the dataset file (dataset.csv) in the same folder

3:Run the script:
python phishing_detector.py

Linux / macOS:

1:Clone the repository:
git clone https://github.com/Aisha031/phishing-url-detector.git
cd phishing-url-detector

2:Ensure dataset file (dataset.csv) is present.

3:Run the script:
python3 phishing_detector.py


#### Output

Model training results (Accuracy, Precision, Recall, F1).

Best model saved as phishing_model.pkl.

Sample predictions on test URLs.

Interactive mode for user input.
