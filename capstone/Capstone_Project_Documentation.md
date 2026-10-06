# College Feedback Intelligence System

## Project Overview

The College Feedback Intelligence System is an NLP-based application
designed to automatically analyze student feedback.

The system identifies the sentiment of feedback as Positive, Negative,
or Neutral and detects the college-related aspect being discussed.

## Objectives

- Analyze student feedback automatically
- Perform text preprocessing
- Classify feedback sentiment
- Identify college-related aspects
- Generate visual insights
- Provide a downloadable analyzed feedback report

## Technologies Used

- Python
- Pandas
- NumPy
- NLTK
- Scikit-learn
- TF-IDF
- Logistic Regression
- Plotly
- Streamlit

## NLP Pipeline

Student Feedback
→ Text Preprocessing
→ TF-IDF Vectorization
→ Logistic Regression
→ Sentiment Classification
→ Aspect Detection
→ Visualization
→ Insights
→ Downloadable Report

## Sentiment Analysis

The sentiment classification model uses TF-IDF features with Logistic
Regression.

The model was evaluated on a test set of 120 feedback responses.

### Model Performance

- Accuracy: 96.67%
- Macro F1-score: 0.97
- Weighted F1-score: 0.97

## Aspect Detection

The system identifies college-related aspects such as:

- Faculty
- Teaching
- Laboratory
- WiFi
- Library
- Hostel
- Canteen
- Placement
- Infrastructure
- Course
- Examination

The current aspect detection approach uses predefined keywords.

## Application Features

- CSV feedback upload
- Automatic feedback column detection
- Sentiment analysis
- College aspect detection
- Overall sentiment visualization
- Aspect analysis
- Sentiment-by-aspect analysis
- Aspect vs sentiment heatmap
- Positive and negative feedback analysis
- Automatic insights
- Analyzed feedback table
- Downloadable CSV report

## Deployment

The application is deployed using Streamlit Community Cloud.

Live Application:

https://college-feedback-intelligence.streamlit.app

## Future Enhancements

- Transformer-based sentiment models
- Improved semantic aspect detection
- Multilingual and Hinglish feedback support
- Department and semester-wise analysis
- Historical feedback trend analysis
- Administrative dashboard
