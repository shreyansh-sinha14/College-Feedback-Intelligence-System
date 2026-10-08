## 🌐 Live Demo

The College Feedback Intelligence System is deployed using Streamlit Community Cloud.

🔗 **Live Application:** https://college-feedback-intelligence.streamlit.app

# 🎓 College Feedback Intelligence System

## Student Information

| Field | Details |
|---|---|
| **Name** | Shreyansh Sinha |
| **Registration Number** | 23FE10CDS00513 |
| **Branch** | B.Tech CSE - Data Science |
| **Batch** | Batch G |
| **GitHub Username** | shreyansh-sinha14 |
| **Training Program** | MUJ Data Science Training Program |

---

# 📌 Project Title

## College Feedback Intelligence System

### NLP-Based Student Feedback Analysis

---

# 📖 Project Overview

The College Feedback Intelligence System is an NLP and Machine Learning based application that automatically analyzes student feedback.

The system accepts a CSV file containing student feedback and performs:

- Sentiment classification
- College aspect detection
- Data analysis
- Interactive visualizations
- Automatic insights
- Downloadable feedback report

---

# 🎯 Objectives

1. Collect and prepare student feedback data.
2. Perform text preprocessing.
3. Analyze the sentiment of feedback.
4. Identify major college-related aspects.
5. Convert text into numerical features using TF-IDF.
6. Train a Machine Learning model for sentiment classification.
7. Evaluate the model.
8. Generate useful visual insights from student feedback.

---

# 🧠 NLP Pipeline

```text
Student Feedback CSV
        ↓
Data Loading
        ↓
Text Cleaning
        ↓
Batch Feedback
        ↓
Gemini Flash API
        ↓
Prompt + Feedback
        ↓
Sentiment Analysis
        ↓
Aspect Detection
        ↓
Structured JSON Output
        ↓
Data Aggregation
        ↓
Visualizations & Insights
        ↓
Downloadable Report
```
---

# 📖 How the System Works

The College Feedback Intelligence System follows these steps:

1. User uploads a student feedback CSV file.
2. The system identifies the feedback column.
3. Text preprocessing is performed using tokenization and stopword removal.
4. TF-IDF converts the feedback text into numerical features.
5. Logistic Regression predicts the sentiment as Positive, Negative, or Neutral.
6. Keyword-based aspect detection identifies the college area discussed.
7. The system generates interactive visualizations.
8. Key insights are generated from the analyzed feedback.
9. The analyzed feedback can be downloaded as a CSV report.

---

# 📊 Output

The system provides:

- Overall sentiment distribution
- Sentiment analysis by college aspect
- Aspect vs sentiment heatmap
- Positive feedback analysis
- Negative feedback analysis
- Analyzed feedback table
- Downloadable feedback report

---

## 🧪 Model Performance

The sentiment classification model was evaluated using a test set of 120 feedback responses.

| Metric | Score |
|---|---:|
| Accuracy | 96.67% |
| Macro F1-score | 0.97 |
| Weighted F1-score | 0.97 |

### Classification Performance

| Sentiment | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Negative | 1.00 | 0.89 | 0.94 |
| Neutral | 1.00 | 1.00 | 1.00 |
| Positive | 0.94 | 1.00 | 0.97 |

The model achieved 96.67% accuracy on the test set, showing strong performance in classifying student feedback into Positive, Negative, and Neutral categories.

# 🧪 Model Evaluation

The sentiment classification model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

---

# 🚀 Future Enhancements

- Use Transformer-based models for improved sentiment classification
- Improve aspect detection using semantic similarity
- Support multilingual and Hinglish feedback
- Add department and semester-wise analysis
- Add historical feedback trend analysis
- Develop an administrative dashboard

## ⚙️ Installation Guide

### 1. Clone the Repository

```bash
git clone https://github.com/shreyansh-sinha14/MUJ-DS-23FE10CDS00513.git
```
### 2. Navigate to the Project
cd MUJ-DS-23FE10CDS00513/code

### 3. Install Required Dependencies
pip install -r requirements.txt

### 4. Run the Streamlit Application
streamlit run app.py

### 5. Open the Application
After running the command, Streamlit will provide a local URL.
Open the URL in a web browser and upload a student feedback CSV file.
📄 Input File Format
The application accepts CSV files containing a student feedback column.
The feedback column can be named:
- feedback
- review
- comment
- comments
- response
- student_feedback
- student response
Example:
feedback
The faculty explains concepts very well
The hostel rooms need improvement
The WiFi connection is excellent
The library has a good collection of books


📊 Results
The system generates:
- Overall sentiment distribution
- College aspect analysis
- Sentiment distribution by aspect
- Aspect vs sentiment heatmap
- Positive feedback analysis
- Negative feedback analysis
- Key insights
- Analyzed feedback table
- Downloadable CSV report
