# ============================================================
# COLLEGE FEEDBACK INTELLIGENCE SYSTEM
# ============================================================

import re
import joblib
import nltk
import pandas as pd
import gradio as gr
import plotly.express as px

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# ============================================================
# 1. DOWNLOAD NLTK RESOURCES
# ============================================================

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)


# ============================================================
# 2. LOAD TRAINED MODEL AND TF-IDF VECTORIZER
# ============================================================

model = joblib.load("sentiment_model.pkl")

tfidf = joblib.load(
    "tfidf_vectorizer.pkl"
)

print("Model loaded successfully!")
print("TF-IDF vectorizer loaded successfully!")


# ============================================================
# 3. TEXT PREPROCESSING
# ============================================================

stop_words = set(
    stopwords.words("english")
)


def preprocess_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation and numbers
    text = re.sub(
        r"[^a-zA-Z\s]",
        "",
        text
    )

    # Tokenization
    tokens = word_tokenize(text)

    # Remove stopwords
    tokens = [
        word
        for word in tokens
        if word not in stop_words
    ]

    # Join tokens
    return " ".join(tokens)


# ============================================================
# 4. ASPECT KEYWORDS
# ============================================================

aspect_keywords = {

    "Faculty": [
        "faculty",
        "professor",
        "teacher"
    ],

    "Teaching": [
        "teaching",
        "lecture",
        "explain",
        "concept"
    ],

    "Laboratory": [
        "lab",
        "laboratory",
        "computer"
    ],

    "WiFi": [
        "wifi",
        "internet",
        "network"
    ],

    "Library": [
        "library",
        "books"
    ],

    "Hostel": [
        "hostel",
        "room",
        "warden"
    ],

    "Canteen": [
        "canteen",
        "food",
        "cafeteria"
    ],

    "Placement": [
        "placement",
        "job",
        "company",
        "recruitment"
    ],

    "Infrastructure": [
        "building",
        "classroom",
        "infrastructure"
    ],

    "Course": [
        "course",
        "syllabus",
        "subject"
    ],

    "Examination": [
        "exam",
        "examination",
        "test",
        "question"
    ]
}


# ============================================================
# 5. ASPECT DETECTION
# ============================================================

def detect_aspect(feedback):

    text = feedback.lower()

    for aspect, keywords in aspect_keywords.items():

        for keyword in keywords:

            if keyword in text:
                return aspect

    return "General"


# ============================================================
# 6. FIND FEEDBACK COLUMN
# ============================================================

def find_feedback_column(df):

    possible_columns = [

        "feedback",
        "Feedback",
        "review",
        "Review",
        "comment",
        "Comment",
        "comments",
        "Comments"
    ]

    # Check common column names
    for column in possible_columns:

        if column in df.columns:
            return column

    # If no common name exists,
    # find the first text column
    for column in df.columns:

        if df[column].dtype == "object":
            return column

    return None


# ============================================================
# 7. ANALYZE FILE
# ============================================================

def analyze_file(file):

    # No file uploaded
    if file is None:

        return (
            "## ⚠️ Please upload a CSV feedback file.",
            None,
            None,
            None,
            None,
            None,
            None,
            None
        )

    try:

        # ----------------------------------------------------
        # READ CSV
        # ----------------------------------------------------

        df = pd.read_csv(file)

        # ----------------------------------------------------
        # FIND FEEDBACK COLUMN
        # ----------------------------------------------------

        feedback_column = find_feedback_column(df)

        if feedback_column is None:

            return (
                "## ❌ Feedback column not found\n\n"
                f"Available columns: {list(df.columns)}",
                None,
                None,
                None,
                None,
                None,
                None,
                None
            )

        # ----------------------------------------------------
        # REMOVE EMPTY FEEDBACK
        # ----------------------------------------------------

        df = df.dropna(
            subset=[feedback_column]
        ).copy()

        df[feedback_column] = (
            df[feedback_column]
            .astype(str)
        )

        # ----------------------------------------------------
        # PREPROCESS TEXT
        # ----------------------------------------------------

        df["clean_feedback"] = (
            df[feedback_column]
            .apply(preprocess_text)
        )

        # ----------------------------------------------------
        # TF-IDF
        # ----------------------------------------------------

        vectors = tfidf.transform(
            df["clean_feedback"]
        )

        # ----------------------------------------------------
        # SENTIMENT PREDICTION
        # ----------------------------------------------------

        df["Sentiment"] = model.predict(
            vectors
        )

        # ----------------------------------------------------
        # ASPECT DETECTION
        # ----------------------------------------------------

        df["Aspect"] = (
            df[feedback_column]
            .apply(detect_aspect)
        )

        # ====================================================
        # BASIC STATISTICS
        # ====================================================

        total = len(df)

        positive = (
            df["Sentiment"] == "Positive"
        ).sum()

        negative = (
            df["Sentiment"] == "Negative"
        ).sum()

        neutral = (
            df["Sentiment"] == "Neutral"
        ).sum()

        positive_pct = round(
            positive / total * 100,
            1
        )

        negative_pct = round(
            negative / total * 100,
            1
        )

        neutral_pct = round(
            neutral / total * 100,
            1
        )

        # ====================================================
        # SENTIMENT DATA
        # ====================================================

        sentiment_data = (
            df["Sentiment"]
            .value_counts()
            .reindex(
                [
                    "Positive",
                    "Negative",
                    "Neutral"
                ],
                fill_value=0
            )
            .reset_index()
        )

        sentiment_data.columns = [
            "Sentiment",
            "Count"
        ]

        # ====================================================
        # CHART 1 - SENTIMENT PIE CHART
        # ====================================================

        sentiment_fig = px.pie(
            sentiment_data,
            names="Sentiment",
            values="Count",
            hole=0.45,
            title="Overall Sentiment Distribution"
        )

        # ====================================================
        # CHART 2 - SENTIMENT BY ASPECT
        # ====================================================

        aspect_sentiment = pd.crosstab(
            df["Aspect"],
            df["Sentiment"]
        )

        for col in [
            "Positive",
            "Negative",
            "Neutral"
        ]:

            if col not in aspect_sentiment.columns:
                aspect_sentiment[col] = 0

        aspect_sentiment = (
            aspect_sentiment
            .reset_index()
        )

        aspect_fig = px.bar(
            aspect_sentiment,
            x="Aspect",
            y=[
                "Positive",
                "Negative",
                "Neutral"
            ],
            barmode="group",
            title="Sentiment Distribution Across College Aspects"
        )

        aspect_fig.update_layout(
            xaxis_title="College Aspect",
            yaxis_title="Number of Reviews",
            xaxis_tickangle=-45
        )

        # ====================================================
        # CHART 3 - HEATMAP
        # ====================================================

        heatmap_data = pd.crosstab(
            df["Aspect"],
            df["Sentiment"]
        )

        heatmap_data = heatmap_data.reindex(
            columns=[
                "Positive",
                "Negative",
                "Neutral"
            ],
            fill_value=0
        )

        heatmap_fig = px.imshow(
            heatmap_data,
            text_auto=True,
            aspect="auto",
            title="Aspect vs Sentiment Heatmap"
        )

        heatmap_fig.update_layout(
            xaxis_title="Sentiment",
            yaxis_title="College Aspect"
        )

        # ====================================================
        # CHART 4 - NEGATIVE FEEDBACK
        # ====================================================

        negative_aspects = (
            df[
                df["Sentiment"] == "Negative"
            ]["Aspect"]
            .value_counts()
            .reset_index()
        )

        negative_aspects.columns = [
            "Aspect",
            "Negative Reviews"
        ]

        if len(negative_aspects) > 0:

            negative_fig = px.bar(
                negative_aspects,
                x="Negative Reviews",
                y="Aspect",
                orientation="h",
                title="🔴 Areas Receiving Negative Feedback"
            )

        else:

            negative_fig = px.bar(
                title="No Negative Feedback Found"
            )

        # ====================================================
        # CHART 5 - POSITIVE FEEDBACK
        # ====================================================

        positive_aspects = (
            df[
                df["Sentiment"] == "Positive"
            ]["Aspect"]
            .value_counts()
            .reset_index()
        )

        positive_aspects.columns = [
            "Aspect",
            "Positive Reviews"
        ]

        if len(positive_aspects) > 0:

            positive_fig = px.bar(
                positive_aspects,
                x="Positive Reviews",
                y="Aspect",
                orientation="h",
                title="🟢 Areas Receiving Positive Feedback"
            )

        else:

            positive_fig = px.bar(
                title="No Positive Feedback Found"
            )

        # ====================================================
        # AUTOMATIC INSIGHTS
        # ====================================================

        most_discussed = (
            df["Aspect"]
            .value_counts()
            .idxmax()
        )

        # Highest negative aspect

        if len(negative_aspects) > 0:

            highest_negative = (
                negative_aspects.iloc[0]["Aspect"]
            )

            highest_negative_count = int(
                negative_aspects.iloc[0][
                    "Negative Reviews"
                ]
            )

        else:

            highest_negative = "None"
            highest_negative_count = 0

        # Highest positive aspect

        if len(positive_aspects) > 0:

            highest_positive = (
                positive_aspects.iloc[0]["Aspect"]
            )

            highest_positive_count = int(
                positive_aspects.iloc[0][
                    "Positive Reviews"
                ]
            )

        else:

            highest_positive = "None"
            highest_positive_count = 0

        # ====================================================
        # INSIGHTS
        # ====================================================

        insights = f"""
# 🎓 College Feedback Intelligence Report

## 📊 Overall Results

| Metric | Result |
|---|---:|
| **Total Reviews** | **{total}** |
| 🟢 Positive | **{positive_pct}%** |
| 🔴 Negative | **{negative_pct}%** |
| 🟡 Neutral | **{neutral_pct}%** |

---

## 🔎 Key Insights

### 📌 Most Discussed Area

**{most_discussed}**

### 🔴 Area Receiving Most Negative Feedback

**{highest_negative}** — {highest_negative_count} reviews

### 🟢 Area Receiving Most Positive Feedback

**{highest_positive}** — {highest_positive_count} reviews

---

### 🤖 NLP Pipeline

**Text Preprocessing → TF-IDF → Logistic Regression → Sentiment Prediction → Aspect Detection**
"""

        # ====================================================
        # CREATE DOWNLOADABLE REPORT
        # ====================================================

        report_df = df.drop(
            columns=["clean_feedback"]
        )

        report_path = "analyzed_feedback_report.csv"

        report_df.to_csv(
            report_path,
            index=False
        )

        # ====================================================
        # TABLE
        # ====================================================

        display_df = report_df[
            [
                feedback_column,
                "Sentiment",
                "Aspect"
            ]
        ]

        # ====================================================
        # RETURN RESULTS
        # ====================================================

        return (
            insights,
            sentiment_fig,
            aspect_fig,
            heatmap_fig,
            negative_fig,
            positive_fig,
            display_df,
            report_path
        )

    except Exception as e:

        return (
            f"""
## ❌ Error while analyzing the file

**Error:** `{str(e)}`

Please check your CSV file and try again.
""",
            None,
            None,
            None,
            None,
            None,
            None,
            None
        )


# ============================================================
# 8. GRADIO FRONTEND
# ============================================================

custom_css = """
#main-title {
    text-align: center;
    font-size: 36px;
    font-weight: bold;
}

#subtitle {
    text-align: center;
    font-size: 18px;
    margin-bottom: 20px;
}

.gradio-container {
    max-width: 1400px !important;
}
"""


with gr.Blocks(
    theme=gr.themes.Soft(),
    css=custom_css,
    title="College Feedback Intelligence"
) as app:

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    gr.Markdown(
        """
        <div id="main-title">
        🎓 College Feedback Intelligence System
        </div>

        <div id="subtitle">
        NLP-Based Student Feedback Analysis Dashboard
        </div>
        """
    )

    gr.Markdown(
        """
        ### 📁 Upload Student Feedback

        Upload a **CSV file containing student feedback**.
        The system will automatically analyze sentiment,
        identify college aspects, generate visual insights,
        and create a downloadable report.
        """
    )

    # --------------------------------------------------------
    # UPLOAD
    # --------------------------------------------------------

    with gr.Row():

        file_input = gr.File(
            label="📁 Upload Feedback CSV",
            file_types=[".csv"],
            type="filepath"
        )

        analyze_button = gr.Button(
            "🔍 Analyze Feedback",
            variant="primary"
        )

    # --------------------------------------------------------
    # INSIGHTS
    # --------------------------------------------------------

    gr.Markdown("---")

    insights_output = gr.Markdown()

    # --------------------------------------------------------
    # CHARTS
    # --------------------------------------------------------

    gr.Markdown(
        "## 📊 Visual Analytics"
    )

    with gr.Row():

        sentiment_output = gr.Plot(
            label="Overall Sentiment"
        )

        aspect_output = gr.Plot(
            label="Sentiment by Aspect"
        )

    with gr.Row():

        heatmap_output = gr.Plot(
            label="Aspect vs Sentiment"
        )

    with gr.Row():

        negative_output = gr.Plot(
            label="Negative Feedback"
        )

        positive_output = gr.Plot(
            label="Positive Feedback"
        )

    # --------------------------------------------------------
    # TABLE
    # --------------------------------------------------------

    gr.Markdown("---")

    gr.Markdown(
        "## 📋 Detailed Feedback Analysis"
    )

    table_output = gr.Dataframe(
        label="Analyzed Student Feedback",
        interactive=False
    )

    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

    gr.Markdown("---")

    gr.Markdown(
        "## ⬇️ Download Report"
    )

    download_output = gr.File(
        label="Download Analyzed Feedback Report"
    )

    # --------------------------------------------------------
    # BUTTON
    # --------------------------------------------------------

    analyze_button.click(
        fn=analyze_file,
        inputs=file_input,
        outputs=[
            insights_output,
            sentiment_output,
            aspect_output,
            heatmap_output,
            negative_output,
            positive_output,
            table_output,
            download_output
        ]
    )


# ============================================================
# 9. LAUNCH APPLICATION
# ============================================================

app.launch()
