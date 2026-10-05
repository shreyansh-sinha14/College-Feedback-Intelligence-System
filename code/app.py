import streamlit as st
import pandas as pd
import re
import joblib
import nltk
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="College Feedback Intelligence System",
    page_icon="🎓",
    layout="wide"
)


# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "sentiment_model.pkl"
TFIDF_PATH = BASE_DIR / "tfidf_vectorizer.pkl"


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = joblib.load(MODEL_PATH)
    tfidf = joblib.load(TFIDF_PATH)

    return model, tfidf


model, tfidf = load_model()


# ============================================================
# NLTK RESOURCES
# ============================================================

@st.cache_resource
def download_nltk_resources():

    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("stopwords", quiet=True)


download_nltk_resources()

stop_words = set(stopwords.words("english"))


# ============================================================
# TEXT PREPROCESSING
# ============================================================

def preprocess_text(text):

    text = str(text).lower()

    text = re.sub(
        r"[^a-zA-Z\s]",
        "",
        text
    )

    tokens = word_tokenize(text)

    tokens = [
        word
        for word in tokens
        if word not in stop_words
    ]

    return " ".join(tokens)


# ============================================================
# ASPECT DETECTION
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


def detect_aspect(feedback):

    text = str(feedback).lower()

    for aspect, keywords in aspect_keywords.items():

        for keyword in keywords:

            if keyword in text:
                return aspect

    return "General"


# ============================================================
# FIND FEEDBACK COLUMN
# ============================================================

def find_feedback_column(df):

    possible_columns = [
        "feedback",
        "review",
        "comment",
        "comments",
        "response",
        "student_feedback",
        "student response"
    ]

    lower_columns = {
        column.lower(): column
        for column in df.columns
    }

    for column in possible_columns:

        if column in lower_columns:
            return lower_columns[column]

    return None


# ============================================================
# PAGE HEADER
# ============================================================

st.title("🎓 College Feedback Intelligence System")

st.subheader(
    "NLP-Based Student Feedback Analysis"
)

st.write(
    """
    Upload a student feedback CSV file to automatically analyze
    sentiment, identify college-related aspects, visualize the results,
    and download the analyzed feedback report.
    """
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📌 About the Project")

    st.write(
        """
        This system uses Natural Language Processing and Machine
        Learning to analyze student feedback.
        """
    )

    st.write("### Technologies")

    st.write(
        """
        • Python  
        • Pandas  
        • NLTK  
        • Scikit-learn  
        • TF-IDF  
        • Logistic Regression  
        • Plotly  
        • Streamlit
        """
    )

    st.write("### Analysis")

    st.write(
        """
        • Sentiment Analysis  
        • Aspect Detection  
        • Data Visualization  
        • Feedback Insights  
        • Downloadable Report
        """
    )


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "📂 Upload Student Feedback CSV",
    type=["csv"]
)


# ============================================================
# PROCESS FILE
# ============================================================

if uploaded_file is not None:

    try:

        df = pd.read_csv(uploaded_file)

        st.success(
            "CSV file uploaded successfully!"
        )

        st.write(
            f"**Rows:** {df.shape[0]}  |  "
            f"**Columns:** {df.shape[1]}"
        )

        # ----------------------------------------------------
        # FIND FEEDBACK COLUMN
        # ----------------------------------------------------

        feedback_column = find_feedback_column(df)

        if feedback_column is None:

            st.error(
                "No feedback column was found. "
                "Please use a column such as "
                "`feedback`, `review`, `comment`, or `response`."
            )

            st.stop()

        st.info(
            f"Feedback column detected: **{feedback_column}**"
        )

        # ----------------------------------------------------
        # REMOVE EMPTY FEEDBACK
        # ----------------------------------------------------

        df = df.dropna(
            subset=[feedback_column]
        ).copy()

        df[feedback_column] = df[
            feedback_column
        ].astype(str)

        # ----------------------------------------------------
        # PREPROCESS
        # ----------------------------------------------------

        with st.spinner(
            "Processing feedback..."
        ):

            df["clean_feedback"] = df[
                feedback_column
            ].apply(preprocess_text)

            # ------------------------------------------------
            # SENTIMENT PREDICTION
            # ------------------------------------------------

            X_new = tfidf.transform(
                df["clean_feedback"]
            )

            df["sentiment"] = model.predict(
                X_new
            )

            # ------------------------------------------------
            # ASPECT DETECTION
            # ------------------------------------------------

            df["aspect"] = df[
                feedback_column
            ].apply(detect_aspect)

        st.success(
            "Feedback analysis completed!"
        )


        # ====================================================
        # SUMMARY METRICS
        # ====================================================

        st.header("📊 Overall Summary")

        total_feedback = len(df)

        positive_count = (
            df["sentiment"] == "Positive"
        ).sum()

        negative_count = (
            df["sentiment"] == "Negative"
        ).sum()

        neutral_count = (
            df["sentiment"] == "Neutral"
        ).sum()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Feedback",
                total_feedback
            )

        with col2:
            st.metric(
                "Positive",
                positive_count
            )

        with col3:
            st.metric(
                "Negative",
                negative_count
            )

        with col4:
            st.metric(
                "Neutral",
                neutral_count
            )


        # ====================================================
        # SENTIMENT PIE CHART
        # ====================================================

        st.header("📈 Overall Sentiment")

        sentiment_counts = (
            df["sentiment"]
            .value_counts()
            .reset_index()
        )

        sentiment_counts.columns = [
            "sentiment",
            "count"
        ]

        fig_sentiment = px.pie(
            sentiment_counts,
            names="sentiment",
            values="count",
            title="Overall Sentiment Distribution",
            hole=0.4
        )

        st.plotly_chart(
            fig_sentiment,
            use_container_width=True
        )


        # ====================================================
        # ASPECT ANALYSIS
        # ====================================================

        st.header(
            "🏫 College Aspect Analysis"
        )

        aspect_counts = (
            df["aspect"]
            .value_counts()
            .reset_index()
        )

        aspect_counts.columns = [
            "aspect",
            "count"
        ]

        fig_aspects = px.bar(
            aspect_counts,
            x="aspect",
            y="count",
            title="Feedback by College Aspect",
            text="count"
        )

        fig_aspects.update_layout(
            xaxis_title="College Aspect",
            yaxis_title="Number of Feedback Responses"
        )

        st.plotly_chart(
            fig_aspects,
            use_container_width=True
        )


        # ====================================================
        # SENTIMENT BY ASPECT
        # ====================================================

        st.header(
            "📊 Sentiment Across College Aspects"
        )

        aspect_sentiment = pd.crosstab(
            df["aspect"],
            df["sentiment"]
        ).reset_index()

        for sentiment in [
            "Positive",
            "Negative",
            "Neutral"
        ]:

            if sentiment not in aspect_sentiment.columns:
                aspect_sentiment[sentiment] = 0

        fig_aspect_sentiment = px.bar(
            aspect_sentiment,
            x="aspect",
            y=[
                "Positive",
                "Negative",
                "Neutral"
            ],
            barmode="group",
            title="Sentiment Distribution by College Aspect"
        )

        fig_aspect_sentiment.update_layout(
            xaxis_title="College Aspect",
            yaxis_title="Number of Feedback Responses"
        )

        st.plotly_chart(
            fig_aspect_sentiment,
            use_container_width=True
        )


        # ====================================================
        # HEATMAP
        # ====================================================

        st.header(
            "🔥 Aspect vs Sentiment Heatmap"
        )

        heatmap_data = pd.crosstab(
            df["aspect"],
            df["sentiment"]
        )

        for sentiment in [
            "Positive",
            "Negative",
            "Neutral"
        ]:

            if sentiment not in heatmap_data.columns:
                heatmap_data[sentiment] = 0

        heatmap_data = heatmap_data[
            [
                "Positive",
                "Negative",
                "Neutral"
            ]
        ]

        fig_heatmap = go.Figure(
            data=go.Heatmap(
                z=heatmap_data.values,
                x=heatmap_data.columns,
                y=heatmap_data.index,
                text=heatmap_data.values,
                texttemplate="%{text}",
                colorscale="Blues"
            )
        )

        fig_heatmap.update_layout(
            title="Aspect vs Sentiment",
            xaxis_title="Sentiment",
            yaxis_title="College Aspect"
        )

        st.plotly_chart(
            fig_heatmap,
            use_container_width=True
        )


        # ====================================================
        # NEGATIVE FEEDBACK
        # ====================================================

        st.header(
            "🔴 Negative Feedback by Aspect"
        )

        negative_data = df[
            df["sentiment"] == "Negative"
        ]

        negative_counts = (
            negative_data["aspect"]
            .value_counts()
            .reset_index()
        )

        negative_counts.columns = [
            "aspect",
            "count"
        ]

        if len(negative_counts) > 0:

            fig_negative = px.bar(
                negative_counts,
                x="aspect",
                y="count",
                title="Negative Feedback by College Aspect",
                text="count"
            )

            st.plotly_chart(
                fig_negative,
                use_container_width=True
            )

        else:

            st.info(
                "No negative feedback was detected."
            )


        # ====================================================
        # POSITIVE FEEDBACK
        # ====================================================

        st.header(
            "🟢 Positive Feedback by Aspect"
        )

        positive_data = df[
            df["sentiment"] == "Positive"
        ]

        positive_counts = (
            positive_data["aspect"]
            .value_counts()
            .reset_index()
        )

        positive_counts.columns = [
            "aspect",
            "count"
        ]

        if len(positive_counts) > 0:

            fig_positive = px.bar(
                positive_counts,
                x="aspect",
                y="count",
                title="Positive Feedback by College Aspect",
                text="count"
            )

            st.plotly_chart(
                fig_positive,
                use_container_width=True
            )

        else:

            st.info(
                "No positive feedback was detected."
            )


        # ====================================================
        # AUTOMATIC INSIGHTS
        # ====================================================

        st.header("💡 Key Insights")

        most_discussed = (
            df["aspect"]
            .value_counts()
            .idxmax()
        )

        most_positive = (
            df[df["sentiment"] == "Positive"]["aspect"]
            .value_counts()
        )

        most_negative = (
            df[df["sentiment"] == "Negative"]["aspect"]
            .value_counts()
        )

        st.write(
            f"• **Most discussed aspect:** "
            f"{most_discussed}"
        )

        if len(most_positive) > 0:

            st.write(
                f"• **Most positively received aspect:** "
                f"{most_positive.idxmax()}"
            )

        if len(most_negative) > 0:

            st.write(
                f"• **Aspect with the most negative feedback:** "
                f"{most_negative.idxmax()}"
            )

        st.write(
            f"• **Overall positive feedback:** "
            f"{positive_count} responses"
        )

        st.write(
            f"• **Overall negative feedback:** "
            f"{negative_count} responses"
        )

        st.write(
            f"• **Overall neutral feedback:** "
            f"{neutral_count} responses"
        )


        # ====================================================
        # ANALYZED DATA
        # ====================================================

        st.header(
            "📋 Analyzed Feedback"
        )

        display_columns = [
            column
            for column in df.columns
            if column != "clean_feedback"
        ]

        st.dataframe(
            df[display_columns],
            use_container_width=True
        )


        # ====================================================
        # DOWNLOAD REPORT
        # ====================================================

        st.header(
            "📥 Download Analysis Report"
        )

        report_df = df.drop(
            columns=["clean_feedback"]
        )

        csv_data = report_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="⬇️ Download Analyzed Feedback CSV",
            data=csv_data,
            file_name="analyzed_feedback_report.csv",
            mime="text/csv"
        )


    except Exception as e:

        st.error(
            f"An error occurred while processing the file: {e}"
        )


else:

    st.info(
        "👆 Upload a CSV file above to start the analysis."
    )
