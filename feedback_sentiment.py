"""
EduTech Learner Feedback Sentiment and Theme Analysis
======================================================
Analyses course feedback using sentiment analysis (DistilBERT) and
zero-shot theme classification. Produces visualisations of rating
distributions, sentiment-theme breakdowns, and a cross-course
feedback pivot table.

Dataset: feedback.csv
"""

import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
warnings.filterwarnings("ignore")


SENTIMENT_MODEL  = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
THEME_MODEL      = "typeform/distilbert-base-uncased-mnli"
THEMES           = ["support", "practice", "explanations", "unknown"]


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """Load feedback CSV and print a preview."""
    df = pd.read_csv(filepath)
    print(f"Shape: {df.shape}")
    print(df.head())
    return df


# ── Exploratory Visualisations ────────────────────────────────────────────────

def plot_rating_distribution(df: pd.DataFrame):
    """Histogram of overall rating distribution."""
    plt.figure(figsize=(8, 5))
    plt.hist(df["Rating"], bins=5, color="skyblue", edgecolor="black")
    plt.title("Distribution of Course Ratings")
    plt.xlabel("Rating")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("rating_distribution.png", dpi=150)
    plt.show()
    print("Saved: rating_distribution.png")


def plot_comments_per_course(df: pd.DataFrame):
    """Bar chart of the number of comments per course."""
    course_counts = df["Course Name"].value_counts()
    plt.figure(figsize=(10, 6))
    course_counts.plot(kind="bar", color="teal")
    plt.title("Number of Comments per Course")
    plt.xlabel("Course Name")
    plt.ylabel("Number of Comments")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("comments_per_course.png", dpi=150)
    plt.show()
    print("Saved: comments_per_course.png")


def plot_ratings_by_course(df: pd.DataFrame):
    """FacetGrid of rating histograms broken down by course."""
    g = sns.FacetGrid(df, col="Course Name", col_wrap=3, height=4,
                      sharex=False, sharey=False)
    g.map(plt.hist, "Rating", bins=5, color="slateblue", edgecolor="white")
    g.set_titles("{col_name}")
    g.set_xlabels("Rating")
    g.set_ylabels("Count")
    plt.suptitle("Rating Distribution by Course", y=1.02)
    plt.tight_layout()
    plt.savefig("ratings_by_course.png", dpi=150)
    plt.show()
    print("Saved: ratings_by_course.png")


# ── NLP Pipelines ─────────────────────────────────────────────────────────────

def load_pipelines():
    """Load sentiment analysis and zero-shot classification pipelines."""
    sentiment_pipe = pipeline("sentiment-analysis", model=SENTIMENT_MODEL)
    theme_pipe     = pipeline("zero-shot-classification", model=THEME_MODEL)
    return sentiment_pipe, theme_pipe


def safe_sentiment(text: str, sentiment_pipe) -> str:
    """Return sentiment label for valid strings, 'Unknown' otherwise."""
    if isinstance(text, str) and text.strip():
        return sentiment_pipe(text)[0]["label"]
    return "Unknown"


def extract_theme(text: str, theme_pipe) -> str:
    """Return the top theme label for valid strings, 'Unknown' otherwise."""
    if isinstance(text, str) and text.strip():
        result = theme_pipe(text, candidate_labels=THEMES)
        return result["labels"][0]
    return "Unknown"


def apply_nlp(df: pd.DataFrame, sentiment_pipe, theme_pipe) -> pd.DataFrame:
    """Apply sentiment analysis and theme extraction to the Comment column."""
    df["sentiment"] = df["Comment"].apply(lambda x: safe_sentiment(x, sentiment_pipe))
    df["theme"]     = df["Comment"].apply(lambda x: extract_theme(x, theme_pipe))
    return df


# ── Analysis and Visualisation ────────────────────────────────────────────────

def plot_sentiment_by_theme(df: pd.DataFrame, top_n: int = 5):
    """Stacked bar chart of sentiment distribution for the top N themes."""
    theme_sentiment = (
        df.groupby(["theme", "sentiment"])
        .size()
        .unstack(fill_value=0)
    )
    top_themes = theme_sentiment.sum(axis=1).nlargest(top_n).index
    theme_sentiment.loc[top_themes].plot(kind="bar", stacked=True, figsize=(10, 6))
    plt.title(f"Sentiment Distribution for Top {top_n} Themes")
    plt.ylabel("Number of Feedback Entries")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("sentiment_by_theme.png", dpi=150)
    plt.show()
    print("Saved: sentiment_by_theme.png")


def build_course_feedback_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """Create a pivot table: Course x (sentiment, theme) with comment counts."""
    grouped = (
        df.groupby(["Course Name", "sentiment", "theme"])
        .size()
        .reset_index(name="counts")
    )
    pivot = grouped.pivot_table(
        index="Course Name",
        columns=["sentiment", "theme"],
        values="counts",
        fill_value=0,
    )
    print("\nCourse Feedback Pivot Table (top rows):")
    print(pivot.head().to_string())
    return pivot


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DATA_PATH = "feedback.csv"

    df = load_data(DATA_PATH)

    plot_rating_distribution(df)
    plot_comments_per_course(df)
    plot_ratings_by_course(df)

    sentiment_pipe, theme_pipe = load_pipelines()
    df = apply_nlp(df, sentiment_pipe, theme_pipe)

    print("\nSentiment distribution:")
    print(df["sentiment"].value_counts())
    print("\nTheme distribution:")
    print(df["theme"].value_counts())

    plot_sentiment_by_theme(df)

    pivot = build_course_feedback_pivot(df)
