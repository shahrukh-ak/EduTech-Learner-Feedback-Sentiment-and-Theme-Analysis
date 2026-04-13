# EduTech Learner Feedback Sentiment and Theme Analysis

Analyses learner feedback using two NLP techniques: sentiment analysis via a fine-tuned DistilBERT model and zero-shot theme classification. Produces visualisations including rating distributions, a sentiment-by-theme breakdown, and a cross-course feedback pivot table.

## Business Context

As an online learning platform grows its course catalogue, manually reviewing student feedback at scale becomes impractical. This project automates the categorisation of feedback by sentiment and theme, surfacing which courses have negative feedback on specific dimensions (explanations, practice, support) so curriculum teams can prioritise improvements.

## Dataset

`feedback.csv` contains student feedback records with columns including `Course Name`, `Rating`, and `Comment`.

## Methodology

**EDA:**
- Histogram of overall rating distribution
- Bar chart of comment volume per course
- Per-course rating histograms via Seaborn FacetGrid

**Sentiment Analysis:** `distilbert-base-uncased-finetuned-sst-2-english` classifies each comment as POSITIVE or NEGATIVE. Non-string and empty inputs return "Unknown".

**Theme Classification:** `typeform/distilbert-base-uncased-mnli` performs zero-shot classification across four candidate themes: support, practice, explanations, unknown. No task-specific training data is required.

**Sentiment-Theme Analysis:** A stacked bar chart shows how positive and negative sentiment is distributed across the top 5 themes.

**Course Pivot Table:** A multi-level pivot of course × (sentiment, theme) with comment counts enables comparison across courses.

## Project Structure

```
13_edutech_feedback_sentiment/
├── feedback_sentiment.py  # Full analysis pipeline
├── requirements.txt
└── README.md
```

## Requirements

```
pandas
matplotlib
seaborn
transformers
torch
```

Install with:

```bash
pip install -r requirements.txt
```

## Usage

Place `feedback.csv` in the same directory and run:

```bash
python feedback_sentiment.py
```

Outputs: `rating_distribution.png`, `comments_per_course.png`, `ratings_by_course.png`, `sentiment_by_theme.png`, and a printed pivot table.

## Notes

Models are downloaded automatically from Hugging Face Hub on first run. For large feedback datasets, setting `device=0` in the pipeline calls enables GPU acceleration.
