# Revio AI – Intelligent Review Analysis and Customer Resolution Assistant

## 🔑 Demo Access

Try the live demonstration using the following access links and demo credentials:

### 👤 Employee Dashboard
- **URL:** https://revio-ai-review-assistant.streamlit.app/

*Note: These are demo credentials for exploring review intelligence, real-time analytics, urgent escalation handling, and the internal AI response approval workflow.*

### 🌐 Public Customer Review Form
- **URL:** https://revio-ai-review-assistant.streamlit.app/?page=submit_review&biz=demo-corp
- **Access:** No login required (embeddable public customer review submission widget).

---

## Project Overview

Revio AI is a proof-of-concept customer review intelligence system designed to help businesses understand customer feedback and respond more effectively.

The application processes customer reviews and performs four main tasks:

1. **Sentiment Analysis:** Classifies reviews as Positive, Negative, or Neutral using TF-IDF and Logistic Regression.
2. **Topic Classification:** Identifies issues such as Delivery, Pricing, Product Quality, and Customer Service using keyword-based classification.
3. **Priority Detection:** Identifies urgent complaints using predefined keywords and sentiment rules.
4. **Response Generation:** Creates context-aware suggested responses using rule-based templates.

The system provides an interactive Streamlit dashboard for reviewing customer feedback and exploring analysis results.

## Technologies Used

- **Programming Language:** Python
- **Frontend / Dashboard:** Streamlit
- **Data Processing:** Pandas
- **Machine Learning:** Scikit-learn
- **Text Vectorization:** TF-IDF
- **Classification Algorithm:** Logistic Regression
- **Data Visualization:** Plotly
- **Development Environment:** VS Code

## System Workflow

1. **Input:** Customer reviews are loaded from a sample CSV file, uploaded CSV, or individual text input.
2. **Preprocessing:** Review text is processed by the sentiment analysis pipeline.
3. **Sentiment Analysis:** TF-IDF converts text into numerical features, and Logistic Regression predicts sentiment.
4. **Topic Classification:** Keyword matching identifies the customer issue category.
5. **Priority Detection:** Rules identify urgent complaints and assign priority levels.
6. **Response Generation:** Rule-based templates generate suggested customer responses.
7. **Dashboard:** Streamlit displays review analytics, sentiment distribution, topic trends, priority levels, and suggested responses.

## Model Evaluation

The sentiment classifier was evaluated using 30 manually labelled test reviews.

| Metric | Result |
|---|---:|
| Accuracy | 83.3% |
| Macro Precision | 86.0% |
| Macro Recall | 83.3% |
| Macro F1-Score | 83.3% |

The evaluation dataset contains 10 positive, 10 neutral, and 10 negative reviews.

**Note:** These are preliminary results from a small manually curated dataset. Further evaluation using a larger, independent real-world dataset is required before production use.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/kiransree2002/revio-ai-review-assistant.git
cd revio-ai-review-assistant
