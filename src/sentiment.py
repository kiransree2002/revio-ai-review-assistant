
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


# Small labelled training dataset for our prototype

training_reviews = [
    # Positive reviews
    ("Excellent product and outstanding quality", "positive"),
    ("Amazing experience and fast delivery", "positive"),
    ("Very happy with my purchase", "positive"),
    ("Customer support was friendly and helpful", "positive"),
    ("Fantastic product, highly recommended", "positive"),
    ("The product works perfectly", "positive"),
    ("Great value for money", "positive"),
    ("Received my order earlier than expected", "positive"),
    ("The packaging was excellent", "positive"),
    ("Quick response from the support team", "positive"),
    ("Very satisfied with the service", "positive"),
    ("Everything was perfect", "positive"),
    ("Excellent customer service", "positive"),
    ("Good quality and reasonable pricing", "positive"),
    ("I would definitely buy again", "positive"),

    # Neutral reviews
    ("The product is okay", "neutral"),
    ("Average experience overall", "neutral"),
    ("Delivery was acceptable", "neutral"),
    ("The product works as expected", "neutral"),
    ("Nothing special about the service", "neutral"),
    ("Packaging could be improved", "neutral"),
    ("The price seems reasonable", "neutral"),
    ("I received what I ordered", "neutral"),
    ("The experience was neither good nor bad", "neutral"),
    ("Product quality is average", "neutral"),
    ("The service was normal", "neutral"),
    ("It took the expected delivery time", "neutral"),
    ("The product meets basic requirements", "neutral"),
    ("No major issues with my purchase", "neutral"),
    ("The experience was satisfactory", "neutral"),

    # Negative reviews
    ("Terrible product and very poor quality", "negative"),
    ("My order arrived late", "negative"),
    ("The product arrived damaged", "negative"),
    ("Customer support never responded", "negative"),
    ("I want a refund immediately", "negative"),
    ("Very disappointed with this purchase", "negative"),
    ("The product stopped working after two days", "negative"),
    ("I was charged twice for the same order", "negative"),
    ("Delivery service was extremely poor", "negative"),
    ("Nobody helped me resolve my complaint", "negative"),
    ("The product is defective", "negative"),
    ("Very expensive and not worth the money", "negative"),
    ("My package has not been delivered", "negative"),
    ("I received the wrong product", "negative"),
    ("Extremely unhappy with the customer service", "negative"),


    # Additional negative examples
    ("I regret buying this product", "negative"),
    ("The service was disappointing", "negative"),
    ("My package never arrived", "negative"),
    ("The item was faulty when delivered", "negative"),
    ("I am unhappy with the purchase", "negative"),
    ("The company failed to solve my issue", "negative"),
    ("This product is not worth buying", "negative"),
    ("I had a horrible shopping experience", "negative"),
    ("The delivery was frustrating and unreliable", "negative"),
    ("I cannot recommend this product", "negative"),

    # Additional positive examples
    ("The entire shopping experience was wonderful", "positive"),
    ("Everything arrived in perfect condition", "positive"),
    ("The team resolved my issue quickly", "positive"),
    ("I am impressed with the product", "positive"),
    ("Excellent experience from start to finish", "positive"),

    # Additional neutral examples
    ("The order was processed normally", "neutral"),
    ("The product has standard features", "neutral"),
    ("I have no additional feedback", "neutral"),
    ("The item matches its basic description", "neutral"),
    ("The service was as described", "neutral"),

]



def train_sentiment_model():
    texts = [review for review, label in training_reviews]
    labels = [label for review, label in training_reviews]

    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    model.fit(texts, labels)
    return model


def analyze_sentiment(review, model=None):
    if model is None:
        model = train_sentiment_model()

    prediction = model.predict([review])[0]

    probabilities = model.predict_proba([review])[0]
    confidence = max(probabilities)

    return {
        "sentiment": prediction,
        "confidence": round(float(confidence), 3)
    }
