from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from sentiment import train_sentiment_model

# Train the model
model = train_sentiment_model()

# Larger unseen test dataset
test_reviews = [
    # Positive
    "The product quality is excellent and I love it",
    "Amazing customer support and quick delivery",
    "Very satisfied with my purchase",
    "The product exceeded my expectations",
    "Great service and excellent value",
    "I received my package earlier than expected",
    "Everything works perfectly",
    "The support team resolved my issue quickly",
    "Wonderful experience, highly recommended",
    "I would happily purchase this again",

    # Neutral
    "The product is average",
    "Delivery took the expected amount of time",
    "The item works as described",
    "The packaging is standard",
    "The price is reasonable",
    "I received the correct product",
    "The service was acceptable",
    "The product has basic features",
    "Nothing unusual about my experience",
    "The order was processed normally",

    # Negative
    "The product quality is terrible",
    "My delivery was delayed by several days",
    "The customer support team ignored me",
    "I received a broken item",
    "I want my money back",
    "The product stopped working immediately",
    "This was a very disappointing experience",
    "I was charged twice for one purchase",
    "The company did not resolve my complaint",
    "I cannot recommend this service"
]

# Expected labels
actual_labels = (
    ["positive"] * 10
    + ["neutral"] * 10
    + ["negative"] * 10
)

# Predict sentiment
predicted_labels = model.predict(test_reviews)

# Evaluation
print("\nMODEL EVALUATION")
print("-------------------------")

print(
    "Accuracy:",
    round(accuracy_score(actual_labels, predicted_labels), 3)
)

print("\nClassification Report:")
print(
    classification_report(
        actual_labels,
        predicted_labels,
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        actual_labels,
        predicted_labels,
        labels=["positive", "neutral", "negative"]
    )
)