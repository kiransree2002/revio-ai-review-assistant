
from sentiment import train_sentiment_model, analyze_sentiment

# Train the model
model = train_sentiment_model()

# Test customer reviews
reviews = [
    "Excellent product and fast delivery",
    "The product is okay",
    "Terrible service and damaged product"
]

# Predict sentiment
for review in reviews:
    result = analyze_sentiment(review, model)

    print("\nReview:", review)
    print("Sentiment:", result["sentiment"])
    print("Confidence:", result["confidence"])
