
TOPIC_KEYWORDS = {
        
    "Delivery": [
        "delivery", "shipping", "arrived late",
        "arrived on time", "on time", "fast shipping",
        "delayed", "courier", "tracking",
        "not delivered", "late delivery",
        "quick delivery", "slow delivery",
        "delivery time", "shipment"
    ],

    "Pricing": [
        "price", "expensive", "cost",
        "charged", "payment", "refund"
    ],
    "Product Quality": [
        "quality", "damaged", "broken",
        "defective", "stopped working", "packaging"
    ],
    "Customer Service": [
        "support", "service", "complaint",
        "responded", "help", "staff"
    ]
}


def classify_topic(review):
    review = review.lower()

    matched_topics = []

    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(keyword in review for keyword in keywords):
            matched_topics.append(topic)

    if not matched_topics:
        return ["General Feedback"]

    return matched_topics
