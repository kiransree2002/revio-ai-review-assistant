
HIGH_PRIORITY_KEYWORDS = [
    "immediately",
    "urgent",
    "charged twice",
    "double charged",
    "refund",
    "damaged",
    "broken",
    "defective",
    "third complaint",
    "never responded",
    "nobody has helped",
    "not working"
]

MEDIUM_PRIORITY_KEYWORDS = [
    "late",
    "delay",
    "expensive",
    "disappointed",
    "poor",
    "complaint",
    "slow",
    "problem"
]


def detect_priority(review, sentiment):
    review = review.lower()

    if any(word in review for word in HIGH_PRIORITY_KEYWORDS):
        return "High"

    if sentiment == "negative" or any(
        word in review for word in MEDIUM_PRIORITY_KEYWORDS
    ):
        return "Medium"

    return "Low"
