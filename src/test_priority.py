
from priority import detect_priority

reviews = [
    ("I was charged twice. Refund immediately!", "negative"),
    ("My delivery was slightly late", "negative"),
    ("Excellent product and great service", "positive")
]

for review, sentiment in reviews:
    priority = detect_priority(review, sentiment)

    print("Review:", review)
    print("Priority:", priority)
    print()
