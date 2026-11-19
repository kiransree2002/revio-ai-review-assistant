
from response_generator import generate_response

review = "My order arrived damaged. I want an immediate replacement."

sentiment = "negative"
topic = ["Product Quality"]
priority = "High"

response = generate_response(
    review,
    sentiment,
    topic,
    priority
)

print("Customer Review:", review)
print("\nSuggested Response:")
print(response)
