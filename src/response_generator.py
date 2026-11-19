
def generate_response(review, sentiment, topic, priority):

    if priority == "High":
        return (
            "We sincerely apologize for this experience. "
            "We understand the urgency of your concern. "
            "Our support team should review this issue as a priority. "
            "Please share your order details through our official "
            "support channel so we can investigate and assist you."
        )

    if sentiment == "negative":

        if "Delivery" in topic:
            return (
                "We apologize for the delay in your delivery. "
                "We understand how frustrating this can be. "
                "Please share your order details so our team can "
                "check the delivery status."
            )

        if "Pricing" in topic:
            return (
                "Thank you for sharing your feedback about our pricing. "
                "We appreciate your concern and will pass it to "
                "the relevant team for review."
            )

        if "Product Quality" in topic:
            return (
                "We are sorry that the product did not meet your "
                "expectations. Our team would like to understand "
                "the issue and help you with an appropriate resolution."
            )

        if "Customer Service" in topic:
            return (
                "We apologize for the support experience you received. "
                "Your feedback is important, and we will work to "
                "improve our customer service."
            )

        return (
            "We are sorry about your experience. "
            "Thank you for bringing this to our attention. "
            "Our team will review your feedback."
        )

    if sentiment == "positive":
        return (
            "Thank you for your wonderful feedback! "
            "We are delighted to know you had a positive experience. "
            "We appreciate your support."
        )

    return (
        "Thank you for sharing your feedback. "
        "We appreciate your suggestions and will use them "
        "to improve our products and services."
    )
