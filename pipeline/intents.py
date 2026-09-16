"""
Intent taxonomy, named by reading outputs/intent_clusters.md (KMeans, k=11,
over 1200 embedded English customer messages - see scripts/cluster_intents.py).

Several raw clusters were merged into delivery_problem: clusters 1 ("order
gone missing"), 2 ("tracking says delivered/attempted, customer disputes
it"), 4 ("hasn't shipped / delivery date pushed back") and 7 ("guaranteed
delivery date broken, carrier at fault") were lexically and semantically
close enough that a downstream classifier couldn't reliably tell them apart,
and - more importantly - the right first response is the same for all four
(apologize, check status, offer to investigate/refund). Cluster 8 stayed
separate as prime_membership_value even though it's also a delivery
complaint, because customers there explicitly frame it as a membership-value/
churn question, which is a different escalation signal (retention risk) than
a one-off late package.

This merge is a judgment call, not something the clustering did for us -
logged in report/decision_log.md.
"""

INTENTS = {
    "delivery_problem": (
        "Package is late, missing, mis-tracked, or marked delivered/"
        "attempted when the customer says it wasn't."
    ),
    "prime_membership_value": (
        "Customer ties a late or broken delivery to their Prime membership, "
        "questioning its value or threatening to cancel."
    ),
    "refund_or_return_request": (
        "Customer wants a refund or return processed and it hasn't happened, "
        "or a return pickup hasn't occurred."
    ),
    "poor_support_experience": (
        "Complaint about the support interaction itself - unresponsive, "
        "rude, unresolved after multiple contacts - not a specific order issue."
    ),
    "payment_or_giftcard_issue": (
        "Problem with a gift card, payment method, EMI option, or an "
        "unauthorized/unexpected charge."
    ),
    "packaging_or_damage_complaint": (
        "Item arrived damaged, or complaint about excessive/wasteful packaging."
    ),
    "device_or_app_tech_support": (
        "Bug or setup problem with an Amazon device or app - Alexa, Fire "
        "tablet, Prime Video playback, account profile."
    ),
    "general_inquiry_or_other": (
        "Pre-purchase question, praise/thanks, or a complaint that doesn't "
        "fit another bucket (e.g. a search feature bug)."
    ),
}

# KMeans cluster id (from scripts/cluster_intents.py, k=11) -> intent name.
# Labels stored as str because the baseline classifier's y is cast to str.
CLUSTER_TO_INTENT = {
    "0": "poor_support_experience",
    "1": "delivery_problem",
    "2": "delivery_problem",
    "3": "payment_or_giftcard_issue",
    "4": "delivery_problem",
    "5": "general_inquiry_or_other",
    "6": "refund_or_return_request",
    "7": "delivery_problem",
    "8": "prime_membership_value",
    "9": "packaging_or_damage_complaint",
    "10": "device_or_app_tech_support",
}

MAJORITY_INTENT = "delivery_problem"
