import os
import random
import csv
from datetime import datetime, timedelta

import pandas as pd  # for tabular data handling

# Ensure data directory exists
os.makedirs("data", exist_ok=True)

# --- Configuration ---
N_COMPLAINTS = 10_000

CATEGORIES = [
    "Water Supply",
    "Roads",
    "Waste Management",
    "Safety",
    "Health",
    "Electricity",
    "Sanitation",
    "Public Transport",
]

WARDS = [f"Ward {i}" for i in range(1, 51)]
DISTRICTS = ["Mumbai", "Delhi", "Jaipur"]

GENDERS = ["M", "F", "Other"]
CASTES = ["General", "OBC", "SC", "ST"]
INCOME_BRACKETS = ["0-3L", "3-6L", "6-10L", "10L+"]

# Simple templates to make the text more realistic
TEXT_TEMPLATES = [
    "There is a serious {category} issue in {ward}.",
    "Residents of {ward} are facing {category_lower} problems for many days.",
    "{category} problem has not been resolved in {ward} for weeks.",
    "Despite multiple complaints, the {category_lower} issue in {ward} continues.",
    "Urgent attention needed for {category_lower} in {ward}.",
]

def biased_choice(options, weights):
    """Helper to choose with weights."""
    return random.choices(options, weights=weights, k=1)[0]

def generate_complaints(n: int) -> pd.DataFrame:
    rows = []

    for i in range(n):
        category = random.choice(CATEGORIES)
        ward = random.choice(WARDS)
        district = random.choice(DISTRICTS)

        gender = biased_choice(GENDERS, weights=[0.45, 0.5, 0.05])  # slightly more F
        caste = biased_choice(CASTES, weights=[0.45, 0.25, 0.2, 0.1])
        income = biased_choice(INCOME_BRACKETS, weights=[0.35, 0.3, 0.2, 0.15])

        # Submission date within last 365 days
        days_ago = random.randint(0, 365)
        date_submitted = datetime.now() - timedelta(days=days_ago)

        # Base probabilities for response_status
        # We will introduce some bias here intentionally
        if income == "0-3L":
            status_weights = [0.6, 0.25, 0.15]  # more NO_RESPONSE
        elif income == "10L+":
            status_weights = [0.25, 0.35, 0.4]  # more RESOLVED
        else:
            status_weights = [0.5, 0.3, 0.2]

        response_status = biased_choice(["NO_RESPONSE", "RESPONDED", "RESOLVED"], status_weights)

        # Days in system: if resolved, cap by when it might reasonably be resolved
        if response_status == "RESOLVED":
            days_in_system = random.randint(1, min(days_ago + 1, 120))
        else:
            days_in_system = min(days_ago + 1, 365)

        text_template = random.choice(TEXT_TEMPLATES)
        # Precompute a lowercase variant for templates that need it.
        text = text_template.format(category=category, category_lower=category.lower(), ward=ward)

        rows.append(
            {
                "id": i + 1,
                "text": text,
                "category": category,
                "date_submitted": date_submitted.date().isoformat(),
                "response_status": response_status,
                "days_in_system": days_in_system,
                "gender": gender,
                "caste": caste,
                "income_bracket": income,
                "ward": ward,
                "district": district,
            }
        )

    return pd.DataFrame(rows)

if __name__ == "__main__":
    df = generate_complaints(N_COMPLAINTS)
    out_path = "data/complaints_raw.csv"
    # Quote all fields so commas in text do not break CSV parsing later.
    df.to_csv(out_path, index=False, quoting=csv.QUOTE_ALL)
    print(f"✓ Generated {len(df)} complaints → {out_path}")
    print("Columns:", list(df.columns))
    print(df.head().to_string(index=False))