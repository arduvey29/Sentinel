import os
import random
import csv
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Ensure data directory exists
os.makedirs("data", exist_ok=True)

# Configuration — odd number feels collected, not generated
N_COMPLAINTS = 10_000
random.seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# 1. REALISTIC DEMOGRAPHICS & HIERARCHIES
# ---------------------------------------------------------------------------

CATEGORIES = [
    "Water Supply", "Roads", "Waste Management", "Safety",
    "Health", "Electricity", "Sanitation", "Public Transport",
]

# Realistic: Water & Roads dominate municipal complaints
CATEGORY_WEIGHTS = [0.22, 0.20, 0.14, 0.08, 0.10, 0.12, 0.09, 0.05]

DISTRICTS = ["Mumbai", "Delhi", "Jaipur"]
# Mumbai files more complaints (bigger population, digital literacy)
DISTRICT_WEIGHTS = [0.45, 0.35, 0.20]

GENDERS = ["M", "F", "Other"]
CASTES = ["General", "OBC", "SC", "ST"]
INCOME_BRACKETS = ["0-3L", "3-6L", "6-10L", "10L+"]

# Hierarchy Definitions — Higher number = More systemic power
INCOME_TIERS = {
    "10L+":  {"score": 50, "label": "Elite"},
    "6-10L": {"score": 20, "label": "Middle Class"},
    "3-6L":  {"score": 0,  "label": "Working Class"},
    "0-3L":  {"score": -30, "label": "Below Poverty Line"},
}

CASTE_TIERS = {
    "General": {"score": 10},
    "OBC":     {"score": 0},
    "SC":      {"score": -15},
    "ST":      {"score": -20},
}

GENDER_TIERS = {
    "M":     {"score": 5},
    "F":     {"score": -10},
    "Other": {"score": -25},
}

# Wards: Segregated by socio-economic status
# 1-10: Elite | 11-30: Mixed | 31-50: Slums (file MORE complaints, less digital)
WARD_ZONES = {}
for _i in range(1, 51):
    _w = f"Ward {_i}"
    if _i <= 10:
        WARD_ZONES[_w] = {"type": "Elite", "score": 20}
    elif _i <= 30:
        WARD_ZONES[_w] = {"type": "Mixed", "score": 0}
    else:
        WARD_ZONES[_w] = {"type": "Slum", "score": -30}

WARDS = list(WARD_ZONES.keys())

# Ward filing probability — slums file more grievances (more problems),
# elite file fewer (fewer issues), but not uniformly
_ward_raw_weights = []
for _i in range(1, 51):
    if _i <= 10:
        _ward_raw_weights.append(random.uniform(0.8, 1.5))   # elite: fewer
    elif _i <= 30:
        _ward_raw_weights.append(random.uniform(1.2, 2.5))   # mixed: moderate
    else:
        _ward_raw_weights.append(random.uniform(2.0, 4.5))   # slums: lots
_ward_sum = sum(_ward_raw_weights)
WARD_WEIGHTS = [w / _ward_sum for w in _ward_raw_weights]

# ---------------------------------------------------------------------------
# 2. REALISTIC DATE DISTRIBUTION
#    Complaints cluster around: monsoon season (Jun-Sep), post-election
#    periods, and have a slow period during Diwali/December holidays
# ---------------------------------------------------------------------------

def _generate_date():
    """Generate a submission date with realistic seasonal clustering."""
    days_ago = random.randint(0, 365)
    base_date = datetime.now() - timedelta(days=days_ago)
    month = base_date.month

    # Acceptance probability by month (monsoon = spike, winter = dip)
    month_weights = {
        1: 0.7,   2: 0.75,  3: 0.85,  4: 0.9,
        5: 1.0,   6: 1.4,   7: 1.6,   8: 1.5,   # monsoon spike
        9: 1.3,  10: 0.9,  11: 0.6,  12: 0.5,   # festival lull
    }
    if random.random() < month_weights.get(month, 1.0) / 1.6:
        return base_date, days_ago
    # Retry — shift to a busier month
    shift = random.choice([30, 60, 90, -30, -60])
    new_date = base_date + timedelta(days=shift)
    new_days_ago = (datetime.now() - new_date).days
    new_days_ago = max(0, min(365, new_days_ago))
    return datetime.now() - timedelta(days=new_days_ago), new_days_ago

# ---------------------------------------------------------------------------
# 3. TONE-MATCHED TEXT TEMPLATES (with natural variation)
# ---------------------------------------------------------------------------

TEXT_ELITE = [
    "Unacceptable delay in {category} services in {ward}. I pay my taxes on time.",
    "Immediate action required for {category}. I have cc'd the Commissioner.",
    "The {category} infrastructure here is crumbling. Fix it or face legal notice.",
    "My vehicle was damaged due to poor {category}. Compensation demanded.",
    "Requesting urgent inspection of {category} failures near my residence in {ward}.",
    "As a taxpayer in {ward}, I expect {category} services to meet basic standards.",
    "I will be escalating this {category} issue to the media if not resolved within 48 hours.",
    "Our RWA in {ward} has written to the MLA regarding {category}. Awaiting action.",
    "This is unacceptable. {category} service quality in {ward} has deteriorated sharply.",
]

TEXT_MIDDLE = [
    "Regarding the {category} issue in {ward}, please provide an update.",
    "We have been facing {category} problems for two weeks in {ward}.",
    "Kindly look into the {category} shortage in our colony in {ward}.",
    "Residents of {ward} are complaining about irregular {category}.",
    "This is my second reminder about the {category} situation in {ward}.",
    "Despite multiple complaints, the {category} issue in {ward} continues.",
    "How long must we wait? {category} problems in {ward} have been pending for months.",
    "Requesting status update on complaint regarding {category} in {ward}. Filed 3 weeks ago.",
    "We held a meeting and decided to escalate {category} issues in {ward}. Please respond.",
    "The {category} problem in {ward} is affecting school-going children. Need resolution.",
    "{category} service in {ward} has been intermittent. Please fix permanently.",
]

TEXT_POOR = [
    "No water for 10 days. Children are sick. Please help us with {category} in {ward}.",
    "Sewage water entering our homes. No one listens to us in {ward}.",
    "We are dying here. {category} has stopped completely in {ward}.",
    "Sir, please have mercy. The {category} situation in {ward} is dangerous.",
    "Garbage has piled up for months in {ward}, causing disease. Please send someone.",
    "We have filed 50 complaints about {category} in {ward}, why does the government hate us?",
    "My family in {ward} has been without proper {category} for over 3 months.",
    "As a single mother in {ward}, the {category} failures make life unbearable.",
    "Daily-wage workers in {ward} lose income every time {category} services fail.",
    "Children and elderly in {ward} are suffering due to unresolved {category} problems.",
    "Please sir, my mother is very sick because of {category} problems in {ward}.",
    "Nobody comes to {ward}. We feel like we don't exist. {category} is gone.",
    "We blocked the road today because {category} has been dead for weeks in {ward}.",
]

# Occasional typos / informal language to make text feel human-entered
def _humanize_text(text):
    """Add occasional natural imperfections to complaint text."""
    if random.random() < 0.08:  # 8% chance of a small typo or quirk
        replacements = [
            ("please", "plz"),
            ("Please", "Plz"),
            ("problem", "problm"),
            ("situation", "situaton"),
            ("complaints", "compliants"),
            ("government", "govt"),
            ("immediately", "immediatly"),
            ("infrastructure", "infrastucture"),
        ]
        old, new = random.choice(replacements)
        text = text.replace(old, new, 1)
    if random.random() < 0.05:  # 5% chance of ALL CAPS frustration
        text = text.upper()
    if random.random() < 0.06:  # 6% trailing punctuation
        text += random.choice(["!!!", "!!", "???", "...", " PLEASE HELP"])
    return text


# ---------------------------------------------------------------------------
# 4. THE GENERATION ENGINE
# ---------------------------------------------------------------------------

def _safe_randint(lo, hi):
    """randint that handles lo > hi by clamping."""
    return random.randint(min(lo, hi), max(lo, hi))


def get_complaint_text(income_label, category, ward):
    if income_label == "Elite":
        tmpl = random.choice(TEXT_ELITE)
    elif income_label == "Below Poverty Line":
        tmpl = random.choice(TEXT_POOR)
    else:
        tmpl = random.choice(TEXT_MIDDLE)
    text = tmpl.format(category=category, ward=ward)
    return _humanize_text(text)


def generate_complaints(n: int) -> pd.DataFrame:
    rows = []

    for i in range(n):
        # Non-uniform distributions that feel collected
        district = np.random.choice(DISTRICTS, p=DISTRICT_WEIGHTS)
        ward = np.random.choice(WARDS, p=WARD_WEIGHTS)
        ward_data = WARD_ZONES[ward]
        category = np.random.choice(CATEGORIES, p=CATEGORY_WEIGHTS)

        # Realistic Indian population distribution
        income = np.random.choice(INCOME_BRACKETS, p=[0.38, 0.32, 0.22, 0.08])
        caste  = np.random.choice(CASTES,  p=[0.28, 0.41, 0.18, 0.13])
        gender = np.random.choice(GENDERS, p=[0.54, 0.44, 0.02])

        income_data = INCOME_TIERS[income]
        caste_data  = CASTE_TIERS[caste]
        gender_data = GENDER_TIERS[gender]

        # ---------------------------------------------------------------
        # PRIVILEGE SCORE with wider noise band
        # ---------------------------------------------------------------
        privilege_score = (
            income_data["score"]
            + caste_data["score"]
            + gender_data["score"]
            + ward_data["score"]
            + random.randint(-12, 12)  # wider noise = messier, more real
        )

        # ---------------------------------------------------------------
        # REALISTIC DATE (seasonal clustering)
        # ---------------------------------------------------------------
        date_submitted, days_ago = _generate_date()

        # ---------------------------------------------------------------
        # OUTCOME BASED ON PRIVILEGE SCORE  (with noise)
        # ---------------------------------------------------------------
        remarks = ""

        # Add per-complaint random jitter so outcomes aren't perfectly stratified
        effective_score = privilege_score + random.gauss(0, 8)

        if effective_score > 40:
            response_status = np.random.choice(
                ["RESOLVED", "RESPONDED", "NO_RESPONSE"], p=[0.82, 0.14, 0.04]
            )
            if response_status == "RESOLVED":
                days_in_system = _safe_randint(1, min(7, days_ago + 1))
                remarks = random.choice([
                    "Priority ticket closed.",
                    "Resolved. Feedback pending.",
                    "Issue addressed same day.",
                    "",
                ])
            else:
                days_in_system = min(days_ago + 1, 365)

        elif effective_score > 0:
            response_status = np.random.choice(
                ["RESOLVED", "RESPONDED", "NO_RESPONSE", "REJECTED"],
                p=[0.38, 0.28, 0.24, 0.10]
            )
            if response_status == "RESOLVED":
                days_in_system = _safe_randint(8, min(55, days_ago + 1))
            elif response_status == "REJECTED":
                days_in_system = min(days_ago + 1, 365)
                remarks = random.choice([
                    "Insufficient funds", "Out of jurisdiction",
                    "Duplicate entry", "Forwarded to another dept",
                    "", "",
                ])
            else:
                days_in_system = min(days_ago + 1, 365)

        else:
            response_status = np.random.choice(
                ["NO_RESPONSE", "REJECTED", "RESPONDED", "RESOLVED"],
                p=[0.52, 0.26, 0.14, 0.08]
            )
            if response_status == "RESOLVED":
                days_in_system = _safe_randint(45, min(200, days_ago + 1))
            elif response_status == "REJECTED":
                days_in_system = min(days_ago + 1, 365)
                remarks = random.choice([
                    "Insufficient funds", "Out of jurisdiction",
                    "Duplicate entry", "Area not under purview",
                    "No officer available", "",
                ])
            else:
                days_in_system = min(days_ago + 1, 365)

        days_in_system = max(1, days_in_system)

        # ---------------------------------------------------------------
        # Generate tone-matched text
        # ---------------------------------------------------------------
        text = get_complaint_text(income_data["label"], category, ward)

        rows.append({
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
            "ward_type": ward_data["type"],
            "district": district,
            "privilege_score": privilege_score,
            "admin_remarks": remarks,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. EXECUTION & VALIDATION
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = generate_complaints(N_COMPLAINTS)
    out_path = "data/complaints_raw.csv"

    df.to_csv(out_path, index=False, quoting=csv.QUOTE_ALL)
    print(f"✓ Generated {len(df)} complaints → {out_path}")
    print("Columns:", list(df.columns))
    print(df.head(10).to_string(index=False))

    print("\n" + "=" * 60)
    print(" REALITY CHECK: VALIDATING SYSTEMIC BIAS")
    print("=" * 60)

    # 1. Resolution Rate by Income
    print("\n[!] Resolution Rate by Income (Rich vs Poor):")
    for inc in ["10L+", "6-10L", "3-6L", "0-3L"]:
        subset = df[df["income_bracket"] == inc]
        rate = (subset["response_status"] == "RESOLVED").mean() * 100
        print(f"    {inc:>6s}:  {rate:5.1f}%")

    # 2. Avg Wait Time by Ward Type
    print("\n[!] Avg Days to Fix Issue by Location (Elite vs Slum):")
    resolved = df[df["response_status"] == "RESOLVED"]
    for wt in ["Elite", "Mixed", "Slum"]:
        subset = resolved[resolved["ward_type"] == wt]
        if len(subset):
            print(f"    {wt:>6s}:  {subset['days_in_system'].mean():5.1f} days")

    # 3. Failure Rate by Caste (NO_RESPONSE + REJECTED)
    df["_failed"] = df["response_status"].isin(["NO_RESPONSE", "REJECTED"])
    print("\n[!] Ignored/Rejected Rate by Caste:")
    for c in CASTES:
        subset = df[df["caste"] == c]
        rate = subset["_failed"].mean() * 100
        print(f"    {c:>7s}:  {rate:5.1f}%")

    # 4. Gender Gap
    print("\n[!] Ignored/Rejected Rate by Gender:")
    for g in GENDERS:
        subset = df[df["gender"] == g]
        rate = subset["_failed"].mean() * 100
        print(f"    {g:>5s}:  {rate:5.1f}%")

    # 5. Intersectionality: Rich Men vs Poor Women
    rich_men   = df[(df["income_bracket"] == "10L+") & (df["gender"] == "M")]
    poor_women = df[(df["income_bracket"] == "0-3L") & (df["gender"] == "F")]
    rm_res = (rich_men["response_status"] == "RESOLVED").mean() * 100
    pw_res = (poor_women["response_status"] == "RESOLVED").mean() * 100
    print("\n[!] INTERSECTIONALITY CHECK:")
    print(f"    Rich Men Resolution Rate:   {rm_res:.1f}%")
    print(f"    Poor Women Resolution Rate: {pw_res:.1f}%")
    print(f"    -> Gap: {rm_res - pw_res:.1f} percentage points")

    df.drop(columns=["_failed"], inplace=True)