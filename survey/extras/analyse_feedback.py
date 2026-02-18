import pandas as pd


def main():
    df = pd.read_csv("Data/cve_survey_feedback_2025_11_24_10_03_14.csv").convert_dtypes()
    df["CollectionDate"] = pd.to_datetime(df["CollectionDate"], dayfirst=True)

    likert_labels = [
        "Very poor",
        "Poor",
        "Neutral",
        "Good",
        "Excellent",
    ]

    df["ClarityLabel"] = df["Clarity"].map(lambda x: likert_labels[x-1])
    df["EaseLabel"] = df["Ease"].map(lambda x: likert_labels[x-1])
    df["ConfidenceLabel"] = df["Confidence"].map(lambda x: likert_labels[x-1])
    df["FatigueLabel"] = df["Fatigue"].map(lambda x: likert_labels[x-1])


    def likert_summary(series):
        counts = series.value_counts().sort_index()
        pct = (counts / len(series) * 100).round(1)
        return pd.DataFrame({"Count": counts, "Percent": pct})

    print("Clarity:")
    print(likert_summary(df["ClarityLabel"]))

    print("\nEase:")
    print(likert_summary(df["EaseLabel"]))

    print("\nConfidence:")
    print(likert_summary(df["ConfidenceLabel"]))

    print("\nFatigue:")
    print(likert_summary(df["FatigueLabel"]))

    comments = df["FeedbackComment"].dropna()

    print("\n=== Comments ===")
    for c in comments:
        print("-", c)



    print(df[["Clarity", "Ease", "Confidence", "Fatigue"]].corr())

    valid_emails = df["FeedbackEmail"].notna() & df["FeedbackEmail"].str.contains("@")
    print(df[valid_emails])

if __name__ == "__main__":
    main()