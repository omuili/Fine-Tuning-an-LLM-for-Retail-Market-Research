"""Requirements 5 and 6: bias assessment and fairness strategies."""

import pandas as pd


def country_representation(sales: pd.DataFrame) -> pd.Series:
    """Percentage of transactions from each country."""
    return sales["Country"].value_counts() / len(sales) * 100


def missing_customer_rate(df: pd.DataFrame) -> float:
    """Percentage of raw records without a CustomerID."""
    return float(df["CustomerID"].isna().mean() * 100)


def missing_customer_by_country(sales: pd.DataFrame) -> pd.DataFrame:
    """Missing-CustomerID rate and transaction count per country."""
    stats = (
        sales.assign(MissingCustomer=sales["CustomerID"].isna())
        .groupby("Country")["MissingCustomer"]
        .agg(["mean", "size"])
    )
    stats["mean"] *= 100
    stats.columns = ["MissingCustomerPercentage", "TransactionCount"]
    return stats


FAIRNESS_STRATEGIES = {
    "Geographic representation": (
        "The retailer's transaction data may contain unequal country "
        "representation. Report country-specific findings rather than "
        "generalizing results to all consumers. The prompt dataset gives each "
        "included country the same number of country-level research examples."
    ),
    "Missing customer information": (
        "Retain transactions with missing customer IDs for overall sales "
        "analysis. Exclude them from analyses requiring customer "
        "identification. Report missing-data rates by country and avoid "
        "assuming missing records are random."
    ),
    "Prompt bias": (
        "Use neutral prompt wording. Avoid questions that assume a country, "
        "product or customer segment is superior without evidence. Require "
        "conclusions to use supplied data."
    ),
    "Unsupported competitor conclusions": (
        "Do not calculate market share without competitor sales and total "
        "market data. State that competitor comparisons cannot be determined "
        "from this dataset."
    ),
    "Evaluation fairness": (
        "Evaluate both models on identical questions with identical factual "
        "context. Report performance separately by research scenario to "
        "identify uneven results."
    ),
    "Limitations": (
        "The dataset does not provide sufficient demographic information to "
        "evaluate fairness across protected demographic groups. Geographic "
        "representation and scenario-level accuracy are not substitutes for a "
        "demographic fairness assessment."
    ),
}
