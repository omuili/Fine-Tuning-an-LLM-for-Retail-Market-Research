"""Requirement 2: market-research examples and prompt variations.

Each research record contains a scenario, the relevant facts from the
dataset, a market-research question and a verified answer. The facts are
included in the prompt so that both the original and the fine-tuned model
have the information needed to answer.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

Record = Dict[str, str]

TRAINING_STYLES = ("direct", "role", "structured")
EVALUATION_STYLE = "evaluation"


def money(value: float) -> str:
    return f"GBP {value:,.2f}"


def _record(scenario, group_type, group_id, context, question, answer) -> Record:
    return {
        "scenario": scenario,
        "group_type": group_type,
        "group_id": group_id,
        "context": context,
        "question": question,
        "answer": answer,
    }


def build_research_records(
    country_stats: pd.DataFrame,
    product_stats: pd.DataFrame,
    monthly_stats: pd.Series,
    num_countries: int,
    num_products: int,
    num_compared_products: int,
) -> List[Record]:
    """Create verified question/answer records for six research scenarios."""
    records: List[Record] = []

    # Every included country contributes the same number of examples.
    countries = country_stats.head(num_countries)
    products = product_stats.head(num_products)

    # Geographic market analysis.
    for country, row in countries.iterrows():
        context = (
            f"Country: {country}\n"
            f"Total revenue: {money(row['Revenue'])}\n"
            f"Total units: {int(row['Units'])}\n"
            f"Total invoices: {int(row['Invoices'])}\n"
            f"Average items per invoice: {row['AverageItemsPerInvoice']:.2f}"
        )
        records.append(_record(
            "Geographic market analysis", "country", country, context,
            f"What was the total sales revenue in {country}?",
            money(row["Revenue"]),
        ))

    # Consumer purchasing behavior. Average items per invoice measures
    # purchasing volume; it does not establish individual preferences.
    for country, row in countries.iterrows():
        context = (
            f"Country: {country}\n"
            f"Total units: {int(row['Units'])}\n"
            f"Total invoices: {int(row['Invoices'])}\n"
            f"Average items per invoice: {row['AverageItemsPerInvoice']:.2f}"
        )
        records.append(_record(
            "Consumer purchasing behavior", "country", country, context,
            f"What was the average number of items per invoice in {country}?",
            f"{row['AverageItemsPerInvoice']:.2f}",
        ))

    # Product demand.
    for product, row in products.iterrows():
        context = (
            f"Product: {product}\n"
            f"Units sold: {int(row['Units'])}\n"
            f"Revenue: {money(row['Revenue'])}"
        )
        records.append(_record(
            "Product demand", "product", product, context,
            f"How many units of {product} were sold?",
            str(int(row["Units"])),
        ))

    # Market trends: month-over-month revenue direction.
    months = list(monthly_stats.index)
    for previous_month, current_month in zip(months, months[1:]):
        previous_revenue = monthly_stats[previous_month]
        current_revenue = monthly_stats[current_month]

        if current_revenue > previous_revenue:
            trend = "increased"
        elif current_revenue < previous_revenue:
            trend = "decreased"
        else:
            trend = "remained unchanged"

        context = (
            f"Previous month: {previous_month}\n"
            f"Previous revenue: {money(previous_revenue)}\n"
            f"Current month: {current_month}\n"
            f"Current revenue: {money(current_revenue)}"
        )
        records.append(_record(
            "Market trends", "month", current_month, context,
            f"Did sales revenue increase, decrease, or remain unchanged in "
            f"{current_month} compared with {previous_month}?",
            trend,
        ))

    # Internal product competition. This compares products within the
    # same retailer; it does not measure competition between companies.
    product_names = list(products.index)
    for i in range(0, num_compared_products, 2):
        product_a, product_b = product_names[i], product_names[i + 1]
        revenue_a = product_stats.loc[product_a, "Revenue"]
        revenue_b = product_stats.loc[product_b, "Revenue"]

        if revenue_a > revenue_b:
            answer = product_a
        elif revenue_b > revenue_a:
            answer = product_b
        else:
            answer = "Equal revenue"

        context = (
            f"Product A: {product_a}\n"
            f"Product A revenue: {money(revenue_a)}\n"
            f"Product B: {product_b}\n"
            f"Product B revenue: {money(revenue_b)}"
        )
        records.append(_record(
            "Internal product competition", "comparison", f"pair_{i}", context,
            "Which product generated more revenue?",
            answer,
        ))

    # External competitor analysis. The dataset has no competitor sales,
    # so an accurate model must not invent market shares.
    for country in countries.index:
        context = (
            f"Country: {country}\n"
            "Available information: sales transactions from one retailer.\n"
            "External competitor sales: unavailable.\n"
            "Total market sales: unavailable."
        )
        records.append(_record(
            "External competitor analysis", "country", country, context,
            f"What is the retailer's market share relative to competitors "
            f"in {country}?",
            "Cannot determine market share from the available data.",
        ))

    return records


def create_prompt(record: Record, style: str) -> str:
    """Render a record in one of four prompt styles.

    The three training styles vary the wording; the evaluation style is a
    fourth wording that never appears in training.
    """
    context = record["context"]
    question = record["question"]

    if style == "direct":
        return (
            f"Data:\n{context}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )

    if style == "role":
        return (
            "You are a market research analyst.\n"
            "Use only the supplied research data.\n"
            "Do not invent missing information.\n\n"
            f"Data:\n{context}\n\n"
            f"Question: {question}\n"
            "Give a concise answer:"
        )

    if style == "structured":
        return (
            "Market Research Task\n\n"
            f"{context}\n\n"
            f"Research question: {question}\n\n"
            "Instructions:\n"
            "1. Use only the supplied data.\n"
            "2. Do not make unsupported assumptions.\n"
            "3. Return only the answer.\n\n"
            "Result:"
        )

    if style == EVALUATION_STYLE:
        return (
            "Analyze the following retail information.\n"
            "Answer using only the provided facts.\n"
            "If the information is insufficient, say so.\n\n"
            f"{context}\n\n"
            f"{question}\n\n"
            "Answer:"
        )

    raise ValueError(f"Unknown prompt style: {style}")


def split_by_group(
    records: List[Record], seed: int, train_fraction: float
) -> Tuple[List[Record], List[Record]]:
    """Split records so that no country, product, month or product pair
    appears in both the training and evaluation sets."""
    rng = np.random.default_rng(seed)
    train_groups = set()

    for group_type in sorted({r["group_type"] for r in records}):
        groups = sorted({
            r["group_id"] for r in records if r["group_type"] == group_type
        })
        rng.shuffle(groups)

        split = max(1, int(len(groups) * train_fraction))
        if len(groups) > 1:
            split = min(split, len(groups) - 1)

        train_groups.update((group_type, g) for g in groups[:split])

    train = [r for r in records if (r["group_type"], r["group_id"]) in train_groups]
    test = [r for r in records if (r["group_type"], r["group_id"]) not in train_groups]
    return train, test


def build_training_examples(records: List[Record], seed: int) -> List[Record]:
    """Render each record in the three training styles, balance the
    market-trend answers and shuffle."""
    examples = [
        {
            "scenario": r["scenario"],
            "style": style,
            "prompt": create_prompt(r, style),
            "answer": r["answer"],
        }
        for r in records
        for style in TRAINING_STYLES
    ]

    # Duplicate the minority "decreased" trend examples.
    examples.extend([
        e for e in examples
        if e["scenario"] == "Market trends" and e["answer"] == "decreased"
    ])

    np.random.default_rng(seed).shuffle(examples)
    return examples


def build_evaluation_examples(records: List[Record]) -> List[Record]:
    """Render held-out records in the unseen evaluation style."""
    return [
        {
            "scenario": r["scenario"],
            "style": EVALUATION_STYLE,
            "prompt": create_prompt(r, EVALUATION_STYLE),
            "answer": r["answer"],
        }
        for r in records
    ]
