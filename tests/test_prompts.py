import pandas as pd
import pytest

from sba_928_olanrewaju_muili import prompts


@pytest.fixture
def records():
    countries = pd.DataFrame(
        {
            "Revenue": [1000.0, 500.0, 250.0, 100.0, 50.0],
            "Units": [100, 50, 25, 10, 5],
            "Invoices": [10, 5, 5, 2, 1],
            "AverageItemsPerInvoice": [10.0, 10.0, 5.0, 5.0, 5.0],
        },
        index=["UK", "France", "Germany", "Spain", "Italy"],
    )
    products = pd.DataFrame(
        {"Revenue": [300.0, 200.0, 100.0, 50.0], "Units": [30, 20, 10, 5]},
        index=["A", "B", "C", "D"],
    )
    monthly = pd.Series(
        [100.0, 80.0, 120.0, 120.0],
        index=["2011-01", "2011-02", "2011-03", "2011-04"],
    )
    return prompts.build_research_records(
        countries, products, monthly,
        num_countries=5, num_products=4, num_compared_products=4,
    )


def test_records_cover_all_scenarios(records):
    assert {r["scenario"] for r in records} == {
        "Geographic market analysis",
        "Consumer purchasing behavior",
        "Product demand",
        "Market trends",
        "Internal product competition",
        "External competitor analysis",
    }


def test_market_trend_answers(records):
    trends = [r["answer"] for r in records if r["scenario"] == "Market trends"]
    assert trends == ["decreased", "increased", "remained unchanged"]


def test_competitor_questions_are_unanswerable(records):
    answers = {
        r["answer"] for r in records
        if r["scenario"] == "External competitor analysis"
    }
    assert answers == {"Cannot determine market share from the available data."}


def test_money_format():
    assert prompts.money(1234567.891) == "GBP 1,234,567.89"


@pytest.mark.parametrize("style", [*prompts.TRAINING_STYLES, prompts.EVALUATION_STYLE])
def test_every_prompt_style_contains_context_and_question(records, style):
    prompt = prompts.create_prompt(records[0], style)
    assert records[0]["context"] in prompt
    assert records[0]["question"] in prompt


def test_unknown_style_raises(records):
    with pytest.raises(ValueError):
        prompts.create_prompt(records[0], "nonexistent")


def test_split_has_no_group_overlap(records):
    train, test = prompts.split_by_group(records, seed=42, train_fraction=0.8)

    train_groups = {(r["group_type"], r["group_id"]) for r in train}
    test_groups = {(r["group_type"], r["group_id"]) for r in test}

    assert train and test
    assert not train_groups & test_groups
    assert len(train) + len(test) == len(records)


def test_split_is_reproducible(records):
    first = prompts.split_by_group(records, seed=42, train_fraction=0.8)
    second = prompts.split_by_group(records, seed=42, train_fraction=0.8)
    assert first == second


def test_training_examples_use_three_styles_and_balance_trends(records):
    examples = prompts.build_training_examples(records, seed=42)

    assert {e["style"] for e in examples} == set(prompts.TRAINING_STYLES)
    decreased = [
        e for e in examples
        if e["scenario"] == "Market trends" and e["answer"] == "decreased"
    ]
    # One "decreased" record x 3 styles, duplicated once.
    assert len(decreased) == 6


def test_evaluation_examples_use_unseen_style(records):
    examples = prompts.build_evaluation_examples(records)
    assert {e["style"] for e in examples} == {prompts.EVALUATION_STYLE}
