from sba_928_olanrewaju_muili import evaluation


def test_normalize_answer():
    assert evaluation.normalize_answer("  GBP 1,234.50 ") == "gbp 1234.50"
    assert evaluation.normalize_answer("£12") == "gbp12"
    assert evaluation.normalize_answer("Increased\n") == "increased"


def test_compare_models_and_scenario_accuracy():
    examples = [
        {"scenario": "Trends", "prompt": "p1", "answer": "increased"},
        {"scenario": "Trends", "prompt": "p2", "answer": "decreased"},
        {"scenario": "Demand", "prompt": "p3", "answer": "1,000"},
    ]
    comparison = evaluation.compare_models(
        examples,
        original_predictions=["increased", "increased", "999"],
        fine_tuned_predictions=["Increased", "decreased", "1000"],
    )

    assert list(comparison["original_correct"]) == [True, False, False]
    assert list(comparison["fine_tuned_correct"]) == [True, True, True]

    accuracy = evaluation.scenario_accuracy(comparison)
    assert accuracy.loc["Trends", "original_correct"] == 50.0
    assert accuracy.loc["Demand", "fine_tuned_correct"] == 100.0
