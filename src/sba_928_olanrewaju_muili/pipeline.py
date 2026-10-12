"""End-to-end SBA 928 experiment: data -> prompts -> fine-tuning ->
evaluation -> bias assessment."""

import json

import numpy as np
import pandas as pd
import torch

from . import bias, config, data, evaluation, model, prompts


def _save_json(obj, path) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def run() -> None:
    results_dir = config.RESULTS_DIR
    results_dir.mkdir(exist_ok=True)
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)

    device = model.get_device()
    print("Device:", device)

    # ------------------------------------------------------------------
    print("\nREQUIREMENT 1: DATASET PREPARATION")

    df = data.load_raw_data(config.DATA_PATH)
    data.report_data_quality(df)

    sales = data.clean_sales(df)
    print("\nCleaned dataset:", sales.shape)
    sales.to_csv(results_dir / "cleaned_retail.csv", index=False)

    country_stats = data.country_statistics(sales)
    product_stats = data.product_statistics(sales)
    monthly_stats = data.monthly_statistics(sales)

    country_stats.to_csv(results_dir / "country_statistics.csv")
    product_stats.to_csv(results_dir / "product_statistics.csv")
    monthly_stats.to_csv(results_dir / "monthly_statistics.csv")

    print("\nTotal revenue:", round(sales["Revenue"].sum(), 2))
    print("\nTop five countries:")
    print(country_stats.head())
    print("\nTop five products:")
    print(product_stats.head())

    records = prompts.build_research_records(
        country_stats,
        product_stats,
        monthly_stats,
        num_countries=config.NUM_COUNTRIES,
        num_products=config.NUM_PRODUCTS,
        num_compared_products=config.NUM_COMPARED_PRODUCTS,
    )
    print("\nCurated research examples:", len(records))

    # ------------------------------------------------------------------
    print("\nREQUIREMENT 2: PROMPT VARIATIONS")

    train_records, test_records = prompts.split_by_group(
        records, seed=config.SEED, train_fraction=config.TRAIN_FRACTION
    )
    training_examples = prompts.build_training_examples(train_records, config.SEED)
    evaluation_examples = prompts.build_evaluation_examples(test_records)

    print("Training examples:", len(training_examples))
    print("Evaluation examples:", len(evaluation_examples))

    pd.DataFrame(training_examples).to_csv(
        results_dir / "training_prompts.csv", index=False
    )
    pd.DataFrame(evaluation_examples).to_csv(
        results_dir / "evaluation_prompts.csv", index=False
    )

    # ------------------------------------------------------------------
    print("\nREQUIREMENT 3: MODEL FINE-TUNING")

    tokenizer, llm = model.load_model(config.MODEL_NAME, device)

    print("\nREQUIREMENT 4: ORIGINAL MODEL EVALUATION")

    original_predictions = model.predict(llm, tokenizer, evaluation_examples, device)
    print("Original model evaluation completed.")

    print("\nStarting fine-tuning...")
    training_losses = model.fine_tune(
        llm,
        tokenizer,
        training_examples,
        device,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        learning_rate=config.LEARNING_RATE,
    )

    llm.save_pretrained(config.MODEL_DIR)
    tokenizer.save_pretrained(config.MODEL_DIR)
    pd.DataFrame(training_losses).to_csv(
        results_dir / "training_loss.csv", index=False
    )
    print("\nFine-tuning completed.")

    print("\nEvaluating fine-tuned model...")
    fine_tuned_predictions = model.predict(
        llm, tokenizer, evaluation_examples, device
    )

    comparison = evaluation.compare_models(
        evaluation_examples, original_predictions, fine_tuned_predictions
    )
    comparison.to_csv(results_dir / "model_comparison.csv", index=False)

    original_accuracy = comparison["original_correct"].mean() * 100
    fine_tuned_accuracy = comparison["fine_tuned_correct"].mean() * 100

    print("\nMODEL COMPARISON")
    print(f"Original accuracy: {original_accuracy:.2f}%")
    print(f"Fine-tuned accuracy: {fine_tuned_accuracy:.2f}%")

    scenario_accuracy = evaluation.scenario_accuracy(comparison)
    print("\nAccuracy by research scenario:")
    print(scenario_accuracy)
    scenario_accuracy.to_csv(results_dir / "scenario_accuracy.csv")

    # ------------------------------------------------------------------
    print("\nREQUIREMENT 5: BIAS ASSESSMENT")

    country_share = bias.country_representation(sales)
    country_share.to_csv(
        results_dir / "country_representation.csv", header=["Percentage"]
    )
    uk_share = country_share.get("United Kingdom", 0)
    print(f"\nUK transaction representation: {uk_share:.2f}%")

    missing_customer = bias.missing_customer_rate(df)
    print(f"Missing customer identifiers: {missing_customer:.2f}%")

    missing_by_country = bias.missing_customer_by_country(sales)
    missing_by_country.to_csv(results_dir / "customer_data_bias.csv")
    print("\nCustomer data coverage:")
    print(missing_by_country.head(10))

    print("\nModel performance differences:")
    print(scenario_accuracy)

    # ------------------------------------------------------------------
    print("\nREQUIREMENT 6: FAIRNESS STRATEGIES")

    _save_json(bias.FAIRNESS_STRATEGIES, results_dir / "fairness_strategies.json")
    for category, strategy in bias.FAIRNESS_STRATEGIES.items():
        print(f"\n{category}:")
        print(strategy)

    summary = {
        "dataset": "UCI Online Retail",
        "original_rows": int(len(df)),
        "cleaned_rows": int(len(sales)),
        "training_examples": len(training_examples),
        "evaluation_examples": len(evaluation_examples),
        "model": config.MODEL_NAME,
        "epochs": config.EPOCHS,
        "original_accuracy": float(original_accuracy),
        "fine_tuned_accuracy": float(fine_tuned_accuracy),
        "uk_transaction_percentage": float(uk_share),
        "missing_customer_percentage": missing_customer,
    }
    _save_json(summary, results_dir / "experiment_summary.json")

    print("\n================================")
    print("SBA 928 EXPERIMENT COMPLETED")
    print("================================")
    print("\nResults saved to:", results_dir.resolve())
    print("Model saved to:", config.MODEL_DIR.resolve())
