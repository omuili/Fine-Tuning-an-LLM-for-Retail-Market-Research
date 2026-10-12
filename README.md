# Fine-Tuning an LLM for Retail Market Research

**Course assessment:** SBA 928

This project builds an end-to-end market research workflow on the
[UCI Online Retail dataset](https://archive.ics.uci.edu/dataset/352/online+retail).
It turns real transaction data into market-research prompts, fine-tunes
`google/flan-t5-small` on them, compares the original and fine-tuned models
on held-out prompts, and assesses bias and fairness in the data and results.

**Result:** accuracy on held-out prompts rose from **65.2%** to
**87.0%** after two epochs of fine-tuning.

## Project structure

```
SBA_928_Olanrewaju_Muili/
├── pyproject.toml                
├── uv.lock                       
├── .python-version               
├── data/
│   └── online+retail.zip         
├── src/sba_928_olanrewaju_muili/
│   ├── __init__.py               
│   ├── __main__.py               
│   ├── config.py                 
│   ├── data.py                   
│   ├── prompts.py                
│   ├── model.py                  
│   ├── evaluation.py             
│   ├── bias.py                   
│   └── pipeline.py               
├── tests/                        
├── models/
│   └── fine_tuned_model/         
└── results/                    
```

## Setup and running

You need [uv](https://docs.astral.sh/uv/) installed.

```bash
git clone <this-repo-url>
cd SBA_928_Olanrewaju_Muili

# Create the virtual environment and install the locked dependencies
uv sync

# Run the full pipeline
uv run sba-928-olanrewaju-muili

# Run the unit tests
uv run pytest
```

On the first run the pipeline downloads `google/flan-t5-small` from Hugging Face.
It trains on a GPU when one is available and falls back to the CPU otherwise
(about a minute on a laptop CPU). Outputs go to `results/` and the model goes
to `models/fine_tuned_model/`. Hyperparameters are in
[`config.py`](src/sba_928_olanrewaju_muili/config.py).

## Methodology

### 1. Dataset preparation (`data.py`)
- Loads `Online Retail.xlsx` straight from the ZIP archive.
- Reports missing values, duplicates, negative or zero prices, and cancelled invoices.
- Keeps valid sales: positive quantity and price, not cancelled, description present.
  Rows with a missing `CustomerID` are kept for sales analysis.
- Adds `Revenue` and `Month` columns, then computes statistics by country, product and month.

### 2. Prompt variations (`prompts.py`)
Builds verified question/answer records for six market-research scenarios:

| Scenario | Example question |
|---|---|
| Geographic market analysis | What was the total sales revenue in *country*? |
| Consumer purchasing behavior | What was the average number of items per invoice in *country*? |
| Product demand | How many units of *product* were sold? |
| Market trends | Did revenue increase, decrease or stay the same versus last month? |
| Internal product competition | Which of two products generated more revenue? |
| External competitor analysis | What is the market share relative to competitors? (should answer that it cannot be determined) |

Each training record is written in three prompt styles: **direct**, **role-based**
and **structured**. Evaluation uses a fourth wording that never appears in training.
The train/test split is made by group (country, product, month), so no entity
appears in both sets. Minority market-trend answers are oversampled to balance
the classes.

### 3. Fine-tuning (`model.py`)
- Model: `google/flan-t5-small`
- 2 epochs, batch size 4, AdamW with learning rate 1e-4, seed 42
- Generation uses deterministic beam search (2 beams, no sampling)

### 4. Model comparison (`evaluation.py`)
Both models answer the same held-out prompts. Answers are scored by exact match
after normalisation, and accuracy is reported overall and by scenario.

### 5–6. Bias assessment and fairness (`bias.py`)
- Country representation in the transactions
- Missing customer identifiers, overall and by country
- Differences in model performance across scenarios
- Mitigation strategies and limitations, written to `results/fairness_strategies.json`

## Results

| Metric | Value |
|---|---|
| Original rows | 541,909 |
| Cleaned rows | 530,104 |
| Training examples | 276 |
| Evaluation examples | 23 |
| Original model accuracy | 65.2% |
| **Fine-tuned model accuracy** | **87.0%** |
| UK share of transactions | 91.5% |
| Missing customer IDs | 24.9% |

Accuracy by scenario (%):

| Scenario | Original | Fine-tuned |
|---|---|---|
| Consumer purchasing behavior | 100.0 | 100.0 |
| External competitor analysis | 0.0 | 100.0 |
| Geographic market analysis | 100.0 | 100.0 |
| Internal product competition | 0.0 | 100.0 |
| Market trends | 66.7 | 0.0 |
| Product demand | 83.3 | 100.0 |

Fine-tuning taught the model to refuse unsupported competitor conclusions and
to compare products correctly. It got worse on the market-trend questions,
which shows why accuracy should be reported per scenario and not only overall.

## Output files (`results/`)

| File | Contents |
|---|---|
| `country_statistics.csv` | Revenue, units, invoices and items per invoice by country |
| `product_statistics.csv` | Revenue and units by product |
| `monthly_statistics.csv` | Revenue by month |
| `training_prompts.csv` | Training prompts (3 styles) with target answers |
| `evaluation_prompts.csv` | Held-out evaluation prompts |
| `training_loss.csv` | Average loss per epoch |
| `model_comparison.csv` | Original vs. fine-tuned output for each evaluation prompt |
| `scenario_accuracy.csv` | Accuracy by scenario |
| `country_representation.csv` | Share of transactions by country |
| `customer_data_bias.csv` | Missing customer ID rate by country |
| `fairness_strategies.json` | Bias mitigation strategies and limitations |
| `experiment_summary.json` | Headline metrics |

Two large generated files are left out of the repository and are recreated
when you run the pipeline:

- `models/fine_tuned_model/model.safetensors`: the model weights (~294 MB,
  over GitHub's 100 MB file limit)
- `results/cleaned_retail.csv`: the cleaned dataset (~52 MB)

## Limitations
- The dataset comes from a single UK-based retailer, and 91.5% of its
  transactions are from the UK, so findings do not generalise to other markets.
- There is no demographic data, so fairness across protected groups cannot be assessed.
- There is no competitor or total-market data, so market share cannot be calculated.
- The evaluation set is small (23 prompts), so per-scenario accuracies rest on only a few examples.
