"""Requirement 1: dataset loading, cleaning and market statistics."""

import zipfile
from io import BytesIO
from pathlib import Path

import pandas as pd


def load_raw_data(path: Path) -> pd.DataFrame:
    """Load the Online Retail Excel file directly from its ZIP archive."""
    with zipfile.ZipFile(path, "r") as archive:
        excel_name = next(
            name for name in archive.namelist() if name.lower().endswith(".xlsx")
        )
        with archive.open(excel_name) as f:
            return pd.read_excel(BytesIO(f.read()))


def report_data_quality(df: pd.DataFrame) -> None:
    """Print missing values, duplicates and invalid or special records."""
    print("Original dataset:", df.shape)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate rows:", df.duplicated().sum())

    print("\nNegative quantities:", (df["Quantity"] < 0).sum())
    print("Negative prices:", (df["UnitPrice"] < 0).sum())
    print("Zero prices:", (df["UnitPrice"] == 0).sum())
    print("Cancelled records:", is_cancelled(df).sum())


def is_cancelled(df: pd.DataFrame) -> pd.Series:
    """Cancelled invoices have numbers starting with 'C'."""
    return df["InvoiceNo"].astype(str).str.startswith("C")


def clean_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Keep positive-value, non-cancelled sales and add Revenue and Month.

    Rows with a missing CustomerID are kept for sales analysis; the raw
    dataframe is left untouched for the bias assessment.
    """
    sales = df[
        (df["Quantity"] > 0)
        & (df["UnitPrice"] > 0)
        & df["Description"].notna()
        & ~is_cancelled(df)
    ].copy()

    sales["Description"] = sales["Description"].astype(str).str.strip()
    sales["Revenue"] = sales["Quantity"] * sales["UnitPrice"]
    sales["Month"] = sales["InvoiceDate"].dt.to_period("M").astype(str)

    return sales


def country_statistics(sales: pd.DataFrame) -> pd.DataFrame:
    """Revenue, units, invoices and average items per invoice by country."""
    stats = sales.groupby("Country").agg(
        Revenue=("Revenue", "sum"),
        Units=("Quantity", "sum"),
        Invoices=("InvoiceNo", "nunique"),
    )
    stats["AverageItemsPerInvoice"] = stats["Units"] / stats["Invoices"]
    return stats.sort_values("Revenue", ascending=False)


def product_statistics(sales: pd.DataFrame) -> pd.DataFrame:
    """Revenue and units sold by product."""
    stats = sales.groupby("Description").agg(
        Revenue=("Revenue", "sum"),
        Units=("Quantity", "sum"),
    )
    return stats.sort_values("Revenue", ascending=False)


def monthly_statistics(sales: pd.DataFrame) -> pd.Series:
    """Total revenue by month, in chronological order."""
    return sales.groupby("Month")["Revenue"].sum().sort_index()
