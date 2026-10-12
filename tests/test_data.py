import pandas as pd

from sba_928_olanrewaju_muili import data


def make_raw() -> pd.DataFrame:
    return pd.DataFrame({
        "InvoiceNo": ["1001", "1001", "C1002", "1003", "1004", "1005"],
        "Description": [" MUG ", "PLATE", "MUG", None, "MUG", "PLATE"],
        "Quantity": [2, 1, -2, 5, 3, -1],
        "UnitPrice": [1.5, 4.0, 1.5, 2.0, 1.5, 4.0],
        "InvoiceDate": pd.to_datetime([
            "2011-01-05", "2011-01-05", "2011-01-06",
            "2011-02-01", "2011-02-03", "2011-02-04",
        ]),
        "CustomerID": [1.0, 1.0, 1.0, None, None, 2.0],
        "Country": ["UK", "UK", "UK", "France", "France", "UK"],
    })


def test_clean_sales_drops_invalid_rows_and_keeps_missing_customers():
    sales = data.clean_sales(make_raw())

    # Cancelled, negative-quantity and missing-description rows are removed.
    assert list(sales["InvoiceNo"]) == ["1001", "1001", "1004"]
    # A missing CustomerID alone does not remove a sale.
    assert sales["CustomerID"].isna().sum() == 1
    assert list(sales["Description"]) == ["MUG", "PLATE", "MUG"]
    assert list(sales["Revenue"]) == [3.0, 4.0, 4.5]
    assert list(sales["Month"]) == ["2011-01", "2011-01", "2011-02"]


def test_country_statistics():
    stats = data.country_statistics(data.clean_sales(make_raw()))

    assert list(stats.index) == ["UK", "France"]  # sorted by revenue
    assert stats.loc["UK", "Revenue"] == 7.0
    assert stats.loc["UK", "Invoices"] == 1
    assert stats.loc["UK", "AverageItemsPerInvoice"] == 3.0


def test_monthly_statistics_is_chronological():
    monthly = data.monthly_statistics(data.clean_sales(make_raw()))
    assert list(monthly.index) == ["2011-01", "2011-02"]
    assert list(monthly) == [7.0, 4.5]
