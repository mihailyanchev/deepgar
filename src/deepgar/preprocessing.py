"""Build the per-country modelling datasets from the interim EU data.

Reads the quarterly series in ``data/interim`` and writes one
``data_<country>_<timestamp>.csv`` per country into ``data/processed``.
"""

from datetime import datetime

import pandas as pd

from deepgar.paths import INTERIM_DATA_DIR, PROCESSED_DATA_DIR

START_PERIOD_COMMON = "2000Q1"
END_PERIOD_COMMON = "2021Q4"


def _load_growths(
    filename: str,
    label: str,
    qoq_growth: bool,
    start_period: str,
    period_q: int,
    end_period: str | None = None,
    lead: bool = True,
) -> pd.DataFrame:
    df = pd.read_csv(INTERIM_DATA_DIR / filename, index_col=0)

    keep = df.index >= start_period
    if end_period is not None:
        keep &= df.index <= end_period
    df = df[keep]

    if qoq_growth:
        print(f"Preprocessing {label} data")
        df = df.pct_change(periods=period_q).iloc[period_q:,]

    if lead:
        df = df.shift(periods=-1)

    return df


def fetch_gdp_data(
    qoq_growth: bool = True,
    start_period: str = START_PERIOD_COMMON,
    end_period: str = END_PERIOD_COMMON,
    period_q: int = 1,
) -> pd.DataFrame:
    return _load_growths(
        "gdp.csv", "GDP", qoq_growth, start_period, period_q, end_period=end_period, lead=False
    )


def fetch_snp_data(
    qoq_growth: bool = True, start_period: str = START_PERIOD_COMMON, period_q: int = 1
) -> pd.DataFrame:
    return _load_growths("snp500.csv", "S&P 500", qoq_growth, start_period, period_q)


def fetch_sentiment_data(
    qoq_growth: bool = True, start_period: str = START_PERIOD_COMMON, period_q: int = 1
) -> pd.DataFrame:
    return _load_growths("sentiment.csv", "sentiment", qoq_growth, start_period, period_q)


def fetch_us_bond_data(
    qoq_growth: bool = True, start_period: str = START_PERIOD_COMMON, period_q: int = 1
) -> pd.DataFrame:
    return _load_growths("us_bond_yield.csv", "US bond yield", qoq_growth, start_period, period_q)


def main() -> None:
    """Assemble and export one modelling dataset per country."""
    gdp_df_growths = fetch_gdp_data()
    snp_growths = fetch_snp_data()
    sentiment = fetch_sentiment_data()
    us_bond_yield = fetch_us_bond_data()

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for country in gdp_df_growths.columns:
        print(f"Preparing file for {country}")
        df_list = [
            gdp_df_growths[country],
            sentiment[country],
            snp_growths["SNP_VOL"],
            snp_growths["SNP_CLOSE"],
            us_bond_yield["US_BOND_YIELD"],
        ]

        final_df = pd.concat(df_list, axis=1)
        final_df = final_df.iloc[1:, :]
        final_df = final_df.set_axis(
            ["GDP", "SENTIMENT", "SNP_VOL", "SNP_CLOSE", "US_BOND_YIELD"], axis=1
        )
        final_df = final_df[final_df.index <= END_PERIOD_COMMON]

        timestamp = datetime.now().strftime("%d.%m.%Y_%H.%M.%S")
        final_df.to_csv(PROCESSED_DATA_DIR / f"data_{country}_{timestamp}.csv")
        print(f"Exporting file to csv format at {timestamp}")

    print("Complete!")


if __name__ == "__main__":
    main()
