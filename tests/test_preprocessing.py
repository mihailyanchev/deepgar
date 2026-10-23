import pandas as pd

from deepgar.paths import PROCESSED_DATA_DIR
from deepgar.preprocessing import (
    END_PERIOD_COMMON,
    fetch_gdp_data,
    fetch_sentiment_data,
    fetch_snp_data,
    fetch_us_bond_data,
)

# The only committed dataset produced by the preprocessing script as-is; the
# others were later extended by hand with country-specific indicators.
REFERENCE_FILE = "data_Romania_14.04.2022_13.00.19.csv"


def test_matches_committed_dataset() -> None:
    """Rebuilding the Romania dataset reproduces the file used in the paper."""
    committed = pd.read_csv(PROCESSED_DATA_DIR / REFERENCE_FILE, index_col=0)
    snp = fetch_snp_data()

    rebuilt = pd.concat(
        [
            fetch_gdp_data()["Romania"].rename("GDP"),
            fetch_sentiment_data()["Romania"].rename("SENTIMENT"),
            snp["SNP_VOL"],
            snp["SNP_CLOSE"],
            fetch_us_bond_data()["US_BOND_YIELD"],
        ],
        axis=1,
    ).iloc[1:]
    rebuilt = rebuilt[rebuilt.index <= END_PERIOD_COMMON]

    pd.testing.assert_frame_equal(rebuilt, committed, check_names=False)
