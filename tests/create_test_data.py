"""
Create comprehensive test HAR file for harpyx testing.

Creates test_harpyx.har with:
- Set headers: COM (3), REG (4), IND (5)
- PROD: 2D labeled float (COM × IND)
- CONS: 2D labeled float (COM × REG)
- TRAD: 3D with duplicate dim (COM × REG × REG) — critical test case
- GDP:  scalar (0D)
- CNTS: 2D labeled integer (COM × REG)
- LABR: 2D labeled float (IND × REG)

Run from the project root:
    python tests/create_test_data.py
"""

from pathlib import Path

import numpy as np

from harpy import HarFileObj
from harpy.header_array import HeaderArrayObj


def create_test_harpyx_har() -> Path:
    """Create comprehensive test HAR file and return its path."""

    output_path = Path(__file__).parent / "testdata" / "test_harpyx.har"
    print(f"Creating test HAR file: {output_path}")

    hf = HarFileObj(str(output_path))

    # --- Set headers (1C) ---

    print("Creating set headers...")

    # SetHeaderFromData auto-prepends "Set {name} " to the long_name
    hf["SCOM"] = HeaderArrayObj.SetHeaderFromData(
        setName="COM",
        setElements=["Agriculture", "Manufacture", "Services"],
        long_name="commodities in the model",
    )
    hf["SREG"] = HeaderArrayObj.SetHeaderFromData(
        setName="REG",
        setElements=["USA", "EU", "China", "Japan"],
        long_name="regions in the model",
    )
    hf["SIND"] = HeaderArrayObj.SetHeaderFromData(
        setName="IND",
        setElements=["i1", "i2", "i3", "i4", "i5"],
        long_name="industries in the model",
    )

    # --- Data headers ---

    print("Creating data headers...")

    # PROD: COM × IND
    prod_data = np.array(
        [
            [1.1, 2.2, 3.3, 4.4, 5.5],
            [6.6, 7.7, 8.8, 9.9, 10.1],
            [11.2, 12.3, 13.4, 14.5, 15.6],
        ],
        dtype=np.float32,
    )
    hf["PROD"] = HeaderArrayObj.HeaderArrayFromData(
        array=prod_data,
        coeff_name="PRODUCTION",
        long_name="Production by commodity and industry",
        sets=["COM", "IND"],
        setElDict={
            "COM": ["Agriculture", "Manufacture", "Services"],
            "IND": ["i1", "i2", "i3", "i4", "i5"],
        },
    )

    # CONS: COM × REG
    cons_data = np.array(
        [
            [100.0, 200.0, 300.0, 400.0],
            [150.0, 250.0, 350.0, 450.0],
            [120.0, 220.0, 320.0, 420.0],
        ],
        dtype=np.float32,
    )
    hf["CONS"] = HeaderArrayObj.HeaderArrayFromData(
        array=cons_data,
        coeff_name="CONSUMPTION",
        long_name="Consumption by commodity and region",
        sets=["COM", "REG"],
        setElDict={
            "COM": ["Agriculture", "Manufacture", "Services"],
            "REG": ["USA", "EU", "China", "Japan"],
        },
    )

    # TRAD: COM × REG × REG — duplicate dimension test case
    trade_data = np.random.rand(3, 4, 4).astype(np.float32) * 100
    trade_data[0, 0, 1] = 123.45  # Agriculture: USA → EU
    trade_data[1, 2, 3] = 678.90  # Manufacture: China → Japan
    trade_data[2, 3, 0] = 111.11  # Services: Japan → USA
    hf["TRAD"] = HeaderArrayObj.HeaderArrayFromData(
        array=trade_data,
        coeff_name="TRADE",
        long_name="Bilateral exports from origin to destination region",
        sets=["COM", "REG", "REG"],
        setElDict={
            "COM": ["Agriculture", "Manufacture", "Services"],
            "REG": ["USA", "EU", "China", "Japan"],
        },
    )

    # GDP: scalar
    hf["GDP"] = HeaderArrayObj.HeaderArrayFromData(
        array=np.array(1234.5, dtype=np.float32),
        coeff_name="TOTGDP",
        long_name="Total GDP across all regions",
        sets=None,
        setElDict=None,
    )

    # CNTS: integer COM × REG
    counts_data = np.array(
        [[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]], dtype=np.int32
    )
    hf["CNTS"] = HeaderArrayObj.HeaderArrayFromData(
        array=counts_data,
        coeff_name="COUNTS",
        long_name="Count data by commodity and region",
        sets=["COM", "REG"],
        setElDict={
            "COM": ["Agriculture", "Manufacture", "Services"],
            "REG": ["USA", "EU", "China", "Japan"],
        },
    )

    # LABR: IND × REG
    labor_data = np.random.rand(5, 4).astype(np.float32) * 1000
    labor_data[0, 0] = 999.99
    hf["LABR"] = HeaderArrayObj.HeaderArrayFromData(
        array=labor_data,
        coeff_name="LABOR",
        long_name="Labor employment by industry and region",
        sets=["IND", "REG"],
        setElDict={
            "IND": ["i1", "i2", "i3", "i4", "i5"],
            "REG": ["USA", "EU", "China", "Japan"],
        },
    )

    # UNLD: 2D float32 with no sets → HARPY writes as 2R (unlabeled)
    # Used to test rejection of unlabeled headers (reject_unlabeled=True)
    unld_data = np.arange(12, dtype=np.float32).reshape(3, 4)
    hf["UNLD"] = HeaderArrayObj.HeaderArrayFromData(
        array=unld_data,
        coeff_name="UNLABELED",
        long_name="Unlabeled 2D array for testing 2R rejection",
        sets=None,
        setElDict=None,
    )

    # TRIP: REG × REG × REG — triple duplicate dimension test case
    trip_data = np.random.rand(4, 4, 4).astype(np.float32) * 10
    hf["TRIP"] = HeaderArrayObj.HeaderArrayFromData(
        array=trip_data,
        coeff_name="TRIPLREG",
        long_name="Triple REG dimension for testing __1/__2/__3 naming",
        sets=["REG", "REG", "REG"],
        setElDict={
            "REG": ["USA", "EU", "China", "Japan"],
        },
    )

    # --- Write ---

    print(f"Writing {len(hf.getHeaderArrayNames())} headers to file...")
    hf.writeToDisk(str(output_path))

    print(f"\nSuccess! Created: {output_path}")
    print(f"File size: {output_path.stat().st_size:,} bytes")

    # --- Verify ---

    print("\nVerifying file contents...")
    hf_verify = HarFileObj(str(output_path))
    headers = hf_verify.getHeaderArrayNames()

    print(f"\nHeaders in file ({len(headers)}):")
    for h in headers:
        hao = hf_verify[h]
        print(
            f"  {h:6s}: shape={str(hao.array.shape):20s} "
            f"dtype={str(hao.array.dtype):10s} sets={hao.setNames}"
        )

    print("\nTest file created successfully!")
    return output_path


if __name__ == "__main__":
    create_test_harpyx_har()
