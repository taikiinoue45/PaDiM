import hashlib


def test_fixed_files() -> None:

    checksum_dict = {
        "run.py": "e607a684de133c3d3a1ce8122471b487f3a1865bb84c902b7c94702c8bd8e6d8",
        "tests/test_coverage.py": "a7f3a19bf4f9bc916cae1fe88f7f4201603ce984c46db487ec9bb4e8449248dc",
        "tests/test_fixed_files.py": "a97417cf9060140b244958dae471b97d039d8c8aae6fdbdad6e937378949a672",
    }

    for file_path, fixed_checksum in checksum_dict.items():

        with open(file_path, "rb") as f:
            checksum = hashlib.sha256(f.read()).hexdigest()

        assert fixed_checksum == checksum, f"{file_path} checksum is changed."
