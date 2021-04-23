import hashlib


def test_checksum() -> None:

    file_path_list = [
        ".github/workflows/format-python.yml",
        ".github/workflows/format-yaml.yml",
        ".github/workflows/merge.yml",
        ".github/workflows/pull-request.yml",
        "run.py",
        "tests/test_coverage.py",
    ]

    checksum_list = [
        "981bd32ad4febe14a44c1c7a0a00245c747373f39d6e22884f80f018bebc68c8",
        "e18032c3c704dc1a391d72cf8bded1c98b0dffcc80d00e84f5a78daf53e4cc0f",
        "e6b63c3aa8412405653d15005b7dd6344756aa886be8d7cc75dff9ccca11c86a",
        "830e762a619f09597077dbab43e2f7ba14d9f6359104bc714a5360f2a1fe3141",
        "e607a684de133c3d3a1ce8122471b487f3a1865bb84c902b7c94702c8bd8e6d8",
        "a7f3a19bf4f9bc916cae1fe88f7f4201603ce984c46db487ec9bb4e8449248dc",
    ]

    for file_path, fixed_checksum in zip(file_path_list, checksum_list):

        with open(file_path, "rb") as f:
            checksum = hashlib.sha256(f.read()).hexdigest()

        assert fixed_checksum == checksum, f"{file_path} checksum is incorrect."
