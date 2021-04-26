import hashlib


def test_checksum() -> None:

    file_path_list = [
        "run.py",
        "tests/test_coverage.py",
        ".github/workflows/format-python.yml",
        ".github/workflows/format-yaml.yml",
        ".github/workflows/merge.yml",
        ".github/workflows/pull-request.yml",
    ]

    checksum_list = [
        "04e248ab755666004e01a596ccdd9729e70609b9a9944a613e6dff5909e9ab81",
        "a7f3a19bf4f9bc916cae1fe88f7f4201603ce984c46db487ec9bb4e8449248dc",
        "981bd32ad4febe14a44c1c7a0a00245c747373f39d6e22884f80f018bebc68c8",
        "e18032c3c704dc1a391d72cf8bded1c98b0dffcc80d00e84f5a78daf53e4cc0f",
        "90c32add3f2200a217b6b080124ab0b23f4beaf807c1ed0e23e7be6dea91c828",
        "e5303ca8521b0ff704ea0337c2c3b897ceda3af5eb725ac5403c98bfabf71fe0",
    ]

    for file_path, correct_checksum in zip(file_path_list, checksum_list):

        with open(file_path, "rb") as f:
            checksum = hashlib.sha256(f.read()).hexdigest()

        assert correct_checksum == checksum, f"{file_path} checksum ({checksum}) is incorrect."
