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
        "1aea0b7a9ed1684334746ab70fd543fd33485c8ac27f265afbd59e2d09410639",
        "8b3737677c06afd488972bbabc16cd56362563cc748ade3ec0809b24ce54a59c",
    ]

    for file_path, correct_checksum in zip(file_path_list, checksum_list):

        with open(file_path, "rb") as f:
            checksum = hashlib.sha256(f.read()).hexdigest()

        assert correct_checksum == checksum, f"{file_path} checksum ({checksum}) is incorrect."
