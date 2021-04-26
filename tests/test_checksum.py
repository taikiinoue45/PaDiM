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
        "8006c70f9e8f68fe2a901861b54cbd42644503b707687964fe172f13b7f2d67f",
        "72af9c07b0020181af7d963a9aa558d17529d7e6460e94f9d4bc6806492e7875",
        "a68af6057f4536300d0e0d9eacb7e7d28ac6a7ab72f00189b4991bd309cd8340",
        "402e99ca5a2760ce284afb5e320067b5daef4549feb2d7cb4aa53e623c348620",
    ]

    for file_path, correct_checksum in zip(file_path_list, checksum_list):

        with open(file_path, "rb") as f:
            checksum = hashlib.sha256(f.read()).hexdigest()

        assert correct_checksum == checksum, f"{file_path} checksum ({checksum}) is incorrect."
