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
        "d78e95e5408c39d33753118d5b2940713f6cea8bbfbd9cfece3f6926c5dd6818",
        "a6569dc571b2c8e21ebd1d0b46de5e26ac02c86fb51d1599417df4731977b816",
    ]

    for file_path, correct_checksum in zip(file_path_list, checksum_list):

        with open(file_path, "rb") as f:
            checksum = hashlib.sha256(f.read()).hexdigest()

        assert correct_checksum == checksum, f"{file_path} checksum ({checksum}) is incorrect."
