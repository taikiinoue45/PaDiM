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
        "a619c475d0ae8dfd4d41764ff79cd0e78fd86acbc50c8dfdb372908506995483",
        "981bd32ad4febe14a44c1c7a0a00245c747373f39d6e22884f80f018bebc68c8",
        "e18032c3c704dc1a391d72cf8bded1c98b0dffcc80d00e84f5a78daf53e4cc0f",
        "0e8ede1867756b4ce214a05d21e8b0448f742e90f0e8d29c968f905a20e57085",
        "9d3f18de6c799ab9d11fa9030ab7e32e4d0325b13e5437f5b3ed09113246b5a3",
    ]

    for file_path, correct_checksum in zip(file_path_list, checksum_list):

        with open(file_path, "rb") as f:
            checksum = hashlib.sha256(f.read()).hexdigest()

        assert correct_checksum == checksum, f"{file_path} checksum {checksum} is incorrect."
