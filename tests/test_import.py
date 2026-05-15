from importlib.metadata import version

import choicekit


def test_package_imports() -> None:
    assert choicekit.__version__ == version("choicekit")
