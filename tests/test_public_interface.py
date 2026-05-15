import choicekit


def test_top_level_exports_are_explicit() -> None:
    assert choicekit.__all__ == ["__version__"]


def test_top_level_exports_exist() -> None:
    for name in choicekit.__all__:
        assert hasattr(choicekit, name)


def test_metadata_helpers_are_not_public_exports() -> None:
    assert not hasattr(choicekit, "version")
    assert not hasattr(choicekit, "PackageNotFoundError")
