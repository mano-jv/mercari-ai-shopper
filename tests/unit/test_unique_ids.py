from mercari_shopper.sources.base import unique_ids


def test_unique_ids_drops_blanks_and_duplicates() -> None:
    assert unique_ids(["m1", "", "m2", "m1", "m3"]) == ["m1", "m2", "m3"]
    assert unique_ids([]) == []
