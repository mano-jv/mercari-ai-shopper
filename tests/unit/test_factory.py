import pytest
from mercari_shopper.config import load_settings
from mercari_shopper.domain.models import SearchQuery
from mercari_shopper.sources.cache import CachedSource
from mercari_shopper.sources.factory import build_source
from mercari_shopper.sources.fixture_source import FixtureSource


def test_load_settings_rejects_unknown_source(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MERCARI_SOURCE", "ftp")
    with pytest.raises(ValueError, match="ftp"):
        load_settings()


def test_build_source_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="unknown Mercari source"):
        build_source("ftp")


async def test_build_fixture_source() -> None:
    source = build_source("fixture")
    assert isinstance(source, CachedSource)
    result = await source.search(SearchQuery(keyword="switch"))
    await source.aclose()
    assert len(result.items) == 3
    assert isinstance(source._inner, FixtureSource)
