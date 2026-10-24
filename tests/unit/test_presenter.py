import io

from mercari_shopper.agent.loop import AgentResult, StepTrace
from mercari_shopper.domain.models import Recommendation
from mercari_shopper.presentation.render import (
    JsonPresenter,
    PlainPresenter,
    RichPresenter,
    build_presenter,
)
from mercari_shopper.presentation.view import ResultViewBuilder, format_yen


def _result(*, verbose_trace: bool = True) -> AgentResult:
    return AgentResult(
        summary="Three OLED Switch listings under 30000 yen.",
        recommendations=(
            Recommendation(
                item_id="m10000000001",
                rank=1,
                reason="Best condition for the price",
                pros=["box included"],
                cons=["not new"],
            ),
            Recommendation(
                item_id="m10000000002",
                rank=2,
                reason="Lowest bid so far",
                pros=["under budget"],
                cons=["auction"],
            ),
            Recommendation(
                item_id="shopItem001abc",
                rank=3,
                reason="Like-new shop listing",
                pros=["condition"],
                cons=["pricier"],
            ),
        ),
        listings=(
            {
                "item_id": "m10000000001",
                "name": "Nintendo Switch 有機EL 本体",
                "price": 24800,
                "condition_id": 3,
                "condition": "no noticeable scratches",
                "url": "https://jp.mercari.com/item/m10000000001",
                "seller": {"name": "sample-seller", "star_rating": 5.0, "num_ratings": 122},
            },
            {
                "item_id": "m10000000002",
                "name": "Nintendo Switch OLED auction sample",
                "price": 22000,
                "condition_id": 4,
                "condition": "some scratches or dirt",
                "url": "https://jp.mercari.com/item/m10000000002",
                "auction": {"total_bids": 5},
                "seller": {"name": "auction-seller", "star_rating": 4.0, "num_ratings": 12},
            },
            {
                "item_id": "shopItem001abc",
                "name": "Switch 有機EL shop listing",
                "price": 26800,
                "condition_id": 2,
                "condition": "like new",
                "url": "https://jp.mercari.com/shops/product/shopItem001abc",
                "is_shop_item": True,
            },
        ),
        steps=3,
        tool_calls=2,
        repairs=0,
        elapsed_s=1.23,
        trace=(
            StepTrace(1, 0.4, ("search_mercari",)),
            StepTrace(2, 0.31, ("get_item_details",)),
            StepTrace(3, 0.52, ("submit_recommendations",)),
        )
        if verbose_trace
        else (),
    )


EXPECTED_PLAIN = """\
Three OLED Switch listings under 30000 yen.

1. Nintendo Switch 有機EL 本体
   ¥24,800  ·  no noticeable scratches
   https://jp.mercari.com/item/m10000000001
   Best condition for the price
   + box included
   - not new
   seller: sample-seller ★5 (122)

2. Nintendo Switch OLED auction sample  (auction)
   ¥22,000  ·  some scratches or dirt
   https://jp.mercari.com/item/m10000000002
   Lowest bid so far
   + under budget
   - auction
   seller: auction-seller ★4 (12)

3. Switch 有機EL shop listing  (shop)
   ¥26,800  ·  like new
   https://jp.mercari.com/shops/product/shopItem001abc
   Like-new shop listing
   + condition
   - pricier
"""

EXPECTED_VERBOSE_TAIL = """\
---
steps: 3  tools: 2  repairs: 0  1.23s
1. search_mercari (0.40s)
2. get_item_details (0.31s)
3. submit_recommendations (0.52s)
"""


def test_format_yen() -> None:
    assert format_yen(24800) == "¥24,800"
    assert format_yen(None) == "¥?"


def test_plain_presenter_snapshot() -> None:
    text = PlainPresenter().render(_result())
    assert text == EXPECTED_PLAIN


def test_plain_presenter_verbose_appends_trace() -> None:
    text = PlainPresenter(verbose=True).render(_result())
    assert text.startswith(EXPECTED_PLAIN.rstrip())
    assert text.endswith(EXPECTED_VERBOSE_TAIL)


def test_json_presenter_snapshot_fields() -> None:
    payload = JsonPresenter().render(_result())
    assert '"item_id": "m10000000001"' in payload
    assert '"price": 24800' in payload
    assert '"star_rating": 5.0' in payload
    assert '"is_auction": true' in payload
    assert '"is_shop_item": true' in payload
    assert '"trace"' not in payload
    verbose = JsonPresenter(verbose=True).render(_result())
    assert '"trace"' in verbose
    assert "search_mercari" in verbose


def test_rich_presenter_render_matches_plain() -> None:
    result = _result()
    assert RichPresenter().render(result) == PlainPresenter().render(result)


def test_rich_presenter_emit_writes_cards() -> None:
    buf = io.StringIO()
    RichPresenter().emit(_result(), buf)
    text = buf.getvalue()
    assert "Nintendo Switch 有機EL 本体" in text
    assert "¥24,800" in text
    assert "sample-seller" in text


def test_view_builder_fills_missing_listing() -> None:
    result = AgentResult(
        summary="",
        recommendations=(Recommendation(item_id="m9", rank=1, reason="only id"),),
        listings=(),
        steps=1,
        tool_calls=0,
        repairs=0,
    )
    view = ResultViewBuilder().build(result)
    assert view.items[0].name == "m9"
    assert view.items[0].price is None
    assert view.items[0].condition == "unknown"


def test_presenter_factory_json_wins() -> None:
    presenter = build_presenter(json_output=True, plain=True, tty=True)
    assert isinstance(presenter, JsonPresenter)


def test_presenter_factory_plain_when_not_tty() -> None:
    presenter = build_presenter(tty=False)
    assert isinstance(presenter, PlainPresenter)


def test_presenter_factory_rich_on_tty() -> None:
    presenter = build_presenter(tty=True)
    assert isinstance(presenter, RichPresenter)


def test_json_error_payload() -> None:
    text = JsonPresenter().render_error(RuntimeError("boom"))
    assert '"error": "boom"' in text
    assert '"type": "RuntimeError"' in text
