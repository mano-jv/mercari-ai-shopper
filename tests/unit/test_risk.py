from mercari_shopper.domain.models import Listing, SellerInfo
from mercari_shopper.domain.risk import (
    PhraseRiskDetector,
    SellerRiskDetector,
    default_risk_detector,
)


def test_phrase_flags_on_japanese_warnings() -> None:
    detector = PhraseRiskDetector()
    text = "動作未確認です。箱なし、傷あり。訳ありのためジャンク扱い。部品取り可。"
    codes = {flag.code for flag in detector.scan(text)}
    assert codes == {"untested", "no_box", "scratched", "flawed", "junk", "for_parts"}


def test_condition_label_is_not_scratched() -> None:
    detector = PhraseRiskDetector()
    assert detector.scan("やや傷や汚れあり") == ()
    assert detector.scan("傷なし、箱あり") == ()
    assert detector.scan("傷ありません。訳ありません。箱なしではありません。") == ()
    assert {flag.code for flag in detector.scan("傷あり、箱なし")} == {"scratched", "no_box"}


def test_listing_detector_reads_name_and_description() -> None:
    listing = Listing(
        item_id="m1",
        name="Switch 箱なし",
        price=20000,
        thumbnail_url="http://img",
        description="動作未確認",
    )
    flags = default_risk_detector().detect(listing)
    codes = {flag.code for flag in flags}
    assert "no_box" in codes
    assert "untested" in codes


def test_new_and_low_rated_sellers() -> None:
    detector = SellerRiskDetector()
    new = detector.inspect(SellerInfo(seller_id="1", num_ratings=2, star_rating=5.0))
    assert [flag.code for flag in new] == ["new_seller"]
    low = detector.inspect(SellerInfo(seller_id="2", num_ratings=20, star_rating=2.1, bad=1))
    assert "low_rated_seller" in {flag.code for flag in low}
    poor = detector.inspect(SellerInfo(seller_id="3", num_ratings=20, star_rating=4.0, bad=8))
    assert "poor_feedback" in {flag.code for flag in poor}
    trusted = detector.inspect(SellerInfo(seller_id="4", num_ratings=50, star_rating=4.8, bad=1))
    assert trusted == ()
    assert detector.inspect(None) == ()
