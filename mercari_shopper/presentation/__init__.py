"""CLI presentation: view models, presenter strategies, interactive session."""

from mercari_shopper.presentation.interactive import InteractiveSession, Stdio, UserIO
from mercari_shopper.presentation.render import (
    JsonPresenter,
    PlainPresenter,
    Presenter,
    PresenterFactory,
    RichPresenter,
    build_presenter,
)
from mercari_shopper.presentation.view import (
    RecommendationView,
    ResultView,
    ResultViewBuilder,
    SellerView,
    format_yen,
)

__all__ = [
    "InteractiveSession",
    "JsonPresenter",
    "PlainPresenter",
    "Presenter",
    "PresenterFactory",
    "RecommendationView",
    "ResultView",
    "ResultViewBuilder",
    "RichPresenter",
    "SellerView",
    "Stdio",
    "UserIO",
    "build_presenter",
    "format_yen",
]
