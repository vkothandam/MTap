import pytest

from mtap_space.synthetic import TICKERS, sessions, synthetic_bars, utc_ms, write_bars  # noqa: F401


@pytest.fixture(scope="session")
def bars_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("bars")
    return write_bars(root, synthetic_bars(sessions()))


@pytest.fixture(scope="session")
def lib(bars_root):
    from mtap_space import library
    built = library.build(bars_root, sessions=25, min_dollar=1e6, log=lambda *_: None)
    return library.Library.from_memory(built)
