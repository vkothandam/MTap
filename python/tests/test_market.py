"""sourcing-py market: SPY daily/minute bars and dividends from Massive, SPX from a file. Offline."""

from __future__ import annotations

from datetime import date, datetime

import httpx
import pyarrow.parquet as pq

from sourcing_py import market

NY = market.NY


def _ms(s: str) -> int:
    return int(datetime.fromisoformat(s).replace(tzinfo=NY).timestamp() * 1000)


def _handler(req: httpx.Request) -> httpx.Response:
    assert req.url.params["apiKey"] == "K"
    p = req.url.path
    if "/range/1/day/" in p:
        return httpx.Response(200, json={"results": [
            {"t": _ms("2026-09-17T00:00"), "o": 660, "h": 662, "l": 659, "c": 661, "v": 5e7, "vw": 660.5, "n": 9},
            {"t": _ms("2026-09-18T00:00"), "o": 659, "h": 663, "l": 658, "c": 662, "v": 6e7, "vw": 661, "n": 8}]})
    if "/range/1/minute/" in p and "cursor" not in req.url.params:
        return httpx.Response(200, json={"results": [{"t": _ms("2026-09-18T09:30"), "o": 659, "h": 659.5, "l": 658.9,
                                                      "c": 659.2, "v": 1e6, "vw": 659.1, "n": 900}],
                                         "next_url": f"https://x{p}?cursor=abc&apiKey=OLD"})
    if "/range/1/minute/" in p:
        return httpx.Response(200, json={"results": [{"t": _ms("2026-09-18T09:31"), "o": 659.2, "h": 659.3,
                                                      "l": 659.0, "c": 659.1, "v": 5e5, "vw": 659.15, "n": 400}]})
    if p.endswith("/v3/reference/dividends"):
        return httpx.Response(200, json={"results": [
            {"ex_dividend_date": "2026-06-18", "cash_amount": 1.90, "pay_date": "2026-07-31", "dividend_type": "CD"},
            {"ex_dividend_date": "2026-09-18", "cash_amount": 1.888834, "dividend_type": "CD"}]})
    return httpx.Response(404)


def test_spy_tables_page_and_map_fields():
    with httpx.Client(transport=httpx.MockTransport(_handler)) as client:
        t = market.spy_tables(client, {"base_url": "https://x", "api_key": "K"}, date(2026, 9, 17), date(2026, 9, 18))
    assert t["spy_daily"].column("date").to_pylist() == [date(2026, 9, 17), date(2026, 9, 18)]
    assert t["spy_1m"].num_rows == 2 and t["spy_1m"].column("n").to_pylist() == [900, 400]
    d = t["spy_dividends"].to_pylist()
    assert (d[1]["ex_date"], d[1]["cash"], d[1]["pay_date"]) == (date(2026, 9, 18), 1.888834, None)


def test_spx_table_and_write(tmp_path):
    f = tmp_path / "spx.csv"
    f.write_text("DT,Open,High,Low,Close,Volume\n2026-11-04,7570,7700,7560,7688.62,0\n2026-11-05,7680,7690,7340,7353.73,0\n")
    t = market.spx_table(f)
    assert t.column_names == ["date", "o", "h", "l", "c"] and t.column("c").to_pylist() == [7688.62, 7353.73]
    market.write({"spx_daily": t}, tmp_path / "market")
    assert pq.read_table(tmp_path / "market" / "spx_daily.parquet").equals(t)
    assert [p.name for p in (tmp_path / "market").iterdir()] == ["spx_daily.parquet"]


def test_the_default_spx_file_is_there():
    t = market.spx_table(market.SPX_CSV)
    assert t.column("date")[0].as_py() == date(2005, 3, 1) and t.num_rows > 5000


def test_cli_rejects_bad_args(capsys):
    assert market.main(["fetch", "--fromdate", "2026-11-06", "--todate", "2026-11-01"]) == 2
    assert market.main(["fetch", "--fromdate", "2026-11-01", "--todate", "2026-11-06", "--x", "1"]) == 2
    assert market.main([]) == 2
