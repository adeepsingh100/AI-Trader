from src.coindcx_client import top_inr_pairs_by_turnover


def test_top_inr_pairs_by_turnover_filters_and_ranks():
    ticker = [
        {"market": "BTCINR", "volume": "10", "last_price": "100"},   # turnover 1000
        {"market": "ETHINR", "volume": "5", "last_price": "500"},    # turnover 2500
        {"market": "BTCUSDT", "volume": "999", "last_price": "999"}, # not INR, excluded
        {"market": "XRPINR", "volume": "1", "last_price": "1"},      # turnover 1
    ]

    top = top_inr_pairs_by_turnover(n=2, ticker=ticker)

    assert [t["market"] for t in top] == ["ETHINR", "BTCINR"]
    assert top[0]["turnover_inr"] == 2500.0
