from inventory.pricing import apply_discount, bulk_price


def test_apply_discount_rounds():
    assert apply_discount(999, 10) == 899


def test_bulk_price_tiers():
    assert bulk_price(100, 20) == 1900
