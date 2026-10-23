"""Price calculations. All amounts are integer cents."""

TAX_RATES = {"CA": 0.0725, "NY": 0.04, "TX": 0.0625}


def apply_discount(price_cents: int, percent: float) -> int:
    """Reduce a price by a percentage discount, rounding to the nearest cent."""
    if not 0 <= percent <= 100:
        raise ValueError("discount percent must be between 0 and 100")
    return round(price_cents * (1 - percent / 100))


def sales_tax(price_cents: int, state: str) -> int:
    """Sales tax owed for a purchase shipped to a US state."""
    return round(price_cents * TAX_RATES.get(state.upper(), 0.0))


def bulk_price(unit_price_cents: int, quantity: int) -> int:
    """Total for a quantity of units with tiered volume discounts."""
    if quantity >= 100:
        rate = 15
    elif quantity >= 20:
        rate = 5
    else:
        rate = 0
    return apply_discount(unit_price_cents * quantity, rate)
