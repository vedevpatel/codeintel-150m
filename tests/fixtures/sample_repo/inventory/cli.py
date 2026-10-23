import argparse

from inventory.pricing import bulk_price
from inventory.storage import JsonStore


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="inventory admin")
    parser.add_argument("store")
    parser.add_argument("--report-low-stock", action="store_true")
    args = parser.parse_args(argv)
    inventory = JsonStore(args.store).load()
    if args.report_low_stock:
        for item in inventory.low_stock():
            print(f"{item.sku}\t{item.quantity}\t{bulk_price(item.unit_price_cents, 20)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
