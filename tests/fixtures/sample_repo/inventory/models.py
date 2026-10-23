from __future__ import annotations

from dataclasses import dataclass, field


class OutOfStockError(Exception):
    """Raised when removing more units than are available."""


@dataclass
class Item:
    sku: str
    name: str
    unit_price_cents: int
    quantity: int = 0
    tags: list[str] = field(default_factory=list)

    def is_available(self) -> bool:
        return self.quantity > 0


class Inventory:
    """In-memory collection of items keyed by SKU."""

    def __init__(self) -> None:
        self._items: dict[str, Item] = {}

    def add(self, item: Item) -> None:
        """Insert a new item or merge its quantity into an existing SKU."""
        existing = self._items.get(item.sku)
        if existing is None:
            self._items[item.sku] = item
        else:
            existing.quantity += item.quantity

    def remove(self, sku: str, quantity: int) -> Item:
        """Take units of an item out of stock, raising if there are not enough."""
        item = self._items[sku]
        if quantity > item.quantity:
            raise OutOfStockError(f"only {item.quantity} units of {sku} left")
        item.quantity -= quantity
        return item

    def restock(self, sku: str, quantity: int) -> None:
        self._items[sku].quantity += quantity

    def low_stock(self, threshold: int = 5) -> list[Item]:
        """Items whose quantity is at or below the reorder threshold."""
        return sorted((i for i in self._items.values() if i.quantity <= threshold), key=lambda i: i.quantity)
