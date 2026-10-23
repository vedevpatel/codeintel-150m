import json
import os
import tempfile
from pathlib import Path

from inventory.models import Inventory, Item


class JsonStore:
    """Persist an Inventory as a JSON file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> Inventory:
        inventory = Inventory()
        if not self.path.exists():
            return inventory
        for row in json.loads(self.path.read_text()):
            inventory.add(Item(**row))
        return inventory

    def save(self, inventory: Inventory) -> None:
        """Write atomically: dump to a temp file in the same directory, then rename over the target."""
        rows = [vars(item) for item in inventory._items.values()]
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            json.dump(rows, f, indent=2)
        os.replace(tmp, self.path)
