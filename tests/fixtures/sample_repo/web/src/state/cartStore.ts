import type { ItemDto } from "../api/client";

export interface CartLine {
  item: ItemDto;
  quantity: number;
}

export type CartAction =
  | { type: "add"; item: ItemDto; quantity: number }
  | { type: "remove"; sku: string }
  | { type: "clear" };

export function cartReducer(state: CartLine[], action: CartAction): CartLine[] {
  switch (action.type) {
    case "add": {
      const existing = state.find((l) => l.item.sku === action.item.sku);
      if (existing) {
        return state.map((l) => (l === existing ? { ...l, quantity: l.quantity + action.quantity } : l));
      }
      return [...state, { item: action.item, quantity: action.quantity }];
    }
    case "remove":
      return state.filter((l) => l.item.sku !== action.sku);
    case "clear":
      return [];
  }
}

export function cartTotalCents(lines: CartLine[]): number {
  return lines.reduce((sum, l) => sum + l.item.unitPriceCents * l.quantity, 0);
}
