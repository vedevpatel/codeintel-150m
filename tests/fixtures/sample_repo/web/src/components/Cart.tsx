import { useReducer } from "react";
import { cartReducer, cartTotalCents } from "../state/cartStore";

function formatCents(cents: number): string {
  return `$${(cents / 100).toFixed(2)}`;
}

export default function Cart() {
  const [lines, dispatch] = useReducer(cartReducer, []);
  return (
    <section>
      <ul>
        {lines.map((l) => (
          <li key={l.item.sku}>
            {l.item.name} x {l.quantity}
            <button onClick={() => dispatch({ type: "remove", sku: l.item.sku })}>Remove</button>
          </li>
        ))}
      </ul>
      <p>Total: {formatCents(cartTotalCents(lines))}</p>
      <button onClick={() => dispatch({ type: "clear" })}>Empty cart</button>
    </section>
  );
}
