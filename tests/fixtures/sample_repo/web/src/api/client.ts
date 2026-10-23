export interface ItemDto {
  sku: string;
  name: string;
  unitPriceCents: number;
  quantity: number;
}

/** Fetch a URL, retrying network errors and 5xx responses with exponential backoff. */
export async function fetchWithRetry(url: string, init: RequestInit = {}, retries = 3): Promise<Response> {
  let delay = 200;
  for (let attempt = 0; ; attempt++) {
    try {
      const res = await fetch(url, init);
      if (res.status < 500 || attempt >= retries) return res;
    } catch (err) {
      if (attempt >= retries) throw err;
    }
    await new Promise((resolve) => setTimeout(resolve, delay));
    delay *= 2;
  }
}

export class ApiClient {
  constructor(private readonly baseUrl: string, private readonly token?: string) {}

  private headers(): HeadersInit {
    return this.token ? { Authorization: `Bearer ${this.token}` } : {};
  }

  /** Load a single item by SKU. */
  async getItem(sku: string): Promise<ItemDto> {
    const res = await fetchWithRetry(`${this.baseUrl}/items/${encodeURIComponent(sku)}`, { headers: this.headers() });
    if (!res.ok) throw new Error(`item ${sku} not found`);
    return res.json();
  }

  async listItems(page = 1): Promise<ItemDto[]> {
    const res = await fetchWithRetry(`${this.baseUrl}/items?page=${page}`, { headers: this.headers() });
    return res.json();
  }
}
