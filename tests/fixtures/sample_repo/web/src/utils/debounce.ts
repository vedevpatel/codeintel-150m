/** Delay calls to `fn` until `ms` milliseconds have passed without another call. */
export function debounce<T extends (...args: any[]) => void>(fn: T, ms = 250) {
  let timer: ReturnType<typeof setTimeout> | undefined;
  return (...args: Parameters<T>) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), ms);
  };
}

/** Invoke `fn` at most once every `ms` milliseconds. */
export const throttle = <T extends (...args: any[]) => void>(fn: T, ms = 250) => {
  let last = 0;
  return (...args: Parameters<T>) => {
    const now = Date.now();
    if (now - last >= ms) {
      last = now;
      fn(...args);
    }
  };
};
