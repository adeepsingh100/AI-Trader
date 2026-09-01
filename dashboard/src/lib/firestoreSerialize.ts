import { Timestamp } from "firebase-admin/firestore";

// Postgres's timestamptz columns always came back as ISO 8601 strings
// (Supabase's PostgREST/JSON layer, then pg's own JSON.stringify);
// Firestore Timestamp objects don't serialize that way — a bare
// NextResponse.json() turns one into {_seconds, _nanoseconds}, and every
// client component does `new Date(field)` expecting a string, which
// silently produces "Invalid Date" against that shape. Every API route
// runs its response through this before returning.
export function serializeTimestamps<T>(data: T): T {
  if (data instanceof Timestamp) return data.toDate().toISOString() as unknown as T;
  if (Array.isArray(data)) return data.map(serializeTimestamps) as unknown as T;
  if (data && typeof data === "object") {
    return Object.fromEntries(
      Object.entries(data as Record<string, unknown>).map(([k, v]) => [k, serializeTimestamps(v)])
    ) as T;
  }
  return data;
}
