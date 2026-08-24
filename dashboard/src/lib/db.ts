import { Pool, types } from "pg";
import { attachDatabasePool } from "@vercel/functions";

// pg defaults NUMERIC (oid 1700) and INT8/bigint (oid 20) to strings, to
// avoid silent precision loss for values too large for a JS number. Every
// value this app reads (trade prices/qtys/pnl, bigserial ids) fits safely
// in a JS number, and every API route below returns them as plain numbers
// (Supabase's PostgREST/JSON layer always did) — parse both back so no
// client component needs numeric-string handling.
types.setTypeParser(1700, parseFloat);
types.setTypeParser(20, (v) => parseInt(v, 10));
// `date` (oid 1082) defaults to a parsed JS Date; daily_pnl.date is used
// today as a plain "YYYY-MM-DD" string (chart x-axis labels) — keep
// Postgres's raw text as-is instead.
types.setTypeParser(1082, (v) => v);

// Server-side only — never imported by a "use client" file. Reads are
// open (no RLS/anon-key layer exists on Neon); the API routes below are
// the only path in, same effective public-read posture Supabase's RLS
// "public read" policy + a shipped-in-the-JS-bundle anon key already had.
const pool = new Pool({ connectionString: process.env.DATABASE_URL });
attachDatabasePool(pool);

export { pool };
