import { createClient } from "@supabase/supabase-js";

// Client-side only: the browser talks to Supabase directly with the
// anon key. Reads are open (RLS "public read" policy); the only write
// this app makes is capital_config, gated to signed-in users by RLS.
// See PROJECT_SPEC.md §8 and src/db/migrations/0002_rls.sql.
export const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
);
