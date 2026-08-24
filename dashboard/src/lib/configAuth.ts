import { createHmac, timingSafeEqual } from "crypto";

// Minimal shared-secret gate for the one write path (/config page's
// capital_config edits) — replaces Supabase Auth, which had no real user
// account configured (the sign-in form always showed, dormant). Single
// operator, no per-user identity/audit needed; stateless (no session
// store) — the cookie IS the proof: an HMAC of a fixed string keyed by
// CONFIG_EDIT_PASSWORD, so validating it is just recomputing and
// comparing, no server-side state at all.
export const COOKIE_NAME = "config_session";

function expectedToken(): string {
  return createHmac("sha256", process.env.CONFIG_EDIT_PASSWORD ?? "").update("authenticated").digest("hex");
}

function timingSafeStringEqual(a: string, b: string): boolean {
  const bufA = Buffer.from(a);
  const bufB = Buffer.from(b);
  return bufA.length === bufB.length && timingSafeEqual(bufA, bufB);
}

export function checkPassword(input: string): boolean {
  const expected = process.env.CONFIG_EDIT_PASSWORD ?? "";
  return expected.length > 0 && timingSafeStringEqual(input, expected);
}

export function sessionToken(): string {
  return expectedToken();
}

export function isValidSession(token: string | undefined): boolean {
  if (!token) return false;
  return timingSafeStringEqual(token, expectedToken());
}
