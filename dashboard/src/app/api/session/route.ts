import { cookies } from "next/headers";
import { NextRequest, NextResponse } from "next/server";
import { adminAuth } from "@/lib/firebaseAdmin";
import { COOKIE_NAME } from "@/lib/session";

// Replaces the old CONFIG_EDIT_PASSWORD cookie gate (dashboard/src/lib/
// configAuth.ts, now deleted) — a session cookie, not per-request ID-token
// verification, since proxy.ts needs to gate every page navigation, not
// just fetch calls that can attach an Authorization header.
const EXPIRES_IN_MS = 60 * 60 * 24 * 14 * 1000; // 14 days — Firebase's max for session cookies

export async function POST(request: NextRequest) {
  const { idToken } = await request.json().catch(() => ({ idToken: "" }));
  if (typeof idToken !== "string" || !idToken) {
    return NextResponse.json({ error: "missing idToken" }, { status: 400 });
  }

  try {
    const sessionCookie = await adminAuth.createSessionCookie(idToken, { expiresIn: EXPIRES_IN_MS });
    const cookieStore = await cookies();
    cookieStore.set(COOKIE_NAME, sessionCookie, {
      httpOnly: true,
      secure: true,
      sameSite: "lax",
      path: "/",
      maxAge: EXPIRES_IN_MS / 1000,
    });
    return NextResponse.json({ ok: true });
  } catch (e) {
    return NextResponse.json({ error: e instanceof Error ? e.message : String(e) }, { status: 401 });
  }
}

export async function DELETE() {
  const cookieStore = await cookies();
  cookieStore.delete(COOKIE_NAME);
  return NextResponse.json({ ok: true });
}
