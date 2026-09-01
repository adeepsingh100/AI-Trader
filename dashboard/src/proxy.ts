import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";
import { adminAuth } from "@/lib/firebaseAdmin";
import { COOKIE_NAME } from "@/lib/session";

// Whole-dashboard gate (replaces the old per-page CONFIG_EDIT_PASSWORD
// check, dashboard/src/lib/configAuth.ts — deleted). Runs before every
// route the matcher below doesn't exclude. Node.js runtime is the
// default for proxy in this Next.js version (16) — no `runtime` export
// here, setting one throws.
export default async function proxy(request: NextRequest) {
  const sessionCookie = request.cookies.get(COOKIE_NAME)?.value;
  if (sessionCookie) {
    try {
      await adminAuth.verifySessionCookie(sessionCookie, true);
      return NextResponse.next();
    } catch {
      // fall through to redirect — expired/revoked/invalid cookie
    }
  }

  const signInUrl = new URL("/sign-in", request.url);
  return NextResponse.redirect(signInUrl);
}

export const config = {
  matcher: [
    /*
     * Everything except:
     * - /sign-in (the page itself, or it'd redirect-loop)
     * - /api/session (the sign-in/sign-out exchange itself)
     * - /api/cron (GitHub Actions trigger, gated by its own ?key= secret,
     *   not a Firebase session — see api/cron/[workflow]/route.ts)
     * - Next.js internals and static assets
     */
    "/((?!sign-in|api/session|api/cron|_next/static|_next/image|favicon.ico).*)",
  ],
};
