import { cookies } from "next/headers";
import { NextRequest, NextResponse } from "next/server";
import { checkPassword, COOKIE_NAME, isValidSession, sessionToken } from "@/lib/configAuth";

export async function GET() {
  const cookieStore = await cookies();
  return NextResponse.json({ authed: isValidSession(cookieStore.get(COOKIE_NAME)?.value) });
}

export async function POST(request: NextRequest) {
  const { password } = await request.json().catch(() => ({ password: "" }));
  if (typeof password !== "string" || !checkPassword(password)) {
    return NextResponse.json({ error: "incorrect password" }, { status: 401 });
  }

  const cookieStore = await cookies();
  cookieStore.set(COOKIE_NAME, sessionToken(), {
    httpOnly: true,
    secure: true,
    sameSite: "lax",
    path: "/",
    maxAge: 60 * 60 * 24 * 30, // 30 days
  });
  return NextResponse.json({ ok: true });
}

export async function DELETE() {
  const cookieStore = await cookies();
  cookieStore.delete(COOKIE_NAME);
  return NextResponse.json({ ok: true });
}
