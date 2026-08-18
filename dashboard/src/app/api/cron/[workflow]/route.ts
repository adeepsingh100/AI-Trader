import { NextRequest, NextResponse } from "next/server";

// Plain GET-and-go trigger for an external cron pinger (GH Actions'
// own `schedule:` is unreliable below ~15min cadence). PAT stays
// server-side; the only thing a caller needs is this URL + ?key=.
const ALLOWED_WORKFLOWS = new Set(["trading_cycle.yml", "risk_check.yml"]);
const REPO = "adeepsingh100/AI-Trader";

export async function GET(request: NextRequest, { params }: { params: Promise<{ workflow: string }> }) {
  const { workflow } = await params;
  if (!ALLOWED_WORKFLOWS.has(workflow)) {
    return NextResponse.json({ error: "unknown workflow" }, { status: 404 });
  }
  const providedKey = request.nextUrl.searchParams.get("key");
  const expectedKey = process.env.CRON_TRIGGER_SECRET;
  if (providedKey !== expectedKey) {
    // Temporary diagnostic — lengths only, never the actual secret values.
    return NextResponse.json(
      {
        error: "unauthorized",
        debug: {
          providedKeyLength: providedKey?.length ?? null,
          expectedKeySet: expectedKey != null,
          expectedKeyLength: expectedKey?.length ?? null,
        },
      },
      { status: 401 }
    );
  }

  const res = await fetch(`https://api.github.com/repos/${REPO}/actions/workflows/${workflow}/dispatches`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${process.env.GITHUB_DISPATCH_PAT}`,
      Accept: "application/vnd.github+json",
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ ref: "main" }),
  });

  if (!res.ok) {
    return NextResponse.json({ error: await res.text() }, { status: res.status });
  }
  return NextResponse.json({ ok: true });
}
