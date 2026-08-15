"use client";

import { useEffect, useState } from "react";
import type { Session } from "@supabase/supabase-js";
import { supabase } from "@/lib/supabase";
import { useMode, ModeToggle } from "@/components/ModeToggle";
import { STATUS } from "@/lib/palette";
import type { CapitalConfig } from "@/lib/types";

export default function ConfigClient() {
  const [session, setSession] = useState<Session | null | undefined>(undefined);

  useEffect(() => {
    supabase.auth.getSession().then(({ data }) => setSession(data.session));
    const { data: sub } = supabase.auth.onAuthStateChange((_event, s) => setSession(s));
    return () => sub.subscription.unsubscribe();
  }, []);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold tracking-tight">Config</h1>
        {session && (
          <button
            onClick={() => supabase.auth.signOut()}
            className="text-sm hover:underline"
            style={{ color: "var(--text-muted)" }}
          >
            Sign out
          </button>
        )}
      </div>

      {session === undefined && <p style={{ color: "var(--text-muted)" }}>Loading…</p>}
      {session === null && <LoginForm />}
      {session && <ConfigForm />}
    </div>
  );
}

function LoginForm() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setSubmitting(true);
    setError(null);
    const { error } = await supabase.auth.signInWithPassword({ email, password });
    setSubmitting(false);
    if (error) setError(error.message);
  }

  return (
    <form
      onSubmit={handleSubmit}
      className="max-w-sm space-y-3 rounded-xl p-5 shadow-sm"
      style={{ background: "var(--surface-1)", border: "1px solid var(--border)" }}
    >
      <p className="text-sm" style={{ color: "var(--text-muted)" }}>
        Sign in to edit capital, target, and loss limits.
      </p>
      <input
        type="email"
        placeholder="Email"
        value={email}
        onChange={(e) => setEmail(e.target.value)}
        className="w-full rounded-md px-3 py-1.5 text-sm outline-none focus:ring-2 focus:ring-neutral-900/20"
        style={{ border: "1px solid var(--border)" }}
        required
      />
      <input
        type="password"
        placeholder="Password"
        value={password}
        onChange={(e) => setPassword(e.target.value)}
        className="w-full rounded-md px-3 py-1.5 text-sm outline-none focus:ring-2 focus:ring-neutral-900/20"
        style={{ border: "1px solid var(--border)" }}
        required
      />
      {error && (
        <p className="text-sm" style={{ color: "var(--status-critical)" }}>
          {error}
        </p>
      )}
      <button
        type="submit"
        disabled={submitting}
        className="rounded-md bg-neutral-900 text-white text-sm px-3 py-1.5 disabled:opacity-50 hover:bg-neutral-800 transition-colors"
      >
        {submitting ? "Signing in…" : "Sign in"}
      </button>
    </form>
  );
}

const FIELDS: { key: keyof CapitalConfig; label: string }[] = [
  { key: "total_capital", label: "Total capital (INR)" },
  { key: "capital_to_use", label: "Capital to use (INR)" },
  { key: "daily_profit_target", label: "Daily profit target (INR)" },
  { key: "max_daily_loss", label: "Max daily loss (INR)" },
  { key: "position_size_pct", label: "Position size (% per trade)" },
  { key: "max_concurrent_positions", label: "Max concurrent positions" },
];

function ConfigForm() {
  const mode = useMode();
  const [config, setConfig] = useState<CapitalConfig | null | undefined>(undefined);
  const [form, setForm] = useState<Record<string, number>>({});
  const [status, setStatus] = useState<"idle" | "saving" | "saved" | "error">("idle");
  const [loadError, setLoadError] = useState<string | null>(null);
  const [pauseSaving, setPauseSaving] = useState(false);
  const [pauseError, setPauseError] = useState<string | null>(null);

  useEffect(() => {
    setConfig(undefined);
    setLoadError(null);
    supabase
      .from("capital_config")
      .select("*")
      .eq("mode", mode)
      .maybeSingle()
      .then(({ data, error }) => {
        setLoadError(error ? error.message : null);
        setConfig(error ? undefined : (data ?? null));
        if (data) {
          const next: Record<string, number> = {};
          for (const f of FIELDS) next[f.key] = data[f.key] as number;
          setForm(next);
        }
      });
  }, [mode]);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setStatus("saving");
    const { error } = await supabase.from("capital_config").update(form).eq("mode", mode);
    setStatus(error ? "error" : "saved");
  }

  async function togglePaused() {
    if (!config) return;
    const next = !config.paused;
    setPauseSaving(true);
    setPauseError(null);
    const { error } = await supabase.from("capital_config").update({ paused: next }).eq("mode", mode);
    setPauseSaving(false);
    if (error) {
      setPauseError(error.message);
    } else {
      setConfig({ ...config, paused: next });
    }
  }

  return (
    <div className="space-y-4">
      <ModeToggle />

      {loadError && (
        <p className="text-sm rounded-lg p-3" style={{ background: "#fdecea", color: "var(--status-critical)" }}>
          Query failed: {loadError}
        </p>
      )}
      {!loadError && config === undefined && <p style={{ color: "var(--text-muted)" }}>Loading…</p>}
      {!loadError && config === null && (
        <p style={{ color: "var(--text-muted)" }}>No capital_config row for {mode} yet.</p>
      )}

      {config && (
        <div
          className="rounded-xl p-4 flex items-center justify-between max-w-sm shadow-sm"
          style={{
            background: config.paused ? "#fdf3e2" : "#eafaea",
            border: `1px solid ${config.paused ? STATUS.warning : STATUS.good}33`,
          }}
        >
          <div className="flex items-start gap-2">
            <span
              className="inline-block w-2 h-2 rounded-full shrink-0 mt-1.5"
              style={{ background: config.paused ? STATUS.critical : STATUS.good }}
              aria-hidden
            />
            <div>
              <div className="text-sm font-medium" style={{ color: "var(--text-primary)" }}>
                {config.paused ? "Stopped" : "Running"}
              </div>
              <div className="text-xs mt-0.5" style={{ color: "var(--text-secondary)" }}>
                {config.paused
                  ? `${mode} is paused — no model calls, no new orders.`
                  : `${mode} is active — cycles run on schedule.`}
              </div>
              {pauseError && (
                <div className="text-xs mt-1" style={{ color: "var(--status-critical)" }}>
                  {pauseError}
                </div>
              )}
            </div>
          </div>
          <button
            onClick={togglePaused}
            disabled={pauseSaving}
            className="rounded-md px-4 py-2 text-sm font-medium text-white disabled:opacity-50 shrink-0 ml-3 transition-colors"
            style={{ background: config.paused ? STATUS.good : STATUS.critical }}
          >
            {pauseSaving ? "…" : config.paused ? "Start" : "Stop"}
          </button>
        </div>
      )}

      {config && (
        <form
          onSubmit={handleSubmit}
          className="max-w-sm space-y-3 rounded-xl p-5 shadow-sm"
          style={{ background: "var(--surface-1)", border: "1px solid var(--border)" }}
        >
          {FIELDS.map((f) => (
            <label key={f.key} className="block text-sm">
              <span style={{ color: "var(--text-muted)" }}>{f.label}</span>
              <input
                type="number"
                step="any"
                value={form[f.key] ?? ""}
                onChange={(e) => setForm({ ...form, [f.key]: Number(e.target.value) })}
                className="mt-1 w-full rounded-md px-3 py-1.5 tabular-nums outline-none focus:ring-2 focus:ring-neutral-900/20"
                style={{ border: "1px solid var(--border)" }}
                required
              />
            </label>
          ))}
          <button
            type="submit"
            disabled={status === "saving"}
            className="rounded-md bg-neutral-900 text-white text-sm px-3 py-1.5 disabled:opacity-50 hover:bg-neutral-800 transition-colors"
          >
            {status === "saving" ? "Saving…" : "Save"}
          </button>
          {status === "saved" && (
            <p className="text-sm" style={{ color: "var(--status-good)" }}>
              Saved.
            </p>
          )}
          {status === "error" && (
            <p className="text-sm" style={{ color: "var(--status-critical)" }}>
              Save failed — check you&apos;re signed in and RLS allows the update.
            </p>
          )}
        </form>
      )}
    </div>
  );
}
