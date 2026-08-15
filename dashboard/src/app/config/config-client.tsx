"use client";

import { useEffect, useState } from "react";
import type { Session } from "@supabase/supabase-js";
import { supabase } from "@/lib/supabase";
import { useMode, ModeToggle } from "@/components/ModeToggle";
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
        <h1 className="text-xl font-semibold">Config</h1>
        {session && (
          <button
            onClick={() => supabase.auth.signOut()}
            className="text-sm text-neutral-500 hover:text-neutral-900"
          >
            Sign out
          </button>
        )}
      </div>

      {session === undefined && <p className="text-neutral-500">Loading…</p>}
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
    <form onSubmit={handleSubmit} className="max-w-sm space-y-3 rounded-lg border border-neutral-200 bg-white p-4">
      <p className="text-sm text-neutral-500">Sign in to edit capital, target, and loss limits.</p>
      <input
        type="email"
        placeholder="Email"
        value={email}
        onChange={(e) => setEmail(e.target.value)}
        className="w-full rounded border border-neutral-300 px-3 py-1.5 text-sm"
        required
      />
      <input
        type="password"
        placeholder="Password"
        value={password}
        onChange={(e) => setPassword(e.target.value)}
        className="w-full rounded border border-neutral-300 px-3 py-1.5 text-sm"
        required
      />
      {error && <p className="text-sm text-red-600">{error}</p>}
      <button
        type="submit"
        disabled={submitting}
        className="rounded bg-neutral-900 text-white text-sm px-3 py-1.5 disabled:opacity-50"
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
        <p className="text-red-600 text-sm rounded border border-red-200 bg-red-50 p-3">
          Query failed: {loadError}
        </p>
      )}
      {!loadError && config === undefined && <p className="text-neutral-500">Loading…</p>}
      {!loadError && config === null && (
        <p className="text-neutral-500">No capital_config row for {mode} yet.</p>
      )}

      {config && (
        <div
          className={
            "rounded-lg border p-4 flex items-center justify-between max-w-sm " +
            (config.paused ? "border-amber-300 bg-amber-50" : "border-emerald-200 bg-emerald-50")
          }
        >
          <div>
            <div className="text-sm font-medium">{config.paused ? "Stopped" : "Running"}</div>
            <div className="text-xs text-neutral-600 mt-0.5">
              {config.paused
                ? `${mode} is paused — no model calls, no new orders.`
                : `${mode} is active — cycles run on schedule.`}
            </div>
            {pauseError && <div className="text-xs text-red-600 mt-1">{pauseError}</div>}
          </div>
          <button
            onClick={togglePaused}
            disabled={pauseSaving}
            className={
              "rounded px-4 py-2 text-sm font-medium text-white disabled:opacity-50 shrink-0 ml-3 " +
              (config.paused ? "bg-emerald-600 hover:bg-emerald-700" : "bg-red-600 hover:bg-red-700")
            }
          >
            {pauseSaving ? "…" : config.paused ? "Start" : "Stop"}
          </button>
        </div>
      )}

      {config && (
        <form
          onSubmit={handleSubmit}
          className="max-w-sm space-y-3 rounded-lg border border-neutral-200 bg-white p-4"
        >
          {FIELDS.map((f) => (
            <label key={f.key} className="block text-sm">
              <span className="text-neutral-500">{f.label}</span>
              <input
                type="number"
                step="any"
                value={form[f.key] ?? ""}
                onChange={(e) => setForm({ ...form, [f.key]: Number(e.target.value) })}
                className="mt-1 w-full rounded border border-neutral-300 px-3 py-1.5"
                required
              />
            </label>
          ))}
          <button
            type="submit"
            disabled={status === "saving"}
            className="rounded bg-neutral-900 text-white text-sm px-3 py-1.5 disabled:opacity-50"
          >
            {status === "saving" ? "Saving…" : "Save"}
          </button>
          {status === "saved" && <p className="text-sm text-emerald-600">Saved.</p>}
          {status === "error" && (
            <p className="text-sm text-red-600">
              Save failed — check you&apos;re signed in and RLS allows the update.
            </p>
          )}
        </form>
      )}
    </div>
  );
}
