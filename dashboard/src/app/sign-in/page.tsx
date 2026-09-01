"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { signInWithEmailAndPassword } from "firebase/auth";
import { auth } from "@/lib/firebase";

export default function SignInPage() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      const credential = await signInWithEmailAndPassword(auth, email, password);
      const idToken = await credential.user.getIdToken();
      const res = await fetch("/api/session", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ idToken }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.error ?? "sign-in failed");
      }
      router.push("/");
      router.refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="flex justify-center">
      <form
        onSubmit={handleSubmit}
        className="w-full max-w-sm space-y-3 rounded-xl p-5 shadow-sm mt-12"
        style={{ background: "var(--surface-1)", border: "1px solid var(--border)" }}
      >
        <h1 className="text-lg font-semibold tracking-tight">Sign in</h1>
        <label className="block text-sm">
          <span style={{ color: "var(--text-muted)" }}>Email</span>
          <input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="mt-1 w-full rounded-md px-3 py-1.5 text-sm outline-none focus:ring-2 focus:ring-neutral-900/20"
            style={{ border: "1px solid var(--border)" }}
            required
            autoFocus
          />
        </label>
        <label className="block text-sm">
          <span style={{ color: "var(--text-muted)" }}>Password</span>
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="mt-1 w-full rounded-md px-3 py-1.5 text-sm outline-none focus:ring-2 focus:ring-neutral-900/20"
            style={{ border: "1px solid var(--border)" }}
            required
          />
        </label>
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
    </div>
  );
}
