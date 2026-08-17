"use client";

import { useEffect, useMemo, useState, type ReactNode } from "react";
import { supabase } from "@/lib/supabase";
import { useMode, ModeToggle } from "@/components/ModeToggle";
import { STATUS } from "@/lib/palette";
import { rejectionBreakdown, worstBucketByDimension } from "@/lib/learningAggregates";
import type {
  AdaptiveStrategyVersion,
  HoldEvaluation,
  LearningStatistic,
  Recommendation,
  StrategySimulation,
} from "@/lib/types";

// LEARNING_HISTORY_WINDOW_DAYS default (src/config.py) — the window
// rejection_breakdown() uses when no explicit `since` is passed.
const HISTORY_WINDOW_DAYS = 180;
// RECOMMENDATION_MIN_SAMPLE_SIZE default (src/config.py) — same floor
// identify_weaknesses() gates buckets on.
const MIN_SAMPLE_SIZE = 20;

interface LearningData {
  learningStats: LearningStatistic[];
  holdEvaluations: HoldEvaluation[];
  recommendations: Recommendation[];
  simulations: StrategySimulation[];
  versions: AdaptiveStrategyVersion[];
}

function DataTable({
  headers,
  rows,
  emptyMessage,
}: {
  headers: string[];
  rows: ReactNode[][];
  emptyMessage: string;
}) {
  if (rows.length === 0) {
    return (
      <p className="text-sm" style={{ color: "var(--text-muted)" }}>
        {emptyMessage}
      </p>
    );
  }
  return (
    <div
      className="overflow-x-auto rounded-xl shadow-sm"
      style={{ background: "var(--surface-1)", border: "1px solid var(--border)" }}
    >
      <table className="w-full text-sm">
        <thead>
          <tr
            className="text-left text-xs uppercase tracking-wide"
            style={{ color: "var(--text-muted)", borderBottom: "1px solid var(--gridline)" }}
          >
            {headers.map((h) => (
              <th key={h} className="px-3 py-2.5 font-medium">
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((cells, i) => (
            <tr
              key={i}
              className="align-top hover:bg-black/[0.02] transition-colors"
              style={{ borderBottom: "1px solid var(--gridline)" }}
            >
              {cells.map((cell, j) => (
                <td key={j} className="px-3 py-2.5">
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div>
      <h2 className="text-xs uppercase tracking-wide mb-2" style={{ color: "var(--text-muted)" }}>
        {title}
      </h2>
      {children}
    </div>
  );
}

const fmt = (n: number | null, digits = 2) => (n == null ? "-" : n.toFixed(digits));

export default function LearningClient() {
  const mode = useMode();
  const [data, setData] = useState<LearningData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      const sinceIso = new Date(Date.now() - HISTORY_WINDOW_DAYS * 24 * 60 * 60 * 1000).toISOString();

      const [stats, holds, recs, sims, versions] = await Promise.all([
        supabase
          .from("learning_statistics")
          .select("dimension_type,dimension_value,expectancy,trades_count")
          .eq("mode", mode),
        supabase
          .from("opportunity_evaluations")
          .select("reason,risk_manager_result")
          .eq("mode", mode)
          .eq("final_decision", "hold")
          .gte("timestamp", sinceIso)
          .limit(5000),
        supabase
          .from("recommendations")
          .select("category,metric_name,current_value,recommended_value,confidence,sample_size,rationale,status,created_at")
          .eq("mode", mode)
          .order("created_at", { ascending: false }),
        supabase
          .from("strategy_simulations")
          .select("id,created_at,passed,p_value,research_note")
          .eq("mode", mode)
          .order("created_at", { ascending: false }),
        supabase
          .from("adaptive_strategy_versions")
          .select("version_number,status,fitness_score,notes,created_at,source_simulation_id")
          .eq("mode", mode)
          .order("created_at", { ascending: false }),
      ]);
      if (cancelled) return;

      const err =
        stats.error?.message ?? holds.error?.message ?? recs.error?.message ?? sims.error?.message ?? versions.error?.message;
      setError(err ?? null);
      setData(
        err
          ? null
          : {
              learningStats: stats.data ?? [],
              holdEvaluations: holds.data ?? [],
              recommendations: recs.data ?? [],
              simulations: sims.data ?? [],
              versions: versions.data ?? [],
            }
      );
    }

    load();
    return () => {
      cancelled = true;
    };
  }, [mode]);

  const weaknesses = useMemo(
    () => (data ? worstBucketByDimension(data.learningStats, MIN_SAMPLE_SIZE) : {}),
    [data]
  );
  const rejections = useMemo(() => (data ? rejectionBreakdown(data.holdEvaluations) : []), [data]);
  const fitnessBySimulationId = useMemo(() => {
    const map = new Map<number, number | null>();
    for (const v of data?.versions ?? []) {
      if (v.source_simulation_id != null) map.set(v.source_simulation_id, v.fitness_score);
    }
    return map;
  }, [data]);

  const recommendationRows = (rows: Recommendation[]) =>
    rows.map((r) => [
      r.category,
      r.metric_name,
      fmt(r.current_value, 3),
      fmt(r.recommended_value, 3),
      r.confidence == null ? "-" : `${r.confidence.toFixed(1)}%`,
      String(r.sample_size),
      r.rationale ?? "",
    ]);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold tracking-tight">Learning</h1>
        <ModeToggle />
      </div>

      {error && (
        <p className="text-sm rounded-lg p-3" style={{ background: "#fdecea", color: "var(--status-critical)" }}>
          Query failed: {error}
        </p>
      )}
      {!error && data === null && <p style={{ color: "var(--text-muted)" }}>Loading…</p>}

      {data && (
        <>
          <Section title="Weaknesses found">
            <DataTable
              headers={["Dimension", "Worst bucket", "Expectancy", "Trades"]}
              rows={Object.entries(weaknesses).map(([dimensionType, bucket]) => [
                dimensionType,
                bucket.value,
                fmt(bucket.expectancy),
                String(bucket.trades_count),
              ])}
              emptyMessage="Not enough data yet."
            />
          </Section>

          <Section title="Rejection breakdown (root cause of &quot;no trade&quot;)">
            <DataTable
              headers={["Rejection reason", "Count", "% of rejections"]}
              rows={rejections.map((r) => [r.reason, String(r.count), `${r.pctOfRejections.toFixed(1)}%`])}
              emptyMessage="No rejected candidates logged yet."
            />
          </Section>

          <Section title="Best pending recommendations">
            <DataTable
              headers={["Category", "Metric", "Current", "Recommended", "Confidence", "Samples", "Rationale"]}
              rows={recommendationRows(
                data.recommendations
                  .filter((r) => r.status === "pending")
                  .sort((a, b) => (b.confidence ?? 0) - (a.confidence ?? 0))
                  .slice(0, 10)
              )}
              emptyMessage="No recommendations yet."
            />
          </Section>

          <Section title="Accepted recommendations">
            <DataTable
              headers={["Category", "Metric", "Current", "Recommended", "Confidence", "Samples", "Rationale"]}
              rows={recommendationRows(data.recommendations.filter((r) => r.status === "approved"))}
              emptyMessage="No recommendations yet."
            />
          </Section>

          <Section title="Rejected recommendations">
            <DataTable
              headers={["Category", "Metric", "Current", "Recommended", "Confidence", "Samples", "Rationale"]}
              rows={recommendationRows(data.recommendations.filter((r) => r.status === "dismissed"))}
              emptyMessage="No recommendations yet."
            />
          </Section>

          <Section title="Simulation results">
            {data.simulations.length === 0 ? (
              <p className="text-sm" style={{ color: "var(--text-muted)" }}>
                No simulations run yet.
              </p>
            ) : (
              <div className="space-y-2">
                {data.simulations.map((s) => (
                  <div
                    key={s.id}
                    className="rounded-xl p-4 shadow-sm space-y-2"
                    style={{ background: "var(--surface-1)", border: "1px solid var(--border)" }}
                  >
                    <div className="flex items-center gap-3 text-sm flex-wrap">
                      <span
                        className="text-xs font-medium px-1.5 py-0.5 rounded"
                        style={{
                          color: s.passed ? STATUS.good : STATUS.critical,
                          background: s.passed ? "#eafaea" : "#fdecea",
                        }}
                      >
                        {s.passed ? "PASSED" : "rejected"}
                      </span>
                      <span style={{ color: "var(--text-secondary)" }}>
                        {new Date(s.created_at).toLocaleString("en-IN", { timeZone: "Asia/Kolkata" })}
                      </span>
                      <span style={{ color: "var(--text-muted)" }}>p-value: {fmt(s.p_value, 4)}</span>
                      <span style={{ color: "var(--text-muted)" }}>
                        Fitness: {fmt(fitnessBySimulationId.get(s.id) ?? null, 1)}
                      </span>
                    </div>
                    {s.research_note && (
                      <pre
                        className="text-sm whitespace-pre-wrap font-sans"
                        style={{ color: "var(--text-secondary)" }}
                      >
                        {s.research_note}
                      </pre>
                    )}
                  </div>
                ))}
              </div>
            )}
          </Section>

          <Section title="Adaptive strategy versions (candidate/approved — never auto-applied)">
            <DataTable
              headers={["Version", "Status", "Fitness", "Notes", "Created"]}
              rows={data.versions.map((v) => [
                `v${v.version_number}`,
                v.status,
                fmt(v.fitness_score, 1),
                v.notes ?? "",
                new Date(v.created_at).toLocaleString("en-IN", { timeZone: "Asia/Kolkata" }),
              ])}
              emptyMessage="No adaptive strategy candidates yet."
            />
          </Section>
        </>
      )}
    </div>
  );
}
