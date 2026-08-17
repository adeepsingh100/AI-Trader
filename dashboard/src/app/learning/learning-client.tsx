"use client";

import { useEffect, useMemo, useState, type ReactNode } from "react";
import { supabase } from "@/lib/supabase";
import { useMode, ModeToggle } from "@/components/ModeToggle";
import { STATUS } from "@/lib/palette";
import { collectEvidence, computeLearningStage, rejectionBreakdown, worstBucketByDimension } from "@/lib/learningAggregates";
import type {
  AdaptiveStrategyVersion,
  LearningStatistic,
  LearningStatus,
  OpportunityEvaluationRow,
  Recommendation,
  StrategySimulation,
} from "@/lib/types";

// LEARNING_HISTORY_WINDOW_DAYS default (src/config.py) — the window
// rejection_breakdown() uses when no explicit `since` is passed.
const HISTORY_WINDOW_DAYS = 180;
// LEARNING_STAGE_OBSERVATION_MIN_TRADES default (src/config.py) — same
// floor identify_weaknesses() gates buckets on.
const MIN_SAMPLE_SIZE = 25;

interface LearningData {
  learningStats: LearningStatistic[];
  evaluations: OpportunityEvaluationRow[];
  featureNames: string[];
  recommendations: Recommendation[];
  simulations: StrategySimulation[];
  versions: AdaptiveStrategyVersion[];
  closedTrades: { pnl: number | null }[];
  promotionEligible: boolean;
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

const STAGE_ORDER = ["BOOTSTRAP", "OBSERVATION", "HYPOTHESIS", "SIMULATION", "VALIDATION"] as const;

function LearningStatusCard({ status }: { status: LearningStatus }) {
  const stageIndex = STAGE_ORDER.indexOf(status.stage);
  const progressPct = ((stageIndex + 1) / STAGE_ORDER.length) * 100;
  const e = status.evidence;

  return (
    <div
      className="rounded-xl p-4 shadow-sm space-y-3"
      style={{ background: "var(--surface-1)", border: "1px solid var(--border)" }}
    >
      <div className="flex items-center gap-3 flex-wrap">
        <span
          className="text-xs font-semibold px-2 py-1 rounded uppercase tracking-wide"
          style={{ color: "#fff", background: "var(--text-primary)" }}
        >
          {status.stage}
        </span>
        <span className="text-lg font-semibold tabular-nums" style={{ color: "var(--text-primary)" }}>
          {status.evidenceReadinessPct.toFixed(0)}%
        </span>
        <span className="text-sm" style={{ color: "var(--text-secondary)" }}>
          evidence readiness
        </span>
        {status.promotionEligible && (
          <span
            className="text-xs font-medium px-1.5 py-0.5 rounded"
            style={{ color: STATUS.good, background: "#eafaea" }}
          >
            promotion eligible
          </span>
        )}
      </div>

      <div className="h-1.5 rounded-full overflow-hidden" style={{ background: "var(--page-plane)" }}>
        <div
          className="h-full rounded-full transition-all"
          style={{ width: `${progressPct}%`, background: "var(--text-primary)" }}
        />
      </div>

      <p className="text-sm" style={{ color: "var(--text-secondary)" }}>
        {status.currentActivity}
      </p>
      <p className="text-xs" style={{ color: "var(--text-muted)" }}>
        {status.reason}
      </p>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs pt-1" style={{ color: "var(--text-muted)" }}>
        <span>
          Closed trades <br />
          <span className="text-sm tabular-nums" style={{ color: "var(--text-secondary)" }}>
            {status.tradesCollected} ({status.winningTrades}W / {status.losingTrades}L)
          </span>
        </span>
        <span>
          Rejected opportunities <br />
          <span className="text-sm tabular-nums" style={{ color: "var(--text-secondary)" }}>
            {status.rejectedTrades}
          </span>
        </span>
        <span>
          Symbols covered <br />
          <span className="text-sm tabular-nums" style={{ color: "var(--text-secondary)" }}>
            {e.symbolsCovered}
          </span>
        </span>
        <span>
          Market regimes covered <br />
          <span className="text-sm tabular-nums" style={{ color: "var(--text-secondary)" }}>
            {e.marketRegimesCovered} / 6
          </span>
        </span>
        <span>
          Trading hours covered <br />
          <span className="text-sm tabular-nums" style={{ color: "var(--text-secondary)" }}>
            {e.tradingHoursCovered} / 24
          </span>
        </span>
        <span>
          Feature coverage <br />
          <span className="text-sm tabular-nums" style={{ color: "var(--text-secondary)" }}>
            {e.featureCoveragePct.toFixed(1)}%
          </span>
        </span>
        <span>
          Confidence coverage <br />
          <span className="text-sm tabular-nums" style={{ color: "var(--text-secondary)" }}>
            {e.confidenceCoveragePct.toFixed(1)}%
          </span>
        </span>
        <span>
          Recs / sims / candidates <br />
          <span className="text-sm tabular-nums" style={{ color: "var(--text-secondary)" }}>
            {status.recommendationsCount} / {status.simulationsCount} / {status.candidatesCount}
          </span>
        </span>
      </div>

      {status.nextStage && status.evidenceGaps.length > 0 && (
        <p className="text-xs pt-1" style={{ color: "var(--text-muted)" }}>
          Missing evidence for {status.nextStage}: {status.evidenceGaps.join(" OR ")}
        </p>
      )}
      {e.symbolsRarelyQualifying.length > 0 && (
        <p className="text-xs" style={{ color: "var(--text-muted)" }}>
          Symbols rarely qualifying: {e.symbolsRarelyQualifying.map((s) => s.symbol).slice(0, 5).join(", ")}
        </p>
      )}
      {e.regimesWithNoCandidates.length > 0 && (
        <p className="text-xs" style={{ color: "var(--text-muted)" }}>
          Regimes with no candidates: {e.regimesWithNoCandidates.join(", ")}
        </p>
      )}
    </div>
  );
}

export default function LearningClient() {
  const mode = useMode();
  const [data, setData] = useState<LearningData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      const sinceIso = new Date(Date.now() - HISTORY_WINDOW_DAYS * 24 * 60 * 60 * 1000).toISOString();

      const [stats, evals, recs, sims, versions, trades, strategyVersions, features] = await Promise.all([
        supabase
          .from("learning_statistics")
          .select("dimension_type,dimension_value,expectancy,trades_count")
          .eq("mode", mode),
        supabase
          .from("opportunity_evaluations")
          .select("symbol,market_regime,timestamp,final_decision,llm_decision,reason,risk_manager_result")
          .eq("mode", mode)
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
        supabase
          .from("trades")
          .select("pnl")
          .eq("mode", mode)
          .eq("status", "closed")
          .gte("closed_at", sinceIso),
        supabase
          .from("strategy_versions")
          .select("promotion_eligible")
          .order("version_number", { ascending: false })
          .limit(1),
        supabase.from("feature_importance").select("feature_name,timeframe").eq("mode", mode),
      ]);
      if (cancelled) return;

      const err =
        stats.error?.message ??
        evals.error?.message ??
        recs.error?.message ??
        sims.error?.message ??
        versions.error?.message ??
        trades.error?.message ??
        strategyVersions.error?.message ??
        features.error?.message;
      setError(err ?? null);
      setData(
        err
          ? null
          : {
              learningStats: stats.data ?? [],
              evaluations: evals.data ?? [],
              featureNames: (features.data ?? [])
                .filter((f) => f.timeframe !== "blended")
                .map((f) => f.feature_name),
              recommendations: recs.data ?? [],
              simulations: sims.data ?? [],
              versions: versions.data ?? [],
              closedTrades: trades.data ?? [],
              promotionEligible: strategyVersions.data?.[0]?.promotion_eligible ?? false,
            }
      );
    }

    load();
    return () => {
      cancelled = true;
    };
  }, [mode]);

  const evidence = useMemo(
    () =>
      data
        ? collectEvidence(
            data.closedTrades,
            data.evaluations,
            new Set(data.featureNames),
            new Set(data.learningStats.map((r) => r.dimension_type))
          )
        : null,
    [data]
  );
  const learningStatus = useMemo(
    () =>
      data && evidence
        ? computeLearningStage(
            evidence,
            data.recommendations.length,
            data.simulations.length,
            data.versions.length,
            data.promotionEligible
          )
        : null,
    [data, evidence]
  );
  const weaknesses = useMemo(
    () => (data ? worstBucketByDimension(data.learningStats, MIN_SAMPLE_SIZE) : {}),
    [data]
  );
  const rejections = useMemo(
    () => (data ? rejectionBreakdown(data.evaluations.filter((e) => e.final_decision === "hold")) : []),
    [data]
  );
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

      {data && learningStatus && (
        <>
          <LearningStatusCard status={learningStatus} />

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
