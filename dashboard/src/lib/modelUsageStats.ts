import type { ModelUsage } from "@/lib/types";

export interface ModelStat {
  model: string;
  calls: number;
  successRate: number;
  fallbackRate: number;
  avgLatencyMs: number;
}

// Mirrors src/agents/reporting_agent.py's _model_usage_stats — same
// grouping, same rates, so the dashboard and the HTML report never
// disagree about what "fallback rate" means.
export function aggregateModelUsage(events: ModelUsage[]): ModelStat[] {
  const byModel = new Map<
    string,
    { calls: number; successes: number; fallbacks: number; totalLatencyMs: number }
  >();

  for (const e of events) {
    const stats = byModel.get(e.model_used) ?? {
      calls: 0,
      successes: 0,
      fallbacks: 0,
      totalLatencyMs: 0,
    };
    stats.calls += 1;
    stats.totalLatencyMs += e.latency_ms;
    if (e.success) stats.successes += 1;
    if (e.fallback_reason) stats.fallbacks += 1;
    byModel.set(e.model_used, stats);
  }

  return Array.from(byModel.entries())
    .map(([model, s]) => ({
      model,
      calls: s.calls,
      successRate: s.successes / s.calls,
      fallbackRate: s.fallbacks / s.calls,
      avgLatencyMs: s.totalLatencyMs / s.calls,
    }))
    .sort((a, b) => b.calls - a.calls);
}
