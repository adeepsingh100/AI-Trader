import type { HoldEvaluation, LearningStatistic } from "@/lib/types";

export interface RejectionCount {
  reason: string;
  count: number;
  pctOfRejections: number;
}

// Mirrors src/learning/rejection_analysis.py::rejection_breakdown +
// _rejection_label — risk_manager_result is the more specific reason when
// present, reason otherwise, "unknown" if neither — grouped and sorted
// desc by count, so the dashboard and the HTML report never disagree.
export function rejectionBreakdown(rows: HoldEvaluation[]): RejectionCount[] {
  if (rows.length === 0) return [];

  const counts = new Map<string, number>();
  for (const row of rows) {
    const label = row.risk_manager_result || row.reason || "unknown";
    counts.set(label, (counts.get(label) ?? 0) + 1);
  }

  const total = rows.length;
  return Array.from(counts.entries())
    .map(([reason, count]) => ({ reason, count, pctOfRejections: (count / total) * 100 }))
    .sort((a, b) => b.count - a.count);
}

export interface WeaknessBucket {
  value: string;
  expectancy: number;
  trades_count: number;
}

// Mirrors src/learning/weakness_detection.py::identify_weaknesses's
// worst_by_dimension — the only half reports.py's own _weakness_rows
// renders, so this matches the reference being mirrored (best_by_dimension
// and the indicator extremes aren't surfaced there either).
export function worstBucketByDimension(
  rows: LearningStatistic[],
  minSampleSize = 20
): Record<string, WeaknessBucket> {
  const byDimension = new Map<string, LearningStatistic[]>();
  for (const row of rows) {
    if ((row.trades_count || 0) < minSampleSize || row.expectancy == null) continue;
    const bucket = byDimension.get(row.dimension_type) ?? [];
    bucket.push(row);
    byDimension.set(row.dimension_type, bucket);
  }

  const worst: Record<string, WeaknessBucket> = {};
  for (const [dimensionType, eligible] of byDimension) {
    const min = eligible.reduce((a, b) => ((b.expectancy as number) < (a.expectancy as number) ? b : a));
    worst[dimensionType] = {
      value: min.dimension_value,
      expectancy: min.expectancy as number,
      trades_count: min.trades_count,
    };
  }
  return worst;
}
