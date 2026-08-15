import { Suspense } from "react";
import EvolutionClient from "./evolution-client";

export default function Page() {
  return (
    <Suspense fallback={<p className="text-neutral-500">Loading…</p>}>
      <EvolutionClient />
    </Suspense>
  );
}
