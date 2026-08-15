import { Suspense } from "react";
import OverviewClient from "./overview-client";

export default function Page() {
  return (
    <Suspense fallback={<p className="text-neutral-500">Loading…</p>}>
      <OverviewClient />
    </Suspense>
  );
}
