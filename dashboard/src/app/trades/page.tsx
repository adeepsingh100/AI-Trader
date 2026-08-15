import { Suspense } from "react";
import TradesClient from "./trades-client";

export default function Page() {
  return (
    <Suspense fallback={<p className="text-neutral-500">Loading…</p>}>
      <TradesClient />
    </Suspense>
  );
}
