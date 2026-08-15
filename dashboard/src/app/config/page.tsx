import { Suspense } from "react";
import ConfigClient from "./config-client";

export default function Page() {
  return (
    <Suspense fallback={<p className="text-neutral-500">Loading…</p>}>
      <ConfigClient />
    </Suspense>
  );
}
