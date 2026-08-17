import { Suspense } from "react";
import LearningClient from "./learning-client";

export default function Page() {
  return (
    <Suspense fallback={<p className="text-neutral-500">Loading…</p>}>
      <LearningClient />
    </Suspense>
  );
}
