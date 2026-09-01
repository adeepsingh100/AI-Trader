import { cert, getApps, initializeApp } from "firebase-admin/app";
import { getAuth } from "firebase-admin/auth";
import { getFirestore } from "firebase-admin/firestore";

// Server-side only — never imported by a "use client" file. Same
// FIREBASE_SERVICE_ACCOUNT_JSON env var as the Python bot (src/config.py),
// one secret shared by both halves of this repo.
//
// package.json's "jose" override (^4.15.9) is load-bearing for this file:
// firebase-admin/auth -> jwks-rsa@4.x -> jose@6 (ESM-only, no CJS build),
// and requiring that ESM package crashed with ERR_REQUIRE_ESM in every
// deployed Vercel function that imports this file (proxy.ts included) —
// production only, `next dev` never hit the failing code path. jwks-rsa
// only calls jose's importJWK/exportSPKI, both present and behavior-
// compatible on jose@4, and we never invoke jwks-rsa's JWKS-fetching
// code anyway (verifySessionCookie doesn't need it) — the override is
// forcing an unused transitive dependency back to a requireable format,
// not changing anything this app actually calls into. Don't drop it
// without confirming a production deploy still works, not just `next dev`.
const app = getApps().length
  ? getApps()[0]
  : initializeApp({
      credential: cert(JSON.parse(process.env.FIREBASE_SERVICE_ACCOUNT_JSON ?? "{}")),
    });

export const adminAuth = getAuth(app);
export const adminDb = getFirestore(app);
