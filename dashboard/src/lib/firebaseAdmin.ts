import { cert, getApps, initializeApp } from "firebase-admin/app";
import { getAuth } from "firebase-admin/auth";
import { getFirestore } from "firebase-admin/firestore";

// Server-side only — never imported by a "use client" file. Same
// FIREBASE_SERVICE_ACCOUNT_JSON env var as the Python bot (src/config.py),
// one secret shared by both halves of this repo.
const app = getApps().length
  ? getApps()[0]
  : initializeApp({
      credential: cert(JSON.parse(process.env.FIREBASE_SERVICE_ACCOUNT_JSON ?? "{}")),
    });

export const adminAuth = getAuth(app);
export const adminDb = getFirestore(app);
