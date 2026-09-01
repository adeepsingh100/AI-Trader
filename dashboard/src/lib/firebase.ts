"use client";

import { initializeApp, getApps, getApp } from "firebase/app";
import { getAuth } from "firebase/auth";

// Client-side only — used by the sign-in page for Firebase Auth's
// Email/Password sign-in. Never used for Firestore reads; every page
// still fetches through this app's own API routes, same as before
// (dashboard/src/lib/firebaseAdmin.ts is the server-side counterpart).
const firebaseConfig = {
  apiKey: process.env.NEXT_PUBLIC_FIREBASE_API_KEY,
  authDomain: process.env.NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN,
  projectId: process.env.NEXT_PUBLIC_FIREBASE_PROJECT_ID,
};

const app = getApps().length ? getApp() : initializeApp(firebaseConfig);

export const auth = getAuth(app);
