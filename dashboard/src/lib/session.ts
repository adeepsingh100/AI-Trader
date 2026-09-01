// Shared between api/session/route.ts and proxy.ts — kept out of the
// route file itself since route.ts files' exports are restricted to
// HTTP method handlers and a small set of route-segment config options.
export const COOKIE_NAME = "session";
