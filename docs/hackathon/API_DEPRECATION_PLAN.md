# API Deprecation Plan

最終更新: 2026-02-28

## Current Policy

UI互換を維持するため、当面は `lib/api-client.ts` で以下の2系統を切替運用する。

- Primary: Firebase Callable (`NEXT_PUBLIC_FIREBASE_FUNCTIONS_BASE_URL` が設定済み)
- Fallback: Next API routes (`/api/*`)

## Callable Endpoints

- `initUser`
- `startDay`
- `beginCommission`
- `runInterpreter`
- `submitComposition`
- `createMusicJob`
- `getMusicJobStatus`
- `purchasePart`
- `createShareLink`
- `syncGameState`

## Legacy Next API Routes (Temporary)

- `/api/interpreter`
- `/api/compose/submit`
- `/api/music/jobs`
- `/api/music/jobs/[jobId]`
- `/api/game/state`

## Exit Criteria

1. Callable経由で主要E2E (Street -> Compose -> Submit -> Job complete -> Gallery) が安定
2. Weave traceと評価指標がCallable経路で取得できる
3. ローカル開発手順に emulator + callable setup が記載されている

## Deprecation Steps

1. API clientに利用経路ログを追加（callable/fallback）
2. CIでcallable mode integration testを追加
3. `/api/*` を read-only diagnostics endpoint に縮退
4. 最終的に `/api/*` を削除（debug routeを除く）
