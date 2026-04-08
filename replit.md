# Workspace

## Overview

pnpm workspace monorepo using TypeScript. Each package manages its own dependencies.

## Stack

- **Monorepo tool**: pnpm workspaces
- **Node.js version**: 24
- **Package manager**: pnpm
- **TypeScript version**: 5.9
- **API framework**: Express 5
- **Database**: PostgreSQL + Drizzle ORM
- **Validation**: Zod (`zod/v4`), `drizzle-zod`
- **API codegen**: Orval (from OpenAPI spec)
- **Build**: esbuild (CJS bundle)

## Key Commands

- `pnpm run typecheck` — full typecheck across all packages
- `pnpm run build` — typecheck + build all packages
- `pnpm --filter @workspace/api-spec run codegen` — regenerate API hooks and Zod schemas from OpenAPI spec
- `pnpm --filter @workspace/db run push` — push DB schema changes (dev only)
- `pnpm --filter @workspace/api-server run dev` — run API server locally

See the `pnpm-workspace` skill for workspace structure, TypeScript setup, and package details.

## Artifacts

### API Server (`artifacts/api-server`)
- Express 5 API server
- Serves `/api` (core API) and `/v1` (AI proxy gateway)
- Proxy routes: `GET /v1/models`, `POST /v1/chat/completions`, `POST /v1/messages`
- Auth: `Authorization: Bearer $PROXY_API_KEY` header required on all `/v1` endpoints
- SDKs: `openai@^6`, `@anthropic-ai/sdk@^0.39`
- Integrations: Replit AI Integrations (OpenAI + Anthropic — no user API keys needed)

### API Portal (`artifacts/api-portal`)
- React + Vite frontend at `/`
- Dark-themed dashboard showing Base URL, auth instructions, endpoint docs, available models, CherryStudio setup guide, and curl examples
- Health check via `GET /api/healthz`

## AI Proxy Gateway

The proxy at `/v1` supports:

- **OpenAI models**: `gpt-5.2`, `gpt-5-mini`, `gpt-5-nano`, `o4-mini`, `o3`
- **Anthropic models**: `claude-opus-4-6`, `claude-sonnet-4-6`, `claude-haiku-4-5`
- Full streaming (SSE), tool calls with bidirectional format conversion
- `POST /v1/chat/completions` — OpenAI-compatible, auto-routes claude models
- `POST /v1/messages` — Anthropic native, auto-routes gpt models
- Keepalive pings every 5s for long-running streams
- 50mb body limit for large payloads

## Secrets

- `PROXY_API_KEY` — Bearer token for `/v1` endpoints
- `SESSION_SECRET` — Express session secret
- `AI_INTEGRATIONS_OPENAI_BASE_URL` / `AI_INTEGRATIONS_OPENAI_API_KEY` — auto-injected by Replit
- `AI_INTEGRATIONS_ANTHROPIC_BASE_URL` / `AI_INTEGRATIONS_ANTHROPIC_API_KEY` — auto-injected by Replit
