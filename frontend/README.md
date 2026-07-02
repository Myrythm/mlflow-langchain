# frontend

Vue 3 + Vite + Tailwind CSS v4 chat UI for the Nimbus support RAG agent.
Design direction: minimal-technical — IBM Plex Sans/Mono, warm paper background,
one orange accent, a conversation column plus a retrieval "inspector" rail.

## Run

Requires Node 20+. The backend must be running on :8000 (see `../backend/README.md`);
the dev server proxies `/api` there.

```bash
npm install
npm run dev        # http://localhost:5173
```

## Other commands

```bash
npm test           # Vitest: SSE parser + useChat composable
npm run build      # production build to dist/
```

## Layout

- `src/api.js` — fetch client + incremental SSE parser for `POST /api/chat`
- `src/composables/useChat.js` — conversation state, stage lifecycle (retrieving →
  generating), history replay, error recovery
- `src/components/` — presentation only
- `src/style.css` — Tailwind v4 theme tokens (colors, fonts) and markdown styles
