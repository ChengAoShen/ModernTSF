# diaugeia

> διαύγεια · Greek for *clarity, lucidity, transparency*

The official website for **Diaugeia.AI** — open, community-run infrastructure for AI research.

Built with **Next.js (App Router) + MDX + Tailwind CSS v4**, with light/dark themes and full English / 中文 bilingual support. Deployed as a **statically-exported site on Cloudflare Workers**, with a thin worker that powers the community contact / join forms.

## Getting started

```bash
bun install
bun run dev      # http://localhost:3000 (forms need the worker — see below)
```

Scripts:

```bash
bun run build    # static export -> out/
bun run preview  # build + run the worker locally (forms work, send_email is simulated)
bun run deploy   # build + wrangler deploy
```

`bun run build` produces a fully static site in `out/` (`output: "export"`). It is served by **Cloudflare Workers Static Assets**; a small `worker.js` only runs for `POST /api/submit`.

## Deploying

Hosted on **Cloudflare Workers** (`diaugeia.ai`). The forms send mail natively through the Email Routing `send_email` binding — no third-party service. See **[DEPLOYMENT.md](./DEPLOYMENT.md)** for the full setup (GitHub auto-deploy, the binding, custom domain).

Because fonts are self-hosted (`next/font`), nothing is fetched from Google at runtime — the site loads fine in mainland China. (The build machine still needs internet to fetch fonts at build time.)

## Community forms

Two forms feed a single endpoint:

- **Contact** — at the bottom of the home page (`#contact`): name, email, message.
- **Join** — its own bilingual `/join` page: name, email, affiliation, role, interests, link, motivation.

Both POST same-origin JSON to `/api/submit`, handled by `worker.js`, which validates + drops honeypot/oversized payloads and emails the submission via the `send_email` binding (Reply-To = submitter). UI lives in `components/form-kit.tsx`, `components/contact-section.tsx`, `components/join-page.tsx`; copy is in `lib/dictionaries.ts`.

## Adding content (Research / News)

Content lives in `content/<category>/<locale>/`:

```
content/
  research/{en,zh}/your-post.mdx
  news/{en,zh}/your-post.mdx
```

To publish a new article, drop an MDX file into the right folders. Use the **same filename (slug)** in `en/` and `zh/` so the language switcher maps between them. Frontmatter:

```markdown
---
title: "Your title"
date: "2026-05-29"          # YYYY-MM-DD — drives ordering
author: "Diaugeia.AI team"
summary: "One-line summary shown in cards and on the page."
tags: ["Infrastructure"]    # the first tag shows on cards
---

Your body in Markdown / MDX. Headings, lists, code blocks, blockquotes,
links, images — styled via the `.prose` rules.
```

Lists, dates and the sitemap update automatically. Commit + push → Cloudflare rebuilds and deploys.

## Project structure

```
app/
  (en)/                # English at the root: /, /research, /news, /join
  zh/                  # Chinese under /zh
  layout.tsx           # fonts, theme provider, metadata (metadataBase: diaugeia.ai)
  robots.ts, sitemap.ts
  globals.css          # design tokens + theme + prose styles
components/            # nav, footer, home, cards, MDX, and the form kit
lib/
  i18n.ts              # locales + path helpers (en at root, zh under /zh)
  dictionaries.ts      # all UI copy (en / zh) — edit text here
  content.ts           # MDX loading
content/               # the articles (see above)
worker.js              # Workers entry: /api/submit + static assets
wrangler.jsonc         # worker name, assets (out/), send_email binding
public/                # logo + manifest
```

## Design

- **Palette:** graphite-led neutral with a single dark-gold accent (`--accent`) from the light-beam logo. Warm-ivory paper in light mode, deep graphite ink in dark.
- **Theme tokens** are CSS variables in `app/globals.css` (`:root` + `.dark`), mapped to Tailwind utilities (`bg-paper`, `text-ink`, `text-accent`, …).
- **Type (self-hosted via `next/font`, nothing fetched at runtime):** Newsreader serif (+ Noto Serif SC for CJK) for display text, Inter for UI/body, Fira Code for code. Code is highlighted with Shiki via `rehype-pretty-code`.
- **Icons:** [lucide-react](https://lucide.dev). **Logo:** `public/diaugeia.png`, also the favicon (`app/icon.png`).
- **UI copy** is centralized in `lib/dictionaries.ts` — change wording there, not in components.

## Editing the navigation / copy

The top nav is intentionally minimal (**Research**, **News**) plus a gold **Join** button, a language switcher and a theme toggle. Labels and all marketing copy live in `lib/dictionaries.ts`.
