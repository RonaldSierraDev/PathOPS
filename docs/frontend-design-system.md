# PathOPS Frontend — "Foundry-Style" Design System

A design language spec for building an operations frontend with the restrained, technical, schematic aesthetic of Palantir Foundry. Use this as the source of truth for tokens, typography, layout, and component conventions.

---

## 1. Design Philosophy

Three rules explain almost everything about the look:

1. **Monochrome by default, color by exception.** The UI is dark ink on white (or light ink on near-black). Color appears only as small functional accents — status, selection, alerts. Scarcity of color is what makes each use of it feel deliberate.
2. **Schematic, not decorative.** Illustrations, icons, and charts look like engineering drawings: uniform thin strokes, hatching instead of fills, no gradients, no drop shadows. The visual grammar says "instrument," not "marketing."
3. **De-emphasize, don't delete.** Secondary UI stays on screen but is grayed down to near-silence. Focus is created by muting everything else, not by hiding it.

---

## 2. Design Tokens

### Color

| Token | Value | Use |
|---|---|---|
| `--ink` | `#1C2127` | Primary text, icons, strokes |
| `--ink-soft` | `#404854` | Secondary text |
| `--ink-mute` | `#ABB3BF` | Disabled/tertiary text, inactive icons |
| `--paper` | `#FFFFFF` | Canvas / card background |
| `--paper-tint` | `#F6F7F9` | App background behind cards |
| `--hairline` | `#DCE0E5` | 1px borders, dividers, table rules |
| `--rail` | `#111418` | Dark nav sidebar |
| `--rail-ink` | `#8F99A8` | Sidebar icon default |
| `--accent` | `#2D72D2` | Primary actions, links, selection |
| `--ok` | `#238551` | Success / healthy status |
| `--warn` | `#C87619` | Warning status |
| `--danger` | `#CD4246` | Error / critical status |

(These are drawn from the Blueprint palette family, which is literally Palantir's own — see §5.)

**Rules:**
- Accent colors only ever appear at small scale: dots, badges, icon tints, 2–3px indicator bars. Never as large filled areas.
- One accent color for interaction (`--accent`). Status colors are semantic only.
- Dark rail + light canvas creates the "document in a machine" framing. Keep the rail nearly black even if the rest of the app is light.

### Borders, radius, elevation

- Borders: **1px hairlines everywhere.** No shadows for separation; shadows only for true overlays (menus, dialogs), and even then keep them tight and low-opacity.
- Radius: small and consistent — `2px` on inputs/buttons, `4–8px` on cards. Never mix radii on sibling elements.
- Dividers over boxes: prefer a single hairline rule to a bordered container when separating sections.

### Spacing

- 8px base grid (`4 / 8 / 16 / 24 / 32 / 48 / 64`).
- Be generous: padding inside cards ≥ 24px, gaps between major sections ≥ 48px.
- **Leave one region of every screen empty.** Confidence to not fill space is half of the premium feel.

---

## 3. Typography

| Role | Face | Weight | Notes |
|---|---|---|---|
| Headlines | Grotesk sans (see options below) | 700 | Tight tracking, sentence case or Title Case |
| Body | Same family | 400 | 15–16px, `line-height: 1.6` |
| Inline emphasis | Same family | 700 | Bold key phrases inside body text — a Palantir signature |
| Labels / eyebrows | Same family or mono | 500–600 | 11–12px, `letter-spacing: 0.06em`, UPPERCASE |
| Data / code / IDs | Monospace | 400 | Timestamps, IDs, metrics, log output |

**Typeface options (pick ONE family + one mono):**
- **Inter** — closest free approximation of the Blueprint/Foundry UI feel; excellent at small sizes.
- **IBM Plex Sans + IBM Plex Mono** — slightly more "engineered" personality, superb matching mono.
- **Neue Haas Grotesk / Helvetica Now** (licensed) — if you want the true grotesk pedigree.
- Mono choices: **JetBrains Mono**, **IBM Plex Mono**, or **Söhne Mono** (licensed).

**Rules:**
- One sans family, 2–3 weights max. The mono is for data, never for prose.
- Strong size contrast between headline and body (e.g., 28–32px vs 15px) instead of many intermediate sizes.
- Use uppercase micro-labels ("eyebrows") above sections and cards — this is a big part of the technical register.

---

## 4. Layout & Components

### App shell
```
┌──┬────────────────────────────────────────────┐
│  │  ┌──────────────────────────────────────┐  │
│ R│  │  CARD / CANVAS                       │  │
│ A│  │   eyebrow label                      │  │
│ I│  │   Bold headline                      │  │
│ L│  │   content …            (whitespace)  │  │
│  │  └──────────────────────────────────────┘  │
└──┴────────────────────────────────────────────┘
```
- **Rail:** 48–56px wide, near-black, icon-only, monochrome gray icons; active item gets a white icon + subtle background block or a 2px accent edge.
- **Canvas:** light background, content in a single hairline-bordered card with generous margin on all sides.
- Asymmetric content layout inside the card (diagram left / text right, etc.). Avoid perfectly centered symmetric compositions.

### Components
- **Buttons:** rectangular, 2px radius, 1px border. Primary = accent fill; everything else = ghost/outline. Sentence case, plain verbs ("Save changes", "Run pipeline").
- **Tables:** the workhorse of an ops UI. Hairline row rules only (no zebra striping), uppercase mono column headers, right-aligned numerics in mono, tight row height (32–36px).
- **Status:** small colored dot + gray text label (`● Healthy`), never a large colored pill.
- **Icons:** single stroke-based set (see §5), one consistent stroke width (1.5px), default color `--ink-mute`, active `--ink`.
- **Charts:** thin lines, no area fills or gradients, hairline gridlines, mono axis labels, ink-colored series with accent only for the highlighted series.
- **Empty states / errors:** plain, directive text ("No pipelines yet. Create one to begin."). No illustrations, no apologies.

### Motion
- Minimal and fast: 120–180ms ease-out on hover/focus/expand. No springs, no bounces, no page-load animation sequences. Respect `prefers-reduced-motion`.

---

## 5. Recommended Libraries

### The shortcut: Blueprint
- **[Blueprint](https://blueprintjs.com/)** (`@blueprintjs/core`, React) — Palantir's own open-source design system, extracted from Foundry itself. Tables, trees, forms, popovers, dark theme. If you want the look with minimum custom work, start here. Tradeoffs: React-only, opinionated, less trendy than modern headless stacks.

### The modern custom stack (more control, same aesthetic)
- **React + TypeScript** — the default for data-dense internal tools.
- **Tailwind CSS** — encode the tokens in §2 as a Tailwind theme; fastest way to enforce hairline-border/monochrome discipline.
- **shadcn/ui or Radix Primitives** — accessible headless components you fully restyle; avoids fighting a styled kit's opinions.
- **TanStack Table** — headless table engine for the dense hairline tables that define this look.
- **TanStack Query** — server-state/caching layer between the UI and the PathOPS API.

### Data visualization
- **visx** (Airbnb) — low-level D3 + React primitives; the best fit for austere schematic charts because nothing comes pre-decorated.
- **Recharts** — quicker to ship; strip its defaults (fills, animation, rounded bars) to stay on-style.
- **D3** directly — for the signature piece: a custom node-graph / lineage / "ontology" visualization.
- **React Flow (xyflow)** — if the frontend includes pipeline/DAG or dependency graphs — very Foundry-like and much cheaper than raw D3.

### Icons
- **Blueprint Icons** — the literal Palantir icon set, usable standalone.
- **Lucide** or **Tabler Icons** — stroke-based sets that match the schematic grammar; pick one, never mix.

### Fonts (free)
- **Inter** or **IBM Plex Sans**, plus **JetBrains Mono** or **IBM Plex Mono** — all on Google Fonts.

---

## 6. Frontend Types That Fit an Ops Backend

Pick based on what PathOPS users actually do:

1. **Operational dashboard / console** — live status of workflows, queues, tasks; tables + status dots + a few austere charts. *Fit if: users monitor and intervene.* Stack: React + Tailwind + TanStack Table/Query + visx.
2. **Pipeline / workflow graph view** — visual DAG of processes with node status, the most "Foundry" experience you can build. *Fit if: PathOPS models flows with dependencies.* Stack: React Flow + the dashboard stack.
3. **Admin / record manager** — CRUD over entities (jobs, clients, audits) with detail panes and hairline forms. *Fit if: users configure and edit.* Stack: shadcn/ui forms + TanStack Table; or Refine/React-Admin heavily restyled if speed matters more than polish.
4. **Command-palette-first workspace** — minimal chrome, `⌘K` palette (e.g., `cmdk`) to jump anywhere; layers well onto any of the above and strongly reinforces the power-tool feel.

Most ops products end up as **1 + 2 combined**: a dashboard home with a graph view per workflow.

---

## 7. Anti-patterns (things that break the look)

- Multiple accent colors or colored section backgrounds
- Drop shadows for card separation
- Large border radii (12px+), pill-shaped buttons
- Zebra-striped or heavily boxed tables
- Filled/duotone icons, or mixing icon sets
- Gradient or area-filled charts, chart animations
- Emoji in UI text; exclamation points in system copy
- Center-aligned symmetric hero layouts
- Filling every region of the screen
