# Enhanced Leaderboard Component

The enhanced leaderboard (`leaderboard-enhanced.tsx`) provides a rich, interactive interface for exploring time-series forecasting results with comprehensive filtering and display options.

## Features

### Dataset Categorization

Datasets are automatically categorized into:

- **Static Datasets** (8 core benchmarks): ETTh1, ETTh2, ETTm1, ETTm2, Weather, Electricity, Traffic, Solar
- **Dynamic Datasets - Stock**: Exchange, FRED-MD, NN5
- **Dynamic Datasets - Traffic**: METR-LA, PEMS-BAY, PEMS03, PEMS04, PEMS07, PEMS08
- **Dynamic Datasets - Air Quality**: cauair, Beijing-Air, AQShunyi, AQWan
- **Other**: Miscellaneous datasets

Each dataset card displays a colored badge indicating its category.

### Filtering Options

#### Dataset Categories
- Toggle visibility for each category independently
- Real-time count of datasets in each category
- Descriptions explain what each category contains

#### Search
- Real-time text search across dataset names
- Case-insensitive matching
- Clear visual feedback when no results match

#### Display Options
Six interactive toggles control visualization:

1. **Show medals** — Display 🥇🥈🥉 for top 3 ranks
2. **Performance bars** — Visual metric indicators below values
3. **Highlight top 3** — Accent background color for leading models
4. **Compact mode** — Reduced spacing for dense viewing
5. **Submission IDs** — Show run identifiers (in detailed view)
6. **Color by rank** — Gradient coloring (gold/silver/bronze for top 3)

### View Modes

Three view modes optimize for different use cases:

- **Table** — Full spreadsheet-style table with all metrics
- **Compact** — Card-based layout, optimized for mobile
- **Detailed** — Table with expandable rows showing submission metadata

### Sorting

All columns are sortable:
- **Model** — Alphabetical
- **MSE** — Ascending (lower is better) or descending
- **MAE** — Ascending (lower is better) or descending
- **Runs** — Number of submissions per model

Click a column header to sort; click again to reverse direction. Active sort shows ↑/↓ indicator.

### Track Switching

Tabs at the top switch between forecasting tracks:
- Time Series
- Spatiotemporal
- Covariate
- RealTime

Each track maintains independent state for filters and sorting.

### Horizon Selection

Each dataset card displays available prediction horizons (e.g., 96, 192, 336, 720 steps). Click to switch between horizons while preserving filters.

## Usage

Component import and basic usage:

```typescript
import { LeaderboardEnhanced } from "@/components/leaderboard-enhanced";
import type { LeaderboardData } from "@/components/leaderboard-enhanced";

<LeaderboardEnhanced 
  data={leaderboardData} 
  dict={dictionary} 
/>
```

### Data Structure

Expected data format:

```json
{
  "schema_version": "1.0",
  "generated_at": "2026-06-02T00:02:02.600509+00:00",
  "primary_metric": "mse",
  "n_submissions": 12,
  "tracks": {
    "time_series": {
      "datasets": {
        "ETTh1": {
          "horizons": {
            "96": [
              {
                "model": "DLinear",
                "mse": 0.294,
                "mae": 0.352,
                "n_runs": 3,
                "submission_ids": ["run_1", "run_2", "run_3"],
                "rank": 1
              }
            ]
          }
        }
      }
    }
  }
}
```

## Responsive Design

- Mobile-first with stacked layout on small screens
- Filter panel adapts to viewport width
- Horizontal scroll for tables on narrow viewports
- Touch-optimized tap targets

## Accessibility

- Semantic HTML structure
- ARIA labels on interactive elements
- Keyboard navigation support
- Focus indicators on all interactive elements
- Color contrast meets WCAG AA standards

## Performance

- Filtering and sorting use React `useMemo` for optimization
- State management via `useState` keeps UI responsive
- No external dependencies beyond React

## Customization

Dataset categories can be extended by modifying the constants:

```typescript
const STATIC_DATASETS = {
  core: ["ETTh1", "ETTh2", ...],
};

const DYNAMIC_DATASETS = {
  stock: ["Exchange", ...],
  traffic: ["METR-LA", ...],
  air_quality: ["cauair", ...],
};
```

## Browser Support

Works on all modern browsers:
- Chrome/Edge 90+
- Firefox 88+
- Safari 14+
- Mobile Safari/Chrome

## Related Files

- `components/leaderboard-enhanced.tsx` — Main component
- `components/tseval-page.tsx` — Page wrapper
- `lib/dictionaries.ts` — i18n copy
- `content/tseval/leaderboard.json` — Data source
