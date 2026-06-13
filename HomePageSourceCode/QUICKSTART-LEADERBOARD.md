# Enhanced Leaderboard - Quick Start Guide

## What's New

The enhanced leaderboard adds rich filtering and interaction to the TSEval page with:

✅ **Dataset categorization** — Static (8 core) vs Dynamic (Stock/Traffic/Air Quality)  
✅ **15 interactive controls** — Checkboxes for categories, display options, search  
✅ **3 view modes** — Table, Compact, Detailed  
✅ **Full sorting** — All columns with visual indicators  
✅ **Real-time filtering** — Instant results as you type/click  
✅ **Responsive design** — Works on mobile and desktop  

## Files Created

```
HomePageSourceCode/
├── components/
│   ├── leaderboard-enhanced.tsx          ← Main component (710 lines)
│   ├── tseval-page.tsx                   ← Updated to use enhanced version
│   ├── LEADERBOARD.md                    ← Full English documentation
│   └── README-LEADERBOARD-ZH.md          ← Full Chinese documentation
└── content/tseval/
    └── leaderboard-sample.json           ← Sample data with all categories
```

## Quick Test

### 1. Start dev server

```bash
cd /Users/mjm/ModernTSF/HomePageSourceCode
npm install  # or bun install
npm run dev
```

### 2. Visit the page

Open: http://localhost:3000/tseval

### 3. Try the features

- **Track tabs** at top — Switch between Time Series / Spatiotemporal / Covariate
- **Filters & Options panel** — Expand to see all controls
- **Dataset categories** — Uncheck "Static Datasets" to hide ETTh1, ETTh2, etc.
- **Search** — Type "METR" to filter to traffic datasets
- **View modes** — Toggle between Table / Compact / Detailed
- **Sorting** — Click column headers (Model, MSE, MAE, Runs)
- **Display options** — Toggle medals, performance bars, colors, etc.

## Integration with Real Data

The component is already integrated! It reads from:

```typescript
// components/tseval-page.tsx
import raw from "@/content/tseval/leaderboard.json";
```

To use the sample data with all categories:

```bash
cd /Users/mjm/ModernTSF/HomePageSourceCode
cp content/tseval/leaderboard-sample.json content/tseval/leaderboard.json
npm run dev
```

## Architecture

```
┌─────────────────────────────────────────────┐
│  TSEvalPage (tseval-page.tsx)              │
│  - Loads leaderboard.json                   │
│  - Passes to LeaderboardEnhanced            │
└─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│  LeaderboardEnhanced (main component)      │
│  - Track tabs                               │
│  - Filter panel with 15 controls            │
│  - Search + category checkboxes             │
│  - Display option toggles                   │
│  - View mode selector                       │
└─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│  DatasetCard (per dataset)                 │
│  - Category badge                           │
│  - Horizon selector                         │
│  - Table or CompactView                     │
│  - Expandable rows (detailed mode)          │
└─────────────────────────────────────────────┘
```

## Key Components

### LeaderboardEnhanced
Main container with all state management and filtering logic.

### DatasetCard
Individual dataset display with sorting and view switching.

### CompactView
Mobile-optimized card layout.

### Checkbox
Reusable checkbox with label and description.

### Th / SortTh
Table headers with sorting indicators.

## State Management

15 independent state variables for fine-grained control:

```typescript
// Track and sorting
const [track, setTrack] = useState(trackKeys[0]);
const [sortKey, setSortKey] = useState<SortKey>("mse");
const [sortDir, setSortDir] = useState(1);

// View and search
const [viewMode, setViewMode] = useState<ViewMode>("table");
const [searchQuery, setSearchQuery] = useState("");

// Category filters (5 checkboxes)
const [showStatic, setShowStatic] = useState(true);
const [showDynamicStock, setShowDynamicStock] = useState(true);
const [showDynamicTraffic, setShowDynamicTraffic] = useState(true);
const [showDynamicAir, setShowDynamicAir] = useState(true);
const [showOther, setShowOther] = useState(true);

// Display options (6 checkboxes)
const [showMedals, setShowMedals] = useState(true);
const [showVisualization, setShowVisualization] = useState(true);
const [highlightTop3, setHighlightTop3] = useState(true);
const [compactMode, setCompactMode] = useState(false);
const [showSubmissionIds, setShowSubmissionIds] = useState(false);
const [colorByPerformance, setColorByPerformance] = useState(true);
```

## Customization Examples

### Add a new dataset category

Edit constants in leaderboard-enhanced.tsx - add to DYNAMIC_DATASETS then update the filter checkboxes.

### Change category colors

Edit categoryBadge object in the DatasetCard component to customize badge colors.

### Adjust mobile breakpoints

Tailwind breakpoints are in className strings (sm:, lg: prefixes).

## Performance Notes

- useMemo caches filtered datasets and stats
- No external dependencies beyond React
- Efficient sorting with native Array.sort
- Lazy evaluation - filters only run when dependencies change

## Next Steps

1. Test with real data - Ensure your leaderboard.json has varied datasets
2. Customize colors - Match your brand (currently uses Diaugeia gold accent)
3. Add translations - Update lib/dictionaries.ts for other languages
4. Mobile testing - Check on actual devices
5. Deploy - Run build and deploy commands

## Troubleshooting

### No datasets showing
- Check leaderboard.json is valid
- Verify track names match (time_series, spatiotemporal, covariate)
- Try clicking "Reset all" in the filter panel

### Filters not working
- Open browser console for errors
- Verify dataset names match the category patterns

### Styling issues
- Check Tailwind CSS is loading
- Verify app/globals.css has the theme tokens

## Documentation

- Full English docs: components/LEADERBOARD.md
- Full Chinese docs: README-LEADERBOARD-ZH.md
- Component code: components/leaderboard-enhanced.tsx (well-commented)

## Support

For issues specific to ModernTSF or the leaderboard backend:
https://github.com/Diaugeia/ModernTSF
