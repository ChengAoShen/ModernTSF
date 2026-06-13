"use client";

import { useMemo, useState } from "react";
import React from "react";
import type { Dictionary } from "@/lib/dictionaries";

// ---------------------------------------------------------------------------
//  Enhanced Leaderboard with rich filtering and interaction
// ---------------------------------------------------------------------------

export interface LeaderRow {
  model: string;
  mse: number | null;
  mae: number | null;
  n_runs: number;
  submission_ids: string[];
  rank?: number;
}

interface DatasetBlock {
  horizons: Record<string, LeaderRow[]>;
}

interface TrackBlock {
  datasets: Record<string, DatasetBlock>;
}

export interface LeaderboardData {
  schema_version: string;
  generated_at: string;
  primary_metric: string;
  n_submissions: number;
  n_rejected: number;
  tracks: Record<string, TrackBlock>;
}

const MEDAL: Record<number, string> = { 1: "🥇", 2: "🥈", 3: "🥉" };

type SortKey = "mse" | "mae" | "model" | "runs";
type ViewMode = "table" | "compact" | "detailed";

// Dataset categorization
const STATIC_DATASETS = {
  core: ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "Electricity", "Traffic", "Solar"],
};

const DYNAMIC_DATASETS = {
  stock: ["Exchange", "FRED-MD", "NN5"],
  traffic: ["METR-LA", "PEMS-BAY", "PEMS03", "PEMS04", "PEMS07", "PEMS08"],
  air_quality: ["cauair", "Beijing-Air", "AQShunyi", "AQWan"],
};

function fmt(v: number | null | undefined): string {
  return v === null || v === undefined ? "—" : v.toFixed(4);
}

function getDatasetCategory(datasetName: string): "static" | "dynamic-stock" | "dynamic-traffic" | "dynamic-air" | "other" {
  const lower = datasetName.toLowerCase();
  if (STATIC_DATASETS.core.some(d => lower.includes(d.toLowerCase()))) return "static";
  if (DYNAMIC_DATASETS.stock.some(d => lower.includes(d.toLowerCase()))) return "dynamic-stock";
  if (DYNAMIC_DATASETS.traffic.some(d => lower.includes(d.toLowerCase()))) return "dynamic-traffic";
  if (DYNAMIC_DATASETS.air_quality.some(d => lower.includes(d.toLowerCase()))) return "dynamic-air";
  return "other";
}

export function LeaderboardEnhanced({
  data,
  dict,
}: {
  data: LeaderboardData;
  dict: Dictionary;
}) {
  const copy = dict.tseval;
  const trackKeys = Object.keys(data.tracks);

  // State management
  const [track, setTrack] = useState(trackKeys[0]);
  const [sortKey, setSortKey] = useState<SortKey>("mse");
  const [sortDir, setSortDir] = useState(1);
  const [viewMode, setViewMode] = useState<ViewMode>("table");
  const [searchQuery, setSearchQuery] = useState("");

  // Filter checkboxes
  const [showStatic, setShowStatic] = useState(true);
  const [showDynamicStock, setShowDynamicStock] = useState(true);
  const [showDynamicTraffic, setShowDynamicTraffic] = useState(true);
  const [showDynamicAir, setShowDynamicAir] = useState(true);
  const [showOther, setShowOther] = useState(true);

  // Display options
  const [showMedals, setShowMedals] = useState(true);
  const [showVisualization, setShowVisualization] = useState(true);
  const [highlightTop3, setHighlightTop3] = useState(true);
  const [compactMode, setCompactMode] = useState(false);
  const [showSubmissionIds, setShowSubmissionIds] = useState(false);
  const [colorByPerformance, setColorByPerformance] = useState(true);

  function toggleSort(k: SortKey) {
    if (k === sortKey) {
      setSortDir((d) => -d);
    } else {
      setSortKey(k);
      setSortDir(k === "model" ? 1 : 1);
    }
  }

  const current = data.tracks[track];
  const allDatasets = current ? Object.entries(current.datasets) : [];

  // Apply filters
  const filteredDatasets = useMemo(() => {
    return allDatasets.filter(([name]) => {
      const category = getDatasetCategory(name);
      if (category === "static" && !showStatic) return false;
      if (category === "dynamic-stock" && !showDynamicStock) return false;
      if (category === "dynamic-traffic" && !showDynamicTraffic) return false;
      if (category === "dynamic-air" && !showDynamicAir) return false;
      if (category === "other" && !showOther) return false;

      if (searchQuery && !name.toLowerCase().includes(searchQuery.toLowerCase())) {
        return false;
      }

      return true;
    });
  }, [allDatasets, showStatic, showDynamicStock, showDynamicTraffic, showDynamicAir, showOther, searchQuery]);

  // Stats
  const stats = useMemo(() => {
    const categories = {
      static: 0,
      dynamicStock: 0,
      dynamicTraffic: 0,
      dynamicAir: 0,
      other: 0,
    };

    allDatasets.forEach(([name]) => {
      const cat = getDatasetCategory(name);
      if (cat === "static") categories.static++;
      else if (cat === "dynamic-stock") categories.dynamicStock++;
      else if (cat === "dynamic-traffic") categories.dynamicTraffic++;
      else if (cat === "dynamic-air") categories.dynamicAir++;
      else categories.other++;
    });

    return categories;
  }, [allDatasets]);

  return (
    <div className="space-y-6">
      {/* Track tabs */}
      <div className="flex flex-wrap gap-2">
        {trackKeys.map((t) => {
          const active = t === track;
          return (
            <button
              key={t}
              type="button"
              onClick={() => setTrack(t)}
              className={`rounded-full border px-4 py-1.5 text-sm font-medium transition-colors ${
                active
                  ? "border-accent bg-accent text-accent-fg shadow-sm"
                  : "border-border bg-surface text-muted hover:text-ink hover:border-accent/50"
              }`}
            >
              {copy.tracks[t] ?? t}
            </button>
          );
        })}
      </div>

      {/* Filter panel */}
      <div className="rounded-2xl border border-border bg-surface p-5 space-y-5">
        <div className="flex items-center justify-between">
          <h3 className="font-serif text-lg text-ink">Filters & Options</h3>
          <button
            type="button"
            onClick={() => {
              setShowStatic(true);
              setShowDynamicStock(true);
              setShowDynamicTraffic(true);
              setShowDynamicAir(true);
              setShowOther(true);
              setSearchQuery("");
            }}
            className="text-xs text-accent hover:text-accent-hover transition-colors"
          >
            Reset all
          </button>
        </div>

        {/* Search */}
        <div>
          <label htmlFor="search" className="block text-sm font-medium text-muted mb-2">
            Search datasets
          </label>
          <input
            id="search"
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Type dataset name..."
            className="w-full rounded-lg border border-border bg-paper px-4 py-2 text-sm text-ink placeholder:text-faint focus:border-accent focus:outline-none focus:ring-2 focus:ring-accent/20"
          />
        </div>

        {/* Dataset categories */}
        <div>
          <p className="text-sm font-medium text-muted mb-3">Dataset Categories</p>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
            <Checkbox
              checked={showStatic}
              onChange={setShowStatic}
              label={`Static Datasets (${stats.static})`}
              description="Core benchmark datasets"
            />
            <Checkbox
              checked={showDynamicStock}
              onChange={setShowDynamicStock}
              label={`Stock Market (${stats.dynamicStock})`}
              description="Financial forecasting"
            />
            <Checkbox
              checked={showDynamicTraffic}
              onChange={setShowDynamicTraffic}
              label={`Traffic (${stats.dynamicTraffic})`}
              description="Spatiotemporal traffic data"
            />
            <Checkbox
              checked={showDynamicAir}
              onChange={setShowDynamicAir}
              label={`Air Quality (${stats.dynamicAir})`}
              description="Environmental monitoring"
            />
            <Checkbox
              checked={showOther}
              onChange={setShowOther}
              label={`Other (${stats.other})`}
              description="Miscellaneous datasets"
            />
          </div>
        </div>

        {/* Display options */}
        <div>
          <p className="text-sm font-medium text-muted mb-3">Display Options</p>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
            <Checkbox
              checked={showMedals}
              onChange={setShowMedals}
              label="Show medals"
              description="🥇🥈🥉 for top 3"
            />
            <Checkbox
              checked={showVisualization}
              onChange={setShowVisualization}
              label="Performance bars"
              description="Visual metric indicators"
            />
            <Checkbox
              checked={highlightTop3}
              onChange={setHighlightTop3}
              label="Highlight top 3"
              description="Accent color for leaders"
            />
            <Checkbox
              checked={compactMode}
              onChange={setCompactMode}
              label="Compact mode"
              description="Reduced spacing"
            />
            <Checkbox
              checked={showSubmissionIds}
              onChange={setShowSubmissionIds}
              label="Submission IDs"
              description="Show run identifiers"
            />
            <Checkbox
              checked={colorByPerformance}
              onChange={setColorByPerformance}
              label="Color by rank"
              description="Gradient coloring"
            />
          </div>
        </div>

        {/* View mode */}
        <div>
          <p className="text-sm font-medium text-muted mb-3">View Mode</p>
          <div className="flex gap-2">
            {(["table", "compact", "detailed"] as ViewMode[]).map((mode) => (
              <button
                key={mode}
                type="button"
                onClick={() => setViewMode(mode)}
                className={`rounded-lg border px-4 py-2 text-sm font-medium transition-colors capitalize ${
                  viewMode === mode
                    ? "border-accent bg-accent text-accent-fg"
                    : "border-border bg-paper text-muted hover:text-ink"
                }`}
              >
                {mode}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Results summary */}
      <div className="flex items-center justify-between text-sm text-muted">
        <span>
          Showing <span className="font-semibold text-accent">{filteredDatasets.length}</span> of{" "}
          <span className="font-semibold text-ink">{allDatasets.length}</span> datasets
        </span>
        <span className="text-xs text-faint">
          Primary metric: <span className="font-mono text-accent">{data.primary_metric.toUpperCase()}</span>
        </span>
      </div>

      {/* Dataset cards */}
      {filteredDatasets.length === 0 ? (
        <p className="mt-10 rounded-2xl border border-border bg-surface px-6 py-16 text-center text-muted">
          {searchQuery ? `No datasets match "${searchQuery}"` : copy.emptyTrack}
        </p>
      ) : (
        <div className="space-y-6">
          {filteredDatasets.map(([name, ds]) => (
            <DatasetCard
              key={name}
              name={name}
              ds={ds}
              copy={copy}
              sortKey={sortKey}
              sortDir={sortDir}
              onSort={toggleSort}
              viewMode={viewMode}
              showMedals={showMedals}
              showVisualization={showVisualization}
              highlightTop3={highlightTop3}
              compactMode={compactMode}
              showSubmissionIds={showSubmissionIds}
              colorByPerformance={colorByPerformance}
            />
          ))}
        </div>
      )}
    </div>
  );
}

function DatasetCard({
  name,
  ds,
  copy,
  sortKey,
  sortDir,
  onSort,
  viewMode,
  showMedals,
  showVisualization,
  highlightTop3,
  compactMode,
  showSubmissionIds,
  colorByPerformance,
}: {
  name: string;
  ds: DatasetBlock;
  copy: Dictionary["tseval"];
  sortKey: SortKey;
  sortDir: number;
  onSort: (k: SortKey) => void;
  viewMode: ViewMode;
  showMedals: boolean;
  showVisualization: boolean;
  highlightTop3: boolean;
  compactMode: boolean;
  showSubmissionIds: boolean;
  colorByPerformance: boolean;
}) {
  const horizons = useMemo(
    () => Object.keys(ds.horizons).sort((a, b) => Number(a) - Number(b)),
    [ds],
  );
  const [horizon, setHorizon] = useState(horizons[0]);
  const [expandedRow, setExpandedRow] = useState<string | null>(null);

  const rows = useMemo(() => {
    const base = ds.horizons[horizon] ?? [];
    return base.slice().sort((a, b) => {
      if (sortKey === "model") {
        return a.model.localeCompare(b.model) * sortDir;
      }
      if (sortKey === "runs") {
        return (a.n_runs - b.n_runs) * sortDir;
      }
      const av = a[sortKey];
      const bv = b[sortKey];
      if (av === null || av === undefined) return 1;
      if (bv === null || bv === undefined) return -1;
      return (av - bv) * sortDir;
    });
  }, [ds, horizon, sortKey, sortDir]);

  const maxMse = Math.max(...rows.map((r) => r.mse ?? 0)) || 1;
  const maxMae = Math.max(...rows.map((r) => r.mae ?? 0)) || 1;

  const category = getDatasetCategory(name);
  const categoryBadge = {
    static: { label: "Static", color: "bg-blue-500/10 text-blue-600 dark:text-blue-400" },
    "dynamic-stock": { label: "Stock", color: "bg-green-500/10 text-green-600 dark:text-green-400" },
    "dynamic-traffic": { label: "Traffic", color: "bg-purple-500/10 text-purple-600 dark:text-purple-400" },
    "dynamic-air": { label: "Air Quality", color: "bg-orange-500/10 text-orange-600 dark:text-orange-400" },
    other: { label: "Other", color: "bg-gray-500/10 text-gray-600 dark:text-gray-400" },
  }[category];

  const py = compactMode ? "py-2" : "py-2.5";
  const px = compactMode ? "px-4" : "px-5";

  return (
    <div className="overflow-hidden rounded-2xl border border-border bg-surface shadow-sm hover:shadow-md transition-shadow">
      <div className={`flex flex-wrap items-center justify-between gap-3 border-b border-border ${px} ${compactMode ? 'py-2.5' : 'py-3.5'}`}>
        <div className="flex items-center gap-3">
          <span className="font-serif text-lg tracking-[-0.01em] text-ink">
            {name}
          </span>
          <span className={`rounded-full px-2.5 py-0.5 text-xs font-medium ${categoryBadge.color}`}>
            {categoryBadge.label}
          </span>
        </div>
        <div className="flex flex-wrap gap-1.5">
          {horizons.map((h) => {
            const active = h === horizon;
            return (
              <button
                key={h}
                type="button"
                onClick={() => setHorizon(h)}
                className={`rounded-md border px-2.5 py-1 text-xs font-medium transition-colors ${
                  active
                    ? "border-accent bg-accent/10 text-accent"
                    : "border-border bg-paper-2 text-muted hover:text-ink hover:border-accent/30"
                }`}
              >
                {copy.horizon} {h}
              </button>
            );
          })}
        </div>
      </div>

      {viewMode === "compact" ? (
        <CompactView
          rows={rows}
          showMedals={showMedals}
          sortKey={sortKey}
          sortDir={sortDir}
          onSort={onSort}
          copy={copy}
        />
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full border-collapse [font-variant-numeric:tabular-nums]">
            <thead>
              <tr>
                <Th className="w-12">#</Th>
                <SortTh
                  label={copy.cols.model}
                  active={sortKey === "model"}
                  dir={sortDir}
                  onClick={() => onSort("model")}
                />
                <SortTh
                  label={copy.cols.mse}
                  active={sortKey === "mse"}
                  dir={sortDir}
                  onClick={() => onSort("mse")}
                />
                <SortTh
                  label={copy.cols.mae}
                  active={sortKey === "mae"}
                  dir={sortDir}
                  onClick={() => onSort("mae")}
                />
                <SortTh
                  label={copy.cols.runs}
                  active={sortKey === "runs"}
                  dir={sortDir}
                  onClick={() => onSort("runs")}
                />
                {viewMode === "detailed" && <Th>Actions</Th>}
              </tr>
            </thead>
            <tbody>
              {rows.map((r, i) => {
                const rank = sortKey === "mse" && sortDir > 0 ? r.rank ?? i + 1 : i + 1;
                const isTop3 = rank <= 3;
                const mseW = (((r.mse ?? 0) / maxMse) * 100).toFixed(0);
                const maeW = (((r.mae ?? 0) / maxMae) * 100).toFixed(0);
                const isExpanded = expandedRow === r.model;

                const rowBg = highlightTop3 && isTop3
                  ? "bg-accent/5 hover:bg-accent/10"
                  : "hover:bg-paper-2";

                const rankColor = colorByPerformance && isTop3
                  ? rank === 1 ? "text-yellow-600 dark:text-yellow-400"
                    : rank === 2 ? "text-gray-500 dark:text-gray-400"
                      : "text-amber-600 dark:text-amber-500"
                  : "text-muted";

                return (
                  <React.Fragment key={r.model}>
                    <tr
                      className={`border-b border-border last:border-0 transition-colors ${rowBg}`}
                    >
                      <td className={`${px} ${py} text-sm ${rankColor} font-semibold`}>
                        {showMedals && MEDAL[rank] ? MEDAL[rank] : rank}
                      </td>
                      <td className={`${px} ${py} text-sm font-semibold text-ink`}>
                        {r.model}
                      </td>
                      <td className={`relative ${px} ${py} font-mono text-sm text-accent`}>
                        {fmt(r.mse)}
                        {showVisualization && r.mse !== null && (
                          <span
                            aria-hidden="true"
                            className="absolute inset-x-0 bottom-0 h-px bg-accent/40"
                            style={{ width: `${mseW}%` }}
                          />
                        )}
                      </td>
                      <td className={`relative ${px} ${py} font-mono text-sm text-muted`}>
                        {fmt(r.mae)}
                        {showVisualization && r.mae !== null && (
                          <span
                            aria-hidden="true"
                            className="absolute inset-x-0 bottom-0 h-px bg-muted/30"
                            style={{ width: `${maeW}%` }}
                          />
                        )}
                      </td>
                      <td className={`${px} ${py} text-sm text-muted`}>
                        <span className="inline-flex items-center gap-1">
                          {r.n_runs}
                          {r.n_runs > 1 && <span className="text-xs text-faint">×</span>}
                        </span>
                      </td>
                      {viewMode === "detailed" && (
                        <td className={`${px} ${py} text-sm`}>
                          <button
                            type="button"
                            onClick={() => setExpandedRow(isExpanded ? null : r.model)}
                            className="text-xs text-accent hover:text-accent-hover transition-colors"
                          >
                            {isExpanded ? "Hide" : "Details"}
                          </button>
                        </td>
                      )}
                    </tr>
                    {viewMode === "detailed" && isExpanded && (
                      <tr className="border-b border-border bg-paper-2">
                        <td colSpan={6} className={`${px} ${py} text-sm text-muted space-y-2`}>
                          <div>
                            <span className="font-semibold text-ink">Model:</span> {r.model}
                          </div>
                          <div>
                            <span className="font-semibold text-ink">Runs:</span> {r.n_runs}
                          </div>
                          {showSubmissionIds && r.submission_ids.length > 0 && (
                            <div>
                              <span className="font-semibold text-ink">Submission IDs:</span>
                              <ul className="mt-1 space-y-0.5 font-mono text-xs text-faint">
                                {r.submission_ids.map((id) => (
                                  <li key={id}>• {id}</li>
                                ))}
                              </ul>
                            </div>
                          )}
                        </td>
                      </tr>
                    )}
                  </React.Fragment>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function CompactView({
  rows,
  showMedals,
  sortKey,
  sortDir,
  onSort,
  copy,
}: {
  rows: LeaderRow[];
  showMedals: boolean;
  sortKey: SortKey;
  sortDir: number;
  onSort: (k: SortKey) => void;
  copy: Dictionary["tseval"];
}) {
  return (
    <div className="p-4 space-y-2">
      {rows.map((r, i) => {
        const rank = sortKey === "mse" && sortDir > 0 ? r.rank ?? i + 1 : i + 1;
        return (
          <div
            key={r.model}
            className="flex items-center justify-between rounded-lg border border-border bg-surface px-4 py-2.5 hover:bg-paper-2 transition-colors"
          >
            <div className="flex items-center gap-3">
              <span className="text-sm font-semibold text-muted w-8">
                {showMedals && MEDAL[rank] ? MEDAL[rank] : `#${rank}`}
              </span>
              <span className="text-sm font-semibold text-ink">{r.model}</span>
            </div>
            <div className="flex items-center gap-4 text-sm">
              <span className="font-mono text-accent">{fmt(r.mse)}</span>
              <span className="font-mono text-muted">{fmt(r.mae)}</span>
              <span className="text-faint">{r.n_runs}×</span>
            </div>
          </div>
        );
      })}
    </div>
  );
}

function Checkbox({
  checked,
  onChange,
  label,
  description,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  label: string;
  description?: string;
}) {
  return (
    <label className="flex items-start gap-3 cursor-pointer group">
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        className="mt-0.5 h-4 w-4 rounded border-border text-accent focus:ring-2 focus:ring-accent/20 focus:ring-offset-0 cursor-pointer"
      />
      <div className="flex-1 min-w-0">
        <span className="block text-sm font-medium text-ink group-hover:text-accent transition-colors">
          {label}
        </span>
        {description && (
          <span className="block text-xs text-faint mt-0.5">{description}</span>
        )}
      </div>
    </label>
  );
}

function Th({
  children,
  className = "",
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <th
      className={`border-b border-border px-5 py-2.5 text-left text-[0.7rem] font-medium uppercase tracking-[0.04em] text-faint ${className}`}
    >
      {children}
    </th>
  );
}

function SortTh({
  label,
  active,
  dir,
  onClick,
}: {
  label: string;
  active: boolean;
  dir: number;
  onClick: () => void;
}) {
  return (
    <th className="border-b border-border px-5 py-2.5 text-left text-[0.7rem] font-medium uppercase tracking-[0.04em]">
      <button
        type="button"
        onClick={onClick}
        className={`inline-flex items-center gap-1 uppercase tracking-[0.04em] transition-colors ${
          active ? "text-accent font-semibold" : "text-faint hover:text-ink"
        }`}
      >
        {label}
        <span className="text-xs">
          {active ? (dir > 0 ? "↑" : "↓") : "⇅"}
        </span>
      </button>
    </th>
  );
}
