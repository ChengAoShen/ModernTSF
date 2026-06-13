"use client";

import { useMemo, useState } from "react";
import React from "react";
import type { Dictionary } from "@/lib/dictionaries";


// ---------------------------------------------------------------------------
// TS-Eval: Hundreds Models Campaign - 百模大战
// Horizontal dataset navigation for 100+ models per dataset
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

type SortKey = "mse" | "mae" | "rmse" | "corr" | "fit_time" | "inference_time" | "model" | "runs";
type ViewMode = "table" | "compact" | "detailed";
type MainCategory = "common_static" | "realtime";

// Two-tier classification
const CATEGORY_STRUCTURE = {
  common_static: {
    label: "Common Static Dataset",
    label_zh: "通用静态数据集",
    tracks: ["time_series", "spatiotemporal", "covariate"],
    datasets: ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Traffic", "Solar", "Electricity", "Weather"]
  },
  realtime: {
    label: "Real-Time Dataset",
    label_zh: "实时数据集",
    tracks: ["stock", "traffic_rt", "air_quality"],
    datasets: []
  }
};

const TRACK_META = {
  time_series: { label: "Time Series", icon: "📈", position: "left" },
  spatiotemporal: { label: "Spatial Temporal", icon: "🗺️", position: "center" },
  covariate: { label: "Covariate", icon: "🔗", position: "right" },
  stock: { label: "Stock-HS300", icon: "💰", position: "left" },
  traffic_rt: { label: "Traffic", icon: "🚗", position: "center" },
  air_quality: { label: "Air Quality", icon: "🌫️", position: "right" }
};

function fmt(v: number | null | undefined): string {
  return v === null || v === undefined ? "—" : v.toFixed(4);
}

function getMainCategory(trackName: string): MainCategory {
  if (CATEGORY_STRUCTURE.common_static.tracks.includes(trackName)) {
    return "common_static";
  }
  return "realtime";
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

  // Main category selection
  const [mainCategory, setMainCategory] = useState<MainCategory>("common_static");
  
  const availableTracks = useMemo(() => {
    return trackKeys.filter(t => getMainCategory(t) === mainCategory);
  }, [trackKeys, mainCategory]);
  
  const [track, setTrack] = useState(availableTracks[0] || trackKeys[0]);

  React.useEffect(() => {
    if (!availableTracks.includes(track)) {
      setTrack(availableTracks[0] || trackKeys[0]);
    }
  }, [mainCategory, availableTracks, track, trackKeys]);

  // State management
  const [sortKey, setSortKey] = useState<SortKey>("mse");
  const [sortDir, setSortDir] = useState(1);
  const [viewMode, setViewMode] = useState<ViewMode>("table");
  const [searchQuery, setSearchQuery] = useState("");
  const [modelSearchQuery, setModelSearchQuery] = useState("");

  // Dataset filters (8 core)
  const [showETTh1, setShowETTh1] = useState(true);
  const [showETTh2, setShowETTh2] = useState(true);
  const [showETTm1, setShowETTm1] = useState(true);
  const [showETTm2, setShowETTm2] = useState(true);
  const [showTraffic, setShowTraffic] = useState(true);
  const [showSolar, setShowSolar] = useState(true);
  const [showElectricity, setShowElectricity] = useState(true);
  const [showWeather, setShowWeather] = useState(true);

  // Display options
  const [showMedals, setShowMedals] = useState(true);
  const [showVisualization, setShowVisualization] = useState(true);
  const [highlightTop3, setHighlightTop3] = useState(true);
  const [compactMode, setCompactMode] = useState(false);
  const [showSubmissionIds, setShowSubmissionIds] = useState(false);
  const [colorByPerformance, setColorByPerformance] = useState(true);

  // Metrics selection (not Matrix!)
  const [showMSE, setShowMSE] = useState(true);
  const [showMAE, setShowMAE] = useState(true);
  const [showRMSE, setShowRMSE] = useState(true);
  const [showCorr, setShowCorr] = useState(true);
  const [showFitTime, setShowFitTime] = useState(true);
  const [showInferenceTime, setShowInferenceTime] = useState(true);
  const [showRuns, setShowRuns] = useState(true);

  // Model type filters
  const [showTimeSeriesModels, setShowTimeSeriesModels] = useState(true);
  const [showSpatialTemporalModels, setShowSpatialTemporalModels] = useState(true);

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
      const lower = name.toLowerCase();
      
      if (mainCategory === "common_static") {
        if (lower.includes("etth1") && !showETTh1) return false;
        if (lower.includes("etth2") && !showETTh2) return false;
        if (lower.includes("ettm1") && !showETTm1) return false;
        if (lower.includes("ettm2") && !showETTm2) return false;
        if (lower.includes("traffic") && !showTraffic) return false;
        if (lower.includes("solar") && !showSolar) return false;
        if (lower.includes("electricity") && !showElectricity) return false;
        if (lower.includes("weather") && !showWeather) return false;
      }

      if (searchQuery && !name.toLowerCase().includes(searchQuery.toLowerCase())) {
        return false;
      }

      return true;
    });
  }, [allDatasets, mainCategory, showETTh1, showETTh2, showETTm1, showETTm2, 
      showTraffic, showSolar, showElectricity, showWeather, searchQuery]);

  // Current dataset navigation (horizontal)
  const [currentDatasetIndex, setCurrentDatasetIndex] = useState(0);
  const currentDataset = filteredDatasets[currentDatasetIndex];

  // Reset index when datasets change
  React.useEffect(() => {
    setCurrentDatasetIndex(0);
  }, [filteredDatasets.length, track, mainCategory]);

  const goToPrevDataset = () => {
    setCurrentDatasetIndex((prev) => Math.max(0, prev - 1));
  };

  const goToNextDataset = () => {
    setCurrentDatasetIndex((prev) => Math.min(filteredDatasets.length - 1, prev + 1));
  };

  const categoryInfo = CATEGORY_STRUCTURE[mainCategory];

  return (
    <div className="space-y-6">
      {/* Title */}
      <div className="text-center">
        <h2 className="font-serif text-2xl text-ink mb-2">
          TS-Eval: Hundreds Models Campaign
        </h2>
        <p className="text-sm text-muted">
          百模大战 · Benchmarking at Scale
        </p>
        <p className="text-xs text-faint mt-1">
          {data.n_submissions} submissions · {filteredDatasets.length} datasets
        </p>
      </div>

      {/* Main category tabs */}
      <div className="flex justify-center gap-3">
        {(Object.keys(CATEGORY_STRUCTURE) as MainCategory[]).map((cat) => {
          const active = cat === mainCategory;
          const info = CATEGORY_STRUCTURE[cat];
          return (
            <button
              key={cat}
              type="button"
              onClick={() => setMainCategory(cat)}
              className={`rounded-xl border px-6 py-3 text-sm font-semibold transition-all ${
                active
                  ? "border-accent bg-accent text-accent-fg shadow-lg scale-105"
                  : "border-border bg-surface text-muted hover:text-ink hover:border-accent/50"
              }`}
            >
              <div className="text-base">{info.label}</div>
              <div className="text-xs opacity-75 mt-0.5">{info.label_zh}</div>
            </button>
          );
        })}
      </div>

      {/* Track tabs (three columns) */}
      <div className="grid grid-cols-3 gap-3">
        {availableTracks.map((t) => {
          const active = t === track;
          const meta = TRACK_META[t as keyof typeof TRACK_META];
          return (
            <button
              key={t}
              type="button"
              onClick={() => setTrack(t)}
              className={`rounded-lg border px-4 py-2.5 text-sm font-medium transition-colors ${
                active
                  ? "border-accent bg-accent/10 text-accent"
                  : "border-border bg-surface text-muted hover:text-ink"
              }`}
            >
              <div className="flex items-center justify-center gap-2">
                <span>{meta?.icon}</span>
                <span>{meta?.label || t}</span>
              </div>
            </button>
          );
        })}
      </div>

      {/* Filter panel */}
      <div className="rounded-2xl border border-border bg-surface p-5 space-y-5">
        <div className="flex items-center justify-between">
          <h3 className="font-serif text-lg text-ink">{copy.filters.title}</h3>
          <button
            type="button"
            onClick={() => {
              setShowETTh1(true); setShowETTh2(true); setShowETTm1(true); setShowETTm2(true);
              setShowTraffic(true); setShowSolar(true); setShowElectricity(true); setShowWeather(true);
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
            {copy.filters.title.replace(" & Options", "")}
          </label>
          <input
            id="search"
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder={copy.filters.searchPlaceholder}
            className="w-full rounded-lg border border-border bg-paper px-4 py-2 text-sm text-ink placeholder:text-faint focus:border-accent focus:outline-none focus:ring-2 focus:ring-accent/20"
          />
        </div>

        {/* Dataset Categories */}
        {mainCategory === "common_static" && (
          <div>
            <p className="text-sm font-medium text-muted mb-3">{copy.filters.datasetCategories}</p>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              <Checkbox checked={showETTh1} onChange={setShowETTh1} label={copy.datasets.etth1} description={copy.datasetDescriptions.hourly1} />
              <Checkbox checked={showETTh2} onChange={setShowETTh2} label={copy.datasets.etth2} description={copy.datasetDescriptions.hourly2} />
              <Checkbox checked={showETTm1} onChange={setShowETTm1} label={copy.datasets.ettm1} description={copy.datasetDescriptions.minute1} />
              <Checkbox checked={showETTm2} onChange={setShowETTm2} label={copy.datasets.ettm2} description={copy.datasetDescriptions.minute2} />
              <Checkbox checked={showTraffic} onChange={setShowTraffic} label={copy.datasets.traffic} description={copy.datasetDescriptions.roadTraffic} />
              <Checkbox checked={showSolar} onChange={setShowSolar} label={copy.datasets.solar} description={copy.datasetDescriptions.solarPower} />
              <Checkbox checked={showElectricity} onChange={setShowElectricity} label={copy.datasets.electricity} description={copy.datasetDescriptions.powerUsage} />
              <Checkbox checked={showWeather} onChange={setShowWeather} label={copy.datasets.weather} description={copy.datasetDescriptions.weatherData} />
            </div>
          </div>
        )}

        {/* Display */}
        <div>
          <p className="text-sm font-medium text-muted mb-3">{copy.filters.display}</p>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
            <Checkbox checked={showMedals} onChange={setShowMedals} label={copy.displayOptions.showMedals} description={copy.displayOptions.showMedalsDesc} />
            <Checkbox checked={showVisualization} onChange={setShowVisualization} label={copy.displayOptions.performanceBars} description={copy.displayOptions.performanceBarsDesc} />
            <Checkbox checked={highlightTop3} onChange={setHighlightTop3} label={copy.displayOptions.highlightTop3} description={copy.displayOptions.highlightTop3Desc} />
            <Checkbox checked={compactMode} onChange={setCompactMode} label={copy.displayOptions.compactMode} description={copy.displayOptions.compactModeDesc} />
            <Checkbox checked={showSubmissionIds} onChange={setShowSubmissionIds} label={copy.displayOptions.submissionIds} description={copy.displayOptions.submissionIdsDesc} />
            <Checkbox checked={colorByPerformance} onChange={setColorByPerformance} label={copy.displayOptions.colorByRank} description={copy.displayOptions.colorByRankDesc} />
          </div>
        </div>

        {/* Metrics */}
        <div>
          <p className="text-sm font-medium text-muted mb-3">{copy.filters.metrics}</p>
          <div className="grid grid-cols-3 gap-3">
            <Checkbox checked={showMSE} onChange={setShowMSE} label={copy.metricsOptions.mse} description={copy.metricsOptions.mseDesc} />
            <Checkbox checked={showMAE} onChange={setShowMAE} label={copy.metricsOptions.mae} description={copy.metricsOptions.maeDesc} />
            <Checkbox checked={showRMSE} onChange={setShowRMSE} label={copy.metricsOptions.rmse} description={copy.metricsOptions.rmseDesc} />
            <Checkbox checked={showCorr} onChange={setShowCorr} label={copy.metricsOptions.corr} description={copy.metricsOptions.corrDesc} />
            <Checkbox checked={showFitTime} onChange={setShowFitTime} label={copy.metricsOptions.fitTime} description={copy.metricsOptions.fitTimeDesc} />
            <Checkbox checked={showInferenceTime} onChange={setShowInferenceTime} label={copy.metricsOptions.inferenceTime} description={copy.metricsOptions.inferenceTimeDesc} />
            <Checkbox checked={showRuns} onChange={setShowRuns} label={copy.metricsOptions.runs} description={copy.metricsOptions.runsDesc} />
          </div>
        </div>

        {/* Model Types */}
        <div>
          <p className="text-sm font-medium text-muted mb-3">{copy.modelTypes.title}</p>
          <div className="grid grid-cols-2 gap-3">
            <Checkbox 
              checked={showTimeSeriesModels} 
              onChange={setShowTimeSeriesModels} 
              label={copy.modelTypes.timeSeries} 
              description={copy.modelTypes.timeSeriesDesc} 
            />
            <Checkbox 
              checked={showSpatialTemporalModels} 
              onChange={setShowSpatialTemporalModels} 
              label={copy.modelTypes.spatialTemporal} 
              description={copy.modelTypes.spatialTemporalDesc} 
            />
          </div>
        </div>

        {/* Options */}
        <div>
          <p className="text-sm font-medium text-muted mb-3">{copy.filters.options}</p>
          <div className="flex gap-2">
            {(["table", "compact", "detailed"] as ViewMode[]).map((mode) => (
              <button
                key={mode}
                type="button"
                onClick={() => setViewMode(mode)}
                className={`rounded-lg border px-4 py-2 text-sm font-medium transition-colors ${
                  viewMode === mode
                    ? "border-accent bg-accent text-accent-fg"
                    : "border-border bg-paper text-muted hover:text-ink"
                }`}
              >
                {copy.viewModes[mode]}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Horizontal dataset navigation with buttons */}
      {filteredDatasets.length === 0 ? (
        <p className="mt-10 rounded-2xl border border-border bg-surface px-6 py-16 text-center text-muted">
          {searchQuery ? `${copy.status.noMatch} "${searchQuery}"` : copy.status.noDatasets}
        </p>
      ) : (
        <div className="space-y-4">
          {/* Navigation controls */}
          <div className="flex items-center justify-between">
            <button
              type="button"
              onClick={goToPrevDataset}
              disabled={currentDatasetIndex === 0}
              className="flex items-center gap-2 rounded-lg border border-border bg-surface px-4 py-2 text-sm font-medium text-ink transition-colors hover:bg-paper-2 disabled:opacity-40 disabled:cursor-not-allowed"
            >
              <span>←</span>
              <span>{copy.navigation.previous}</span>
            </button>

            <div className="text-center">
              <p className="text-sm font-semibold text-ink">
                {currentDataset ? currentDataset[0] : ""}
              </p>
              <p className="text-xs text-faint">
                {copy.navigation.dataset} {currentDatasetIndex + 1} {copy.navigation.of} {filteredDatasets.length}
              </p>
            </div>

            <button
              type="button"
              onClick={goToNextDataset}
              disabled={currentDatasetIndex === filteredDatasets.length - 1}
              className="flex items-center gap-2 rounded-lg border border-border bg-surface px-4 py-2 text-sm font-medium text-ink transition-colors hover:bg-paper-2 disabled:opacity-40 disabled:cursor-not-allowed"
            >
              <span>{copy.navigation.next}</span>
              <span>→</span>
            </button>
          </div>

          {/* Current dataset card */}
          {currentDataset && (
            <DatasetCard
              name={currentDataset[0]}
              ds={currentDataset[1]}
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
              showMSE={showMSE}
              showMAE={showMAE}
              showRMSE={showRMSE}
              showCorr={showCorr}
              showFitTime={showFitTime}
              showInferenceTime={showInferenceTime}
              showRuns={showRuns}
              showTimeSeriesModels={showTimeSeriesModels}
              showSpatialTemporalModels={showSpatialTemporalModels}
              modelSearchQuery={modelSearchQuery}
            />
          )}
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
  showMSE,
  showMAE,
  showRMSE,
  showCorr,
  showFitTime,
  showInferenceTime,
  showRuns,
  showTimeSeriesModels,
  showSpatialTemporalModels,
  modelSearchQuery,
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
  showMSE: boolean;
  showMAE: boolean;
  showRMSE: boolean;
  showCorr: boolean;
  showFitTime: boolean;
  showInferenceTime: boolean;
  showRuns: boolean;
  showTimeSeriesModels: boolean;
  showSpatialTemporalModels: boolean;
  modelSearchQuery: string;
}) {
  const horizons = useMemo(
    () => Object.keys(ds.horizons).sort((a, b) => Number(a) - Number(b)),
    [ds],
  );
  const [horizon, setHorizon] = useState(horizons[0]);
  const [expandedRow, setExpandedRow] = useState<string | null>(null);

  const rows = useMemo(() => {
    let base = ds.horizons[horizon] ?? [];
    
    // Filter by model type and search query
    base = base.filter((row) => {
      const modelName = row.model.toLowerCase();
      
      // Model search (fuzzy match)
      if (modelSearchQuery) {
        const query = modelSearchQuery.toLowerCase();
        if (!modelName.includes(query)) {
          return false;
        }
      }
      
      // Detect model type from submission_ids or model patterns
      const isTS = row.submission_ids?.some((id: string) => id.includes('_ts_')) || 
                   !row.submission_ids?.some((id: string) => id.includes('_st_'));
      const isST = row.submission_ids?.some((id: string) => id.includes('_st_'));
      
      if (isST && !showSpatialTemporalModels) return false;
      if (isTS && !showTimeSeriesModels) return false;
      
      return true;
    });
    
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
  }, [ds, horizon, sortKey, sortDir, showTimeSeriesModels, showSpatialTemporalModels, modelSearchQuery]);

  const maxMse = Math.max(...rows.map((r) => r.mse ?? 0)) || 1;
  const maxMae = Math.max(...rows.map((r) => r.mae ?? 0)) || 1;

  const py = compactMode ? "py-2" : "py-2.5";
  const px = compactMode ? "px-4" : "px-5";

  return (
    <div className="overflow-hidden rounded-2xl border border-border bg-surface shadow-sm">
      {/* Model Search */}
      <div className={`border-b border-border ${px} ${compactMode ? 'py-2' : 'py-3'} bg-paper-2/50`}>
        <input
          type="text"
          value={modelSearchQuery}
          onChange={(e) => setModelSearchQuery(e.target.value)}
          placeholder="Search models (e.g., DLinear, LSTM, Transformer)..."
          className="w-full rounded-lg border border-border bg-paper px-3 py-2 text-sm text-ink placeholder:text-faint focus:border-accent focus:outline-none focus:ring-2 focus:ring-accent/20"
        />
      </div>
      
      <div className={`flex flex-wrap items-center justify-between gap-3 border-b border-border ${px} ${compactMode ? 'py-2.5' : 'py-3.5'}`}>
        <div className="flex items-center gap-3 flex-wrap">
          <span className="font-serif text-lg tracking-[-0.01em] text-ink">
            {name}
          </span>
          <span className="text-xs text-accent font-semibold">
            {rows.length} models
          </span>
          {modelSearchQuery && (
            <span className="text-xs text-faint">
              Filtered by: "{modelSearchQuery}"
            </span>
          )}
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
                seq_len: 20 | {copy.horizon} {h}
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
          showMSE={showMSE}
          showMAE={showMAE}
          showRuns={showRuns}
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
                {showMSE && (
                  <SortTh
                    label="MSE"
                    active={sortKey === "mse"}
                    dir={sortDir}
                    onClick={() => onSort("mse")}
                  />
                )}
                {showMAE && (
                  <SortTh
                    label="MAE"
                    active={sortKey === "mae"}
                    dir={sortDir}
                    onClick={() => onSort("mae")}
                  />
                )}
                {showRMSE && (
                  <SortTh
                    label="RMSE"
                    active={sortKey === "rmse"}
                    dir={sortDir}
                    onClick={() => onSort("rmse")}
                  />
                )}
                {showCorr && (
                  <SortTh
                    label="Corr"
                    active={sortKey === "corr"}
                    dir={sortDir}
                    onClick={() => onSort("corr")}
                  />
                )}
                {showFitTime && (
                  <SortTh
                    label="Fit Time"
                    active={sortKey === "fit_time"}
                    dir={sortDir}
                    onClick={() => onSort("fit_time")}
                  />
                )}
                {showInferenceTime && (
                  <SortTh
                    label="Inf Time"
                    active={sortKey === "inference_time"}
                    dir={sortDir}
                    onClick={() => onSort("inference_time")}
                  />
                )}
                {showRuns && (
                  <SortTh
                    label={copy.cols.runs}
                    active={sortKey === "runs"}
                    dir={sortDir}
                    onClick={() => onSort("runs")}
                  />
                )}
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
                    <tr className={`border-b border-border last:border-0 transition-colors ${rowBg}`}>
                      <td className={`${px} ${py} text-sm ${rankColor} font-semibold`}>
                        {showMedals && MEDAL[rank] ? MEDAL[rank] : rank}
                      </td>
                      <td className={`${px} ${py} text-sm font-semibold text-ink`}>
                        {r.model}
                      </td>
                      {showMSE && (
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
                      )}
                      {showMAE && (
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
                      )}
                      {showRMSE && (
                        <td className={`${px} ${py} font-mono text-sm text-muted`}>
                          {fmt((r as any).rmse)}
                        </td>
                      )}
                      {showCorr && (
                        <td className={`${px} ${py} font-mono text-sm text-muted`}>
                          {fmt((r as any).corr)}
                        </td>
                      )}
                      {showFitTime && (
                        <td className={`${px} ${py} font-mono text-sm text-faint`}>
                          {((r as any).fit_time)?.toFixed(2)}s
                        </td>
                      )}
                      {showInferenceTime && (
                        <td className={`${px} ${py} font-mono text-sm text-faint`}>
                          {((r as any).inference_time)?.toFixed(3)}s
                        </td>
                      )}
                      {showRuns && (
                        <td className={`${px} ${py} text-sm text-muted`}>
                          <span className="inline-flex items-center gap-1">
                            {r.n_runs}
                            {r.n_runs > 1 && <span className="text-xs text-faint">×</span>}
                          </span>
                        </td>
                      )}
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
  showMSE,
  showMAE,
  showRuns,
}: {
  rows: LeaderRow[];
  showMedals: boolean;
  sortKey: SortKey;
  sortDir: number;
  onSort: (k: SortKey) => void;
  copy: Dictionary["tseval"];
  showMSE: boolean;
  showMAE: boolean;
  showRMSE: boolean;
  showCorr: boolean;
  showFitTime: boolean;
  showInferenceTime: boolean;
  showRuns: boolean;
  showTimeSeriesModels: boolean;
  showSpatialTemporalModels: boolean;
  modelSearchQuery: string;
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
              {showMSE && <span className="font-mono text-accent">{fmt(r.mse)}</span>}
              {showMAE && <span className="font-mono text-muted">{fmt(r.mae)}</span>}
              {showRuns && <span className="text-faint">{r.n_runs}×</span>}
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
