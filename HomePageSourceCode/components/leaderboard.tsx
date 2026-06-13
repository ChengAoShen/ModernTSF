"use client";

import { useMemo, useState } from "react";
import type { Dictionary } from "@/lib/dictionaries";

// ---------------------------------------------------------------------------
//  Shape of leaderboard.json (produced by `tsf leaderboard-build`).
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

type SortKey = "mse" | "mae";

function fmt(v: number | null | undefined): string {
  return v === null || v === undefined ? "—" : v.toFixed(4);
}

export function Leaderboard({
  data,
  dict,
}: {
  data: LeaderboardData;
  dict: Dictionary;
}) {
  const copy = dict.tseval;
  const trackKeys = Object.keys(data.tracks);
  const [track, setTrack] = useState(trackKeys[0]);
  const [sortKey, setSortKey] = useState<SortKey>("mse");
  const [sortDir, setSortDir] = useState(1); // 1 = ascending (lower is better)

  function toggleSort(k: SortKey) {
    if (k === sortKey) {
      setSortDir((d) => -d);
    } else {
      setSortKey(k);
      setSortDir(1);
    }
  }

  const current = data.tracks[track];
  const datasets = current ? Object.entries(current.datasets) : [];

  return (
    <>
      {/* Track tabs */}
      <div className="flex flex-wrap gap-2">
        {trackKeys.map((t) => {
          const active = t === track;
          return (
            <button
              key={t}
              type="button"
              onClick={() => setTrack(t)}
              className={`rounded-full border px-4 py-1.5 text-sm transition-colors ${
                active
                  ? "border-accent bg-accent text-accent-fg"
                  : "border-border bg-surface text-muted hover:text-ink"
              }`}
            >
              {copy.tracks[t] ?? t}
            </button>
          );
        })}
      </div>

      {datasets.length === 0 ? (
        <p className="mt-10 rounded-2xl border border-border bg-surface px-6 py-16 text-center text-muted">
          {copy.emptyTrack}
        </p>
      ) : (
        <div className="mt-6 space-y-6">
          {datasets.map(([name, ds]) => (
            <DatasetCard
              key={name}
              name={name}
              ds={ds}
              copy={copy}
              sortKey={sortKey}
              sortDir={sortDir}
              onSort={toggleSort}
            />
          ))}
        </div>
      )}
    </>
  );
}

function DatasetCard({
  name,
  ds,
  copy,
  sortKey,
  sortDir,
  onSort,
}: {
  name: string;
  ds: DatasetBlock;
  copy: Dictionary["tseval"];
  sortKey: SortKey;
  sortDir: number;
  onSort: (k: SortKey) => void;
}) {
  const horizons = useMemo(
    () => Object.keys(ds.horizons).sort((a, b) => Number(a) - Number(b)),
    [ds],
  );
  const [horizon, setHorizon] = useState(horizons[0]);

  const rows = useMemo(() => {
    const base = ds.horizons[horizon] ?? [];
    return base.slice().sort((a, b) => {
      const av = a[sortKey];
      const bv = b[sortKey];
      if (av === null || av === undefined) return 1;
      if (bv === null || bv === undefined) return -1;
      return (av - bv) * sortDir;
    });
  }, [ds, horizon, sortKey, sortDir]);

  const max = Math.max(...rows.map((r) => r.mse ?? 0)) || 1;

  return (
    <div className="overflow-hidden rounded-2xl border border-border bg-surface">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border px-5 py-3.5">
        <span className="font-serif text-lg tracking-[-0.01em] text-ink">
          {name}
        </span>
        <div className="flex flex-wrap gap-1.5">
          {horizons.map((h) => {
            const active = h === horizon;
            return (
              <button
                key={h}
                type="button"
                onClick={() => setHorizon(h)}
                className={`rounded-md border px-2.5 py-1 text-xs transition-colors ${
                  active
                    ? "border-accent text-accent"
                    : "border-border bg-paper-2 text-muted hover:text-ink"
                }`}
              >
                {copy.horizon} {h}
              </button>
            );
          })}
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full border-collapse [font-variant-numeric:tabular-nums]">
          <thead>
            <tr>
              <Th className="w-12">#</Th>
              <Th>{copy.cols.model}</Th>
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
              <Th>{copy.cols.runs}</Th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => {
              const rank =
                sortKey === "mse" && sortDir > 0 ? r.rank ?? i + 1 : i + 1;
              const w = (((r.mse ?? 0) / max) * 100).toFixed(0);
              return (
                <tr
                  key={r.model}
                  className="border-b border-border last:border-0 transition-colors hover:bg-paper-2"
                >
                  <td className="px-5 py-2.5 text-sm text-muted">
                    {MEDAL[rank] ?? rank}
                  </td>
                  <td className="px-5 py-2.5 text-sm font-semibold text-ink">
                    {r.model}
                  </td>
                  <td className="relative px-5 py-2.5 font-mono text-sm text-accent">
                    {fmt(r.mse)}
                    <span
                      aria-hidden="true"
                      className="absolute inset-x-0 bottom-0 h-px bg-accent/40"
                      style={{ width: `${w}%` }}
                    />
                  </td>
                  <td className="px-5 py-2.5 font-mono text-sm text-muted">
                    {fmt(r.mae)}
                  </td>
                  <td className="px-5 py-2.5 text-sm text-muted">{r.n_runs}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
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
          active ? "text-accent" : "text-faint hover:text-ink"
        }`}
      >
        {label}
        {active ? (dir > 0 ? "↑" : "↓") : ""}
      </button>
    </th>
  );
}
