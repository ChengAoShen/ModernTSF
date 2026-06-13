import type { Locale } from "@/lib/i18n";

/**
 * Horizontal bar chart comparing time-to-result, in the style of the uv
 * benchmark chart: our row first (tiny bar, bold), slower alternatives below.
 * Pure server component — theme-aware via the design tokens in globals.css.
 */

const DOMAIN_HOURS = 168; // one week — the slowest row
const TRACK_MAX_PCT = 86; // longest bar stops here so its value label fits

interface ChartRow {
  label: string;
  hours: number;
  value: Record<Locale, string>;
  em?: boolean;
}

const ROWS: ChartRow[] = [
  {
    label: "ModernTSF",
    hours: 2.5,
    value: { en: "2–3 hrs", zh: "2–3 小时" },
    em: true,
  },
  {
    label: "Agent + Human",
    hours: 24,
    value: { en: "1 day+", zh: "1 天以上" },
  },
  {
    label: "Only Human",
    hours: 168,
    value: { en: "1 week+", zh: "1 周以上" },
  },
];

const TICKS: { hours: number; label: Record<Locale, string> }[] = [
  { hours: 0, label: { en: "0", zh: "0" } },
  { hours: 72, label: { en: "3 days", zh: "3 天" } },
  { hours: 144, label: { en: "6 days", zh: "6 天" } },
];

const CAPTION: Record<Locale, string> = {
  en: "Typical time from idea to complete results (early-user reports)",
  zh: "从出现 idea 到产出完整结果的典型耗时（早期使用者反馈）",
};

const ARIA: Record<Locale, string> = {
  en: "Bar chart: time from idea to complete results — ModernTSF 2 to 3 hours, agent plus human over 1 day, human only over 1 week.",
  zh: "条形图：从 idea 到完整结果的耗时——ModernTSF 2 到 3 小时，Agent 加人工 1 天以上，纯人工 1 周以上。",
};

const pct = (hours: number) => (hours / DOMAIN_HOURS) * TRACK_MAX_PCT;

export function IdeaTimeChart({ locale = "en" }: { locale?: Locale }) {
  return (
    <figure className="my-10 rounded-2xl border border-border bg-paper-2 px-5 py-6 sm:px-8 sm:py-7">
      <div className="flex flex-col gap-3" role="img" aria-label={ARIA[locale]}>
        {ROWS.map((row) => (
          <div key={row.label} className="flex items-center gap-3 sm:gap-4">
            <span
              className={`w-28 shrink-0 text-right font-mono text-[0.8rem] sm:w-32 sm:text-sm ${
                row.em ? "font-semibold text-ink" : "text-muted"
              }`}
            >
              {row.label}
            </span>
            <div className="flex min-w-0 flex-1 items-center gap-2.5">
              <div
                className="h-[1.15rem] shrink-0 rounded-[3px] bg-accent"
                style={{ width: `${pct(row.hours)}%`, minWidth: "0.3rem" }}
              />
              <span
                className={`whitespace-nowrap font-mono text-[0.8rem] sm:text-sm ${
                  row.em ? "font-semibold text-ink" : "text-muted"
                }`}
              >
                {row.value[locale]}
              </span>
            </div>
          </div>
        ))}

        {/* axis */}
        <div className="flex items-center gap-3 sm:gap-4">
          <span className="w-28 shrink-0 sm:w-32" />
          <div className="relative h-6 flex-1 border-t border-border-strong">
            {TICKS.map((tick) => (
              <span
                key={tick.hours}
                className={`absolute top-1.5 font-mono text-[0.72rem] text-faint ${
                  tick.hours === 0 ? "" : "-translate-x-1/2"
                }`}
                style={{ left: `${pct(tick.hours)}%` }}
              >
                {tick.label[locale]}
              </span>
            ))}
          </div>
        </div>
      </div>
      <figcaption className="mt-4 text-center text-[0.82rem] text-faint">
        {CAPTION[locale]}
      </figcaption>
    </figure>
  );
}
