import { LeaderboardEnhanced, type LeaderboardData } from "./leaderboard-enhanced-v3";
import { getDictionary } from "@/lib/dictionaries";
import type { Locale } from "@/lib/i18n";
import raw from "@/content/tseval/leaderboard.json";

const data = raw as LeaderboardData;

export function TSEvalPage({ locale }: { locale: Locale }) {
  const dict = getDictionary(locale);
  const copy = dict.tseval;

  // `generated_at` is rendered in a stable locale so the static HTML is
  // deterministic (no hydration mismatch between server and client).
  const updated = new Date(data.generated_at).toISOString().slice(0, 16).replace("T", " ");

  return (
    <>
      <section className="border-b border-border">
        <div className="mx-auto max-w-6xl px-5 py-20 sm:px-8 sm:py-24">
          <h1 className="font-serif text-4xl tracking-[-0.02em] text-ink sm:text-6xl">
            {copy.title}
          </h1>
          <p className="mt-5 max-w-2xl text-lg leading-relaxed text-muted">
            {copy.lede}
          </p>
          <p className="mt-6 text-sm text-faint">
            {copy.updated} <span className="text-accent">{updated} UTC</span>
            {" · "}
            <span className="text-accent">{data.n_submissions}</span>{" "}
            {copy.submissions}
            {" · "}
            {copy.rankedBy} {data.primary_metric.toUpperCase()}
          </p>
        </div>
      </section>

      <section className="mx-auto max-w-6xl px-5 py-16 sm:px-8 sm:py-20">
        <LeaderboardEnhanced data={data} dict={dict} />
      </section>
    </>
  );
}
