import Link from "next/link";
import {
  ArrowRight,
  Sparkles,
  Bot,
  Repeat,
  LockOpen,
  type LucideIcon,
} from "lucide-react";
import { localePath, type Locale } from "@/lib/i18n";
import { getDictionary } from "@/lib/dictionaries";
import { ContactDialog } from "./contact-dialog";

const GITHUB_URL = "https://github.com/Diaugeia";

const PILLAR_ICONS: Record<string, LucideIcon> = {
  sparkles: Sparkles,
  bot: Bot,
  repeat: Repeat,
  "lock-open": LockOpen,
};

export function Home({ locale }: { locale: Locale }) {
  const dict = getDictionary(locale);

  return (
    <>
      {/* ---------------------------------------------------------------- */}
      {/*  Hero — graphite ground with a soft gold "light beam" glow        */}
      {/* ---------------------------------------------------------------- */}
      <section className="relative overflow-hidden border-b border-border">
        <GoldGlow className="-right-40 -top-56 h-[46rem] w-[46rem]" strength={20} />

        <div className="mx-auto max-w-6xl px-5 py-24 sm:px-8 sm:py-32 lg:py-40">
          <div className="max-w-3xl animate-rise">
            <p className="text-sm font-medium uppercase tracking-[0.14em] text-accent">
              {dict.hero.eyebrow}
            </p>
            <h1 className="mt-6 font-serif text-[2.6rem] leading-[1.06] tracking-[-0.02em] text-ink sm:text-6xl lg:text-7xl">
              {dict.hero.title}{" "}
              <span className="italic text-accent">{dict.hero.titleAccent}</span>
            </h1>
            <p className="mt-7 max-w-2xl text-lg leading-relaxed text-muted sm:text-xl">
              {dict.hero.lede}
            </p>

            <div className="mt-9 flex flex-wrap items-center gap-3">
              <Link
                href={localePath(locale, "/news")}
                className="inline-flex items-center gap-2 rounded-full bg-accent px-6 py-3 text-sm font-medium text-accent-fg transition-colors hover:bg-accent-hover"
              >
                {dict.hero.ctaNews}
                <ArrowRight className="h-4 w-4" />
              </Link>
              <a
                href={GITHUB_URL}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-2 rounded-full border border-border-strong px-6 py-3 text-sm font-medium text-ink transition-colors hover:bg-paper-2"
              >
                <GithubIcon className="h-4 w-4" />
                {dict.hero.ctaGithub}
              </a>
            </div>

            <p className="mt-12 font-serif text-sm italic text-faint">
              {dict.hero.etymology}
            </p>
          </div>
        </div>
      </section>

      {/* ---------------------------------------------------------------- */}
      {/*  Mission                                                         */}
      {/* ---------------------------------------------------------------- */}
      <section className="mx-auto max-w-6xl px-5 py-20 sm:px-8 sm:py-28">
        <div className="grid gap-x-16 gap-y-6 lg:grid-cols-[16rem_1fr]">
          <SectionLabel>{dict.mission.label}</SectionLabel>
          <div className="max-w-3xl">
            <h2 className="font-serif text-3xl leading-snug tracking-[-0.01em] text-ink sm:text-[2.5rem]">
              {dict.mission.heading}
            </h2>
            <p className="mt-6 text-lg leading-relaxed text-muted">
              {dict.mission.body}
            </p>
          </div>
        </div>
      </section>

      {/* ---------------------------------------------------------------- */}
      {/*  Pillars — the four commitments                                  */}
      {/* ---------------------------------------------------------------- */}
      <section className="border-y border-border bg-paper-2">
        <div className="mx-auto max-w-6xl px-5 py-20 sm:px-8 sm:py-28">
          <SectionLabel>{dict.pillars.label}</SectionLabel>
          <h2 className="mt-4 max-w-2xl font-serif text-3xl leading-snug tracking-[-0.01em] text-ink sm:text-4xl">
            {dict.pillars.heading}
          </h2>

          <div className="mt-12 grid gap-px overflow-hidden rounded-2xl border border-border bg-border sm:grid-cols-2">
            {dict.pillars.items.map((item) => {
              const Icon = PILLAR_ICONS[item.icon] ?? Sparkles;
              return (
                <div
                  key={item.title}
                  className="flex flex-col bg-surface p-8 sm:p-9"
                >
                  <span
                    className="inline-flex h-11 w-11 items-center justify-center rounded-xl bg-accent-soft text-accent"
                    aria-hidden="true"
                  >
                    <Icon className="h-[1.35rem] w-[1.35rem]" strokeWidth={1.75} />
                  </span>
                  <h3 className="mt-4 font-serif text-xl tracking-[-0.01em] text-ink">
                    {item.title}
                  </h3>
                  <p className="mt-3 text-[0.97rem] leading-relaxed text-muted">
                    {item.body}
                  </p>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {/* ---------------------------------------------------------------- */}
      {/*  Closing CTA                                                     */}
      {/* ---------------------------------------------------------------- */}
      <section className="mx-auto max-w-6xl px-5 py-24 sm:px-8 sm:py-32">
        <div className="relative overflow-hidden rounded-3xl border border-border bg-ink px-8 py-16 sm:px-16 sm:py-20">
          <GoldGlow className="-bottom-40 -right-24 h-96 w-96" strength={36} />
          <div className="relative max-w-2xl">
            <h2 className="font-serif text-3xl leading-snug tracking-[-0.01em] text-paper sm:text-4xl">
              {dict.cta.heading}
            </h2>
            <p className="mt-5 text-lg leading-relaxed text-paper/70">
              {dict.cta.body}
            </p>
            <div className="mt-8 flex flex-wrap gap-3">
              <a
                href={GITHUB_URL}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-2 rounded-full bg-accent px-6 py-3 text-sm font-medium text-accent-fg transition-colors hover:bg-accent-hover"
              >
                <GithubIcon className="h-4 w-4" />
                {dict.hero.ctaGithub}
              </a>
              <ContactDialog
                contact={dict.contact}
                forms={dict.forms}
                className="inline-flex items-center gap-2 rounded-full border border-paper/25 px-6 py-3 text-sm font-medium text-paper transition-colors hover:bg-paper/10"
              />
            </div>
          </div>
        </div>
      </section>
    </>
  );
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <p className="text-xs font-semibold uppercase tracking-[0.16em] text-faint">
      {children}
    </p>
  );
}

/** A soft radial "light beam" glow in the gold accent — theme-aware. */
function GoldGlow({
  className = "",
  strength = 22,
}: {
  className?: string;
  strength?: number;
}) {
  return (
    <div
      aria-hidden="true"
      className={`pointer-events-none absolute rounded-full ${className}`}
      style={{
        background: `radial-gradient(circle, color-mix(in srgb, var(--accent) ${strength}%, transparent), transparent 70%)`,
      }}
    />
  );
}

/** GitHub brand mark (kept as a local SVG — lucide dropped brand icons). */
function GithubIcon({ className = "" }: { className?: string }) {
  return (
    <svg viewBox="0 0 16 16" fill="currentColor" className={className} aria-hidden="true">
      <path d="M8 0C3.58 0 0 3.58 0 8a8 8 0 0 0 5.47 7.59c.4.07.55-.17.55-.38v-1.33c-2.23.49-2.7-1.07-2.7-1.07-.36-.93-.89-1.18-.89-1.18-.73-.5.05-.49.05-.49.81.06 1.23.83 1.23.83.72 1.23 1.88.87 2.34.66.07-.52.28-.87.51-1.07-1.78-.2-3.65-.89-3.65-3.96 0-.87.31-1.59.82-2.15-.08-.2-.36-1.01.08-2.11 0 0 .67-.21 2.2.82a7.6 7.6 0 0 1 4 0c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.91.08 2.11.51.56.82 1.28.82 2.15 0 3.07-1.87 3.75-3.66 3.95.29.25.54.73.54 1.48v2.2c0 .21.15.46.55.38A8 8 0 0 0 16 8c0-4.42-3.58-8-8-8Z" />
    </svg>
  );
}
