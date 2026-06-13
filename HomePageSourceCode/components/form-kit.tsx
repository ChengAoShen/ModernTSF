"use client";

/* ------------------------------------------------------------------ */
/*  Shared building blocks for the Contact and Join forms.             */
/*  Both POST same-origin JSON to /api/submit, handled by the thin     */
/*  Cloudflare Worker (worker.js), which emails via the native         */
/*  Email Routing send_email binding.                                  */
/* ------------------------------------------------------------------ */

export type SubmitState = "idle" | "sending" | "success" | "error";

/** Shared graphite + gold input styling, used by inputs / textareas / selects. */
export const fieldInputClass =
  "w-full rounded-lg border border-border bg-surface px-3.5 py-2.5 text-[0.95rem] text-ink placeholder:text-faint transition-colors focus:border-accent";

/** A labelled field wrapper. `required` shows a gold asterisk; `optional`
    renders a faint suffix after the label. */
export function Field({
  label,
  htmlFor,
  required,
  optional,
  children,
}: {
  label: string;
  htmlFor: string;
  required?: boolean;
  optional?: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <label
        htmlFor={htmlFor}
        className="mb-1.5 block text-sm font-medium text-ink-2"
      >
        {label}
        {required && (
          <span aria-hidden="true" className="ml-1 text-accent">
            *
          </span>
        )}
        {optional && (
          <span className="ml-1.5 font-normal text-faint">{optional}</span>
        )}
      </label>
      {children}
    </div>
  );
}

/** Like Field, but for a group of controls with no single labelable input
    (e.g. the interest chips) — uses a plain label, not a dangling htmlFor. */
export function FieldGroup({
  label,
  optional,
  children,
}: {
  label: string;
  optional?: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <p className="mb-1.5 text-sm font-medium text-ink-2">
        {label}
        {optional && (
          <span className="ml-1.5 font-normal text-faint">{optional}</span>
        )}
      </p>
      {children}
    </div>
  );
}

/** A multi-select rendered as toggleable chips. */
export function ChipGroup({
  name,
  options,
  selected,
  onToggle,
}: {
  name: string;
  options: string[];
  selected: string[];
  onToggle: (option: string) => void;
}) {
  return (
    <div className="flex flex-wrap gap-2" role="group" aria-label={name}>
      {options.map((opt) => {
        const active = selected.includes(opt);
        return (
          <button
            type="button"
            key={opt}
            onClick={() => onToggle(opt)}
            aria-pressed={active}
            className={`rounded-full border px-3.5 py-1.5 text-sm transition-colors ${
              active
                ? "border-accent bg-accent-soft text-accent"
                : "border-border text-muted hover:border-border-strong hover:text-ink"
            }`}
          >
            {opt}
          </button>
        );
      })}
    </div>
  );
}

/** Visually-hidden honeypot. Bots fill it; the server drops those submissions. */
export function Honeypot() {
  return (
    <div aria-hidden="true" className="absolute left-[-9999px] top-[-9999px]">
      <label>
        Company
        <input
          type="text"
          name="company"
          tabIndex={-1}
          autoComplete="off"
        />
      </label>
    </div>
  );
}

/** Primary submit button; shows the sending label and disables while pending. */
export function SubmitButton({
  state,
  label,
  sendingLabel,
}: {
  state: SubmitState;
  label: string;
  sendingLabel: string;
}) {
  const pending = state === "sending";
  return (
    <button
      type="submit"
      disabled={pending}
      className="inline-flex items-center gap-2 rounded-full bg-accent px-6 py-3 text-sm font-medium text-accent-fg transition-colors hover:bg-accent-hover disabled:cursor-not-allowed disabled:opacity-60"
    >
      {pending && <Spinner />}
      {pending ? sendingLabel : label}
    </button>
  );
}

/** Success confirmation panel shown in place of the form once submitted. */
export function SuccessPanel({
  title,
  body,
}: {
  title: string;
  body: string;
}) {
  return (
    <div className="flex flex-col items-center rounded-2xl border border-accent/40 bg-accent-soft/40 px-6 py-10 text-center">
      <span className="inline-flex h-11 w-11 items-center justify-center rounded-full bg-accent text-accent-fg">
        <CheckIcon className="h-5 w-5" />
      </span>
      <p className="mt-4 font-serif text-xl tracking-[-0.01em] text-ink">
        {title}
      </p>
      <p className="mt-2 max-w-sm text-[0.95rem] leading-relaxed text-muted">
        {body}
      </p>
    </div>
  );
}

/** Inline error banner; keeps the filled-in form intact for a retry. */
export function ErrorBanner({ message }: { message: string }) {
  return (
    <p
      role="alert"
      className="rounded-lg border border-red-500/30 bg-red-500/5 px-3.5 py-2.5 text-sm text-red-700 dark:text-red-300"
    >
      {message}
    </p>
  );
}

/** POST a submission to the Cloudflare Pages Function. Returns true on success. */
export async function postSubmission(
  payload: Record<string, unknown>,
): Promise<boolean> {
  try {
    const res = await fetch("/api/submit", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) return false;
    const data = (await res.json().catch(() => null)) as { ok?: boolean } | null;
    return Boolean(data?.ok);
  } catch {
    return false;
  }
}

/** Minimal RFC-ish email check for client-side validation. */
export function isValidEmail(value: string): boolean {
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value.trim());
}

function Spinner() {
  return (
    <svg
      className="h-4 w-4 animate-spin"
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden="true"
    >
      <circle
        className="opacity-25"
        cx="12"
        cy="12"
        r="10"
        stroke="currentColor"
        strokeWidth="4"
      />
      <path
        className="opacity-90"
        fill="currentColor"
        d="M4 12a8 8 0 0 1 8-8V0C5.37 0 0 5.37 0 12h4z"
      />
    </svg>
  );
}

function CheckIcon({ className = "" }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
      aria-hidden="true"
    >
      <path d="M20 6 9 17l-5-5" />
    </svg>
  );
}
