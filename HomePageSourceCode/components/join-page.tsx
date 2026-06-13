"use client";

import { useState } from "react";
import { getDictionary } from "@/lib/dictionaries";
import type { Locale } from "@/lib/i18n";
import {
  Field,
  FieldGroup,
  ChipGroup,
  fieldInputClass,
  Honeypot,
  SubmitButton,
  SuccessPanel,
  ErrorBanner,
  postSubmission,
  isValidEmail,
  type SubmitState,
} from "./form-kit";

/** Standalone community membership page served at /join (and /zh/join). */
export function JoinPage({ locale }: { locale: Locale }) {
  const dict = getDictionary(locale);
  const t = dict.join;

  const [state, setState] = useState<SubmitState>("idle");
  const [interests, setInterests] = useState<string[]>([]);
  const [validationError, setValidationError] = useState<string | null>(null);

  function toggleInterest(option: string) {
    setInterests((prev) =>
      prev.includes(option)
        ? prev.filter((x) => x !== option)
        : [...prev, option],
    );
  }

  async function onSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const data = new FormData(e.currentTarget);

    // Honeypot: a bot filled the hidden field — pretend success, send nothing.
    if (String(data.get("company") ?? "").trim() !== "") {
      setState("success");
      return;
    }

    const name = String(data.get("name") ?? "").trim();
    const email = String(data.get("email") ?? "").trim();
    const affiliation = String(data.get("affiliation") ?? "").trim();
    const role = String(data.get("role") ?? "").trim();
    const link = String(data.get("link") ?? "").trim();
    const motivation = String(data.get("motivation") ?? "").trim();

    if (!name || !email || !affiliation || !role || !motivation) {
      setValidationError(dict.forms.required);
      return;
    }
    if (!isValidEmail(email)) {
      setValidationError(dict.forms.invalidEmail);
      return;
    }

    setValidationError(null);
    setState("sending");
    const ok = await postSubmission({
      kind: "join",
      name,
      email,
      affiliation,
      role,
      interests,
      link,
      motivation,
    });
    setState(ok ? "success" : "error");
  }

  return (
    <>
      {/* Hero */}
      <section className="border-b border-border">
        <div className="mx-auto max-w-6xl px-5 py-20 sm:px-8 sm:py-24">
          <h1 className="font-serif text-4xl tracking-[-0.02em] text-ink sm:text-6xl">
            {t.title}
          </h1>
          <p className="mt-5 max-w-2xl text-lg leading-relaxed text-muted">
            {t.lede}
          </p>
        </div>
      </section>

      {/* Intro + form */}
      <section className="mx-auto max-w-3xl px-5 py-16 sm:px-8 sm:py-20">
        <p className="text-lg leading-relaxed text-muted">{t.intro}</p>

        <div className="mt-8 rounded-2xl border border-border bg-surface p-6 sm:p-8">
          {state === "success" ? (
            <SuccessPanel title={t.successTitle} body={t.successBody} />
          ) : (
            <form onSubmit={onSubmit} noValidate className="flex flex-col gap-5">
              <div className="grid gap-5 sm:grid-cols-2">
                <Field label={t.fields.name} htmlFor="join-name" required>
                  <input
                    id="join-name"
                    name="name"
                    type="text"
                    autoComplete="name"
                    required
                    className={fieldInputClass}
                  />
                </Field>
                <Field label={t.fields.email} htmlFor="join-email" required>
                  <input
                    id="join-email"
                    name="email"
                    type="email"
                    autoComplete="email"
                    required
                    className={fieldInputClass}
                  />
                </Field>
              </div>

              <Field
                label={t.fields.affiliation}
                htmlFor="join-affiliation"
                required
              >
                <input
                  id="join-affiliation"
                  name="affiliation"
                  type="text"
                  placeholder={t.placeholders.affiliation}
                  required
                  className={fieldInputClass}
                />
              </Field>

              <Field label={t.fields.role} htmlFor="join-role" required>
                <select
                  id="join-role"
                  name="role"
                  defaultValue=""
                  required
                  className={fieldInputClass}
                >
                  <option value="" disabled>
                    {t.rolePlaceholder}
                  </option>
                  {t.roleOptions.map((opt) => (
                    <option key={opt} value={opt}>
                      {opt}
                    </option>
                  ))}
                </select>
              </Field>

              <FieldGroup label={t.fields.interests} optional={t.optional}>
                <div className="pt-1">
                  <ChipGroup
                    name={t.fields.interests}
                    options={t.interestOptions}
                    selected={interests}
                    onToggle={toggleInterest}
                  />
                </div>
              </FieldGroup>

              <Field
                label={t.fields.link}
                htmlFor="join-link"
                optional={t.optional}
              >
                <input
                  id="join-link"
                  name="link"
                  type="url"
                  placeholder={t.placeholders.link}
                  className={fieldInputClass}
                />
              </Field>

              <Field label={t.fields.motivation} htmlFor="join-motivation" required>
                <textarea
                  id="join-motivation"
                  name="motivation"
                  rows={4}
                  placeholder={t.placeholders.motivation}
                  required
                  className={`${fieldInputClass} resize-y`}
                />
              </Field>

              <Honeypot />

              {(validationError || state === "error") && (
                <ErrorBanner message={validationError ?? t.error} />
              )}

              <div>
                <SubmitButton
                  state={state}
                  label={t.submit}
                  sendingLabel={t.sending}
                />
              </div>
            </form>
          )}
        </div>
      </section>
    </>
  );
}
