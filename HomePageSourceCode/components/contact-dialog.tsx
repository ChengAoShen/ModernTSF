"use client";

import { useEffect, useState } from "react";
import { X } from "lucide-react";
import {
  Field,
  fieldInputClass,
  Honeypot,
  SubmitButton,
  SuccessPanel,
  ErrorBanner,
  postSubmission,
  isValidEmail,
  type SubmitState,
} from "./form-kit";
import type { Dictionary } from "@/lib/dictionaries";

/** "Contact us" button that opens the 3-field contact form in a modal dialog. */
export function ContactDialog({
  contact,
  forms,
  className,
}: {
  contact: Dictionary["contact"];
  forms: Dictionary["forms"];
  className?: string;
}) {
  const [open, setOpen] = useState(false);
  const [state, setState] = useState<SubmitState>("idle");
  const [validationError, setValidationError] = useState<string | null>(null);

  // Close on Escape and lock background scroll while the dialog is open.
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    window.addEventListener("keydown", onKey);
    document.body.style.overflow = "hidden";
    return () => {
      window.removeEventListener("keydown", onKey);
      document.body.style.overflow = "";
    };
  }, [open]);

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
    const message = String(data.get("message") ?? "").trim();

    if (!name || !email || !message) {
      setValidationError(forms.required);
      return;
    }
    if (!isValidEmail(email)) {
      setValidationError(forms.invalidEmail);
      return;
    }

    setValidationError(null);
    setState("sending");
    const ok = await postSubmission({ kind: "contact", name, email, message });
    setState(ok ? "success" : "error");
  }

  return (
    <>
      <button type="button" onClick={() => setOpen(true)} className={className}>
        {contact.cta}
      </button>

      {open && (
        <div
          className="fixed inset-0 z-[100] flex items-center justify-center p-4"
          role="dialog"
          aria-modal="true"
          aria-label={contact.heading}
        >
          <div
            className="absolute inset-0 bg-ink/60 backdrop-blur-sm"
            onClick={() => setOpen(false)}
            aria-hidden="true"
          />
          <div className="relative w-full max-w-md animate-rise rounded-2xl border border-border bg-surface p-6 shadow-[0_24px_60px_-15px_rgba(0,0,0,0.4)] sm:p-8">
            <button
              type="button"
              onClick={() => setOpen(false)}
              aria-label="Close"
              className="absolute right-4 top-4 inline-flex h-8 w-8 items-center justify-center rounded-full text-muted transition-colors hover:bg-paper-2 hover:text-ink"
            >
              <X className="h-[1.1rem] w-[1.1rem]" />
            </button>

            {state === "success" ? (
              <SuccessPanel
                title={contact.successTitle}
                body={contact.successBody}
              />
            ) : (
              <>
                <h2 className="font-serif text-2xl tracking-[-0.01em] text-ink">
                  {contact.heading}
                </h2>
                <p className="mt-2 max-w-sm text-[0.95rem] leading-relaxed text-muted">
                  {contact.body}
                </p>

                <form
                  onSubmit={onSubmit}
                  noValidate
                  className="mt-6 flex flex-col gap-5"
                >
                  <Field label={contact.fields.name} htmlFor="contact-name" required>
                    <input
                      id="contact-name"
                      name="name"
                      type="text"
                      autoComplete="name"
                      required
                      className={fieldInputClass}
                    />
                  </Field>
                  <Field label={contact.fields.email} htmlFor="contact-email" required>
                    <input
                      id="contact-email"
                      name="email"
                      type="email"
                      autoComplete="email"
                      required
                      className={fieldInputClass}
                    />
                  </Field>
                  <Field
                    label={contact.fields.message}
                    htmlFor="contact-message"
                    required
                  >
                    <textarea
                      id="contact-message"
                      name="message"
                      rows={4}
                      required
                      className={`${fieldInputClass} resize-y`}
                    />
                  </Field>

                  <Honeypot />

                  {(validationError || state === "error") && (
                    <ErrorBanner message={validationError ?? contact.error} />
                  )}

                  <div>
                    <SubmitButton
                      state={state}
                      label={contact.submit}
                      sendingLabel={contact.sending}
                    />
                  </div>
                </form>
              </>
            )}
          </div>
        </div>
      )}
    </>
  );
}
