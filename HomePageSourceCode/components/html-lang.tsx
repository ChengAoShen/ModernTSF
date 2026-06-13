"use client";

import { useEffect } from "react";

/** Keeps <html lang> in sync with the current locale (root layout defaults to
    "en"; this updates it for /zh pages and across client navigation). */
export function HtmlLang({ lang }: { lang: string }) {
  useEffect(() => {
    document.documentElement.lang = lang;
  }, [lang]);
  return null;
}
