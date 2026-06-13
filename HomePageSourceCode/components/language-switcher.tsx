"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  locales,
  localeNames,
  localePath,
  stripLocale,
  type Locale,
} from "@/lib/i18n";

export function LanguageSwitcher({ current }: { current: Locale }) {
  const pathname = usePathname();
  // The same page in the other language = same sub-path, different locale.
  const { subpath } = stripLocale(pathname);

  return (
    <div className="inline-flex items-center rounded-full border border-border p-0.5 text-[0.8rem]">
      {locales.map((locale) => {
        const active = locale === current;
        return (
          <Link
            key={locale}
            href={localePath(locale, subpath)}
            aria-current={active ? "true" : undefined}
            className={`rounded-full px-2.5 py-1 leading-none transition-colors ${
              active ? "bg-ink text-paper" : "text-muted hover:text-ink"
            }`}
          >
            {localeNames[locale]}
          </Link>
        );
      })}
    </div>
  );
}
