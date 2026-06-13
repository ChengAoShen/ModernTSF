export const locales = ["en", "zh"] as const;
export type Locale = (typeof locales)[number];
export const defaultLocale: Locale = "en";

export function isLocale(value: string): value is Locale {
  return (locales as readonly string[]).includes(value);
}

export const localeNames: Record<Locale, string> = {
  en: "EN",
  zh: "中文",
};

/**
 * Build a URL path for a locale. The default locale (en) lives at the root with
 * no prefix; zh is served under `/zh`.
 *   localePath("en")            -> "/"
 *   localePath("en", "/news")   -> "/news"
 *   localePath("zh")            -> "/zh"
 *   localePath("zh", "/news")   -> "/zh/news"
 */
export function localePath(locale: Locale, subpath = ""): string {
  const p = subpath && !subpath.startsWith("/") ? `/${subpath}` : subpath;
  if (locale === defaultLocale) return p || "/";
  return `/zh${p}`;
}

/** Split a pathname into its locale and the locale-independent sub-path. */
export function stripLocale(pathname: string): {
  locale: Locale;
  subpath: string;
} {
  // Normalize a trailing slash (keep "/" as-is).
  const path =
    pathname.length > 1 && pathname.endsWith("/")
      ? pathname.slice(0, -1)
      : pathname;

  if (path === "/zh" || path.startsWith("/zh/")) {
    return { locale: "zh", subpath: path.slice(3) }; // "/zh" -> "", "/zh/news" -> "/news"
  }
  return { locale: "en", subpath: path === "/" ? "" : path };
}
