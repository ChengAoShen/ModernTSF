import type { MetadataRoute } from "next";
import { locales, localePath, type Locale } from "@/lib/i18n";
import { getAllSlugs, getPost, type Category } from "@/lib/content";

const BASE = "https://diaugeia.ai";

export const dynamic = "force-static";

/** Absolute, trailing-slashed URL (matches the deployed paths). */
function url(locale: Locale, subpath: string): string {
  const p = localePath(locale, subpath);
  return BASE + (p.endsWith("/") ? p : `${p}/`);
}

/** One entry per locale for a logical page, cross-linked via hreflang. */
function group(
  subpath: string,
  opts: {
    lastModified?: Date;
    changeFrequency?: MetadataRoute.Sitemap[number]["changeFrequency"];
    priority?: number;
  } = {},
): MetadataRoute.Sitemap {
  const languages: Record<string, string> = {};
  for (const locale of locales) languages[locale] = url(locale, subpath);

  return locales.map((locale) => ({
    url: url(locale, subpath),
    lastModified: opts.lastModified,
    changeFrequency: opts.changeFrequency,
    priority: opts.priority,
    alternates: { languages },
  }));
}

function dateOf(value?: string): Date | undefined {
  if (!value) return undefined;
  const d = new Date(value);
  return Number.isNaN(d.getTime()) ? undefined : d;
}

export default function sitemap(): MetadataRoute.Sitemap {
  const entries: MetadataRoute.Sitemap = [
    ...group("", { changeFrequency: "weekly", priority: 1 }),
    ...group("/research", { changeFrequency: "weekly", priority: 0.8 }),
    ...group("/news", { changeFrequency: "weekly", priority: 0.8 }),
    ...group("/join", { changeFrequency: "monthly", priority: 0.7 }),
  ];

  for (const category of ["research", "news"] as Category[]) {
    for (const slug of getAllSlugs(category)) {
      const post = getPost(category, "en", slug) ?? getPost(category, "zh", slug);
      entries.push(
        ...group(`/${category}/${slug}`, {
          lastModified: dateOf(post?.date),
          changeFrequency: "monthly",
          priority: 0.6,
        }),
      );
    }
  }

  return entries;
}
