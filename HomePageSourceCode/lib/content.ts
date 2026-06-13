import fs from "node:fs";
import path from "node:path";
import matter from "gray-matter";
import type { Locale } from "./i18n";

export type Category = "research" | "news";

export interface PostMeta {
  slug: string;
  title: string;
  date: string;
  summary: string;
  author?: string;
  tags?: string[];
  category: Category;
  locale: Locale;
}

export interface Post extends PostMeta {
  content: string;
}

const CONTENT_ROOT = path.join(process.cwd(), "content");

function categoryDir(category: Category, locale: Locale): string {
  return path.join(CONTENT_ROOT, category, locale);
}

function formatMeta(
  raw: Record<string, unknown>,
  slug: string,
  category: Category,
  locale: Locale,
): PostMeta {
  return {
    slug,
    title: String(raw.title ?? slug),
    date: String(raw.date ?? ""),
    summary: String(raw.summary ?? ""),
    author: raw.author ? String(raw.author) : undefined,
    tags: Array.isArray(raw.tags) ? raw.tags.map(String) : undefined,
    category,
    locale,
  };
}

export function getAllPosts(category: Category, locale: Locale): PostMeta[] {
  const dir = categoryDir(category, locale);
  if (!fs.existsSync(dir)) return [];

  return fs
    .readdirSync(dir)
    .filter((file) => file.endsWith(".mdx") || file.endsWith(".md"))
    .map((file) => {
      const slug = file.replace(/\.mdx?$/, "");
      const raw = fs.readFileSync(path.join(dir, file), "utf8");
      const { data } = matter(raw);
      return formatMeta(data, slug, category, locale);
    })
    .sort((a, b) => (a.date < b.date ? 1 : -1));
}

export function getPost(
  category: Category,
  locale: Locale,
  slug: string,
): Post | null {
  const dir = categoryDir(category, locale);
  const candidates = [
    path.join(dir, `${slug}.mdx`),
    path.join(dir, `${slug}.md`),
  ];
  const filePath = candidates.find((p) => fs.existsSync(p));
  if (!filePath) return null;

  const raw = fs.readFileSync(filePath, "utf8");
  const { data, content } = matter(raw);
  return { ...formatMeta(data, slug, category, locale), content };
}

/** Slugs across all locales for a category — used by generateStaticParams. */
export function getAllSlugs(category: Category): string[] {
  const slugs = new Set<string>();
  for (const locale of ["en", "zh"] as Locale[]) {
    for (const post of getAllPosts(category, locale)) {
      slugs.add(post.slug);
    }
  }
  return [...slugs];
}

export function formatDate(date: string, locale: Locale): string {
  if (!date) return "";
  const d = new Date(date);
  if (Number.isNaN(d.getTime())) return date;
  return d.toLocaleDateString(locale === "zh" ? "zh-CN" : "en-US", {
    year: "numeric",
    month: "long",
    day: "numeric",
    // Date-only frontmatter strings ("2026-06-10") parse as UTC midnight;
    // format in UTC too, or the day shifts in timezones behind UTC.
    timeZone: "UTC",
  });
}
