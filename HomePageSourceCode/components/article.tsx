import Link from "next/link";
import { ArrowLeft } from "lucide-react";
import { Mdx } from "./mdx";
import { formatDate, type Post } from "@/lib/content";
import { getDictionary } from "@/lib/dictionaries";
import { localePath, type Locale } from "@/lib/i18n";

export function Article({ post, locale }: { post: Post; locale: Locale }) {
  const dict = getDictionary(locale);
  const backLabel =
    post.category === "research" ? dict.post.backResearch : dict.post.backNews;
  const backHref = localePath(locale, `/${post.category}`);

  return (
    <article className="mx-auto max-w-3xl px-5 py-16 sm:px-8 sm:py-24">
      <Link
        href={backHref}
        className="group inline-flex items-center gap-1.5 text-sm text-muted transition-colors hover:text-accent"
      >
        <ArrowLeft className="h-3.5 w-3.5" />
        {backLabel}
      </Link>

      <header className="mt-8 border-b border-border pb-10">
        <div className="flex flex-wrap items-center gap-3 text-sm text-faint">
          <time dateTime={post.date}>{formatDate(post.date, locale)}</time>
          {post.author && (
            <>
              <span className="h-1 w-1 rounded-full bg-faint" />
              <span>{post.author}</span>
            </>
          )}
        </div>
        <h1 className="mt-4 font-serif text-4xl leading-[1.12] tracking-[-0.02em] text-ink sm:text-5xl">
          {post.title}
        </h1>
        {post.summary && (
          <p className="mt-5 text-lg leading-relaxed text-muted">
            {post.summary}
          </p>
        )}
        {post.tags && post.tags.length > 0 && (
          <div className="mt-6 flex flex-wrap gap-2">
            {post.tags.map((tag) => (
              <span
                key={tag}
                className="rounded-full border border-border bg-paper-2 px-3 py-1 text-xs text-muted"
              >
                {tag}
              </span>
            ))}
          </div>
        )}
      </header>

      <div className="mt-10">
        <Mdx source={post.content} />
      </div>
    </article>
  );
}
