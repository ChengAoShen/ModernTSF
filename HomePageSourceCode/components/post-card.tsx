import Link from "next/link";
import { formatDate, type PostMeta } from "@/lib/content";
import { localePath, type Locale } from "@/lib/i18n";

export function PostCard({
  post,
  locale,
}: {
  post: PostMeta;
  locale: Locale;
}) {
  const href = localePath(locale, `/${post.category}/${post.slug}`);

  return (
    <Link
      href={href}
      className="group flex flex-col rounded-xl border border-border bg-surface p-6 transition-all hover:border-border-strong hover:shadow-[0_1px_24px_-12px_rgba(0,0,0,0.18)]"
    >
      <div className="flex items-center gap-3 text-xs text-faint">
        <time dateTime={post.date}>{formatDate(post.date, locale)}</time>
        {post.tags?.[0] && (
          <>
            <span className="h-1 w-1 rounded-full bg-faint" />
            <span className="text-accent">{post.tags[0]}</span>
          </>
        )}
      </div>

      <h3 className="mt-3 font-serif text-xl leading-snug tracking-[-0.01em] text-ink transition-colors group-hover:text-accent">
        {post.title}
      </h3>

      {post.summary && (
        <p className="mt-2.5 line-clamp-3 text-[0.95rem] leading-relaxed text-muted">
          {post.summary}
        </p>
      )}

      {post.author && (
        <p className="mt-4 text-xs text-faint">{post.author}</p>
      )}
    </Link>
  );
}
