import { PostCard } from "./post-card";
import { getAllPosts, type Category } from "@/lib/content";
import { getDictionary } from "@/lib/dictionaries";
import type { Locale } from "@/lib/i18n";

export function CategoryPage({
  category,
  locale,
}: {
  category: Category;
  locale: Locale;
}) {
  const dict = getDictionary(locale);
  const copy = category === "research" ? dict.research : dict.news;
  const posts = getAllPosts(category, locale);

  return (
    <>
      <section className="border-b border-border">
        <div className="mx-auto max-w-6xl px-5 py-20 sm:px-8 sm:py-24">
          <h1 className="font-serif text-4xl tracking-[-0.02em] text-ink sm:text-6xl">
            {copy.title}
          </h1>
          <p className="mt-5 max-w-2xl text-lg leading-relaxed text-muted">
            {copy.lede}
          </p>
        </div>
      </section>

      <section className="mx-auto max-w-6xl px-5 py-16 sm:px-8 sm:py-20">
        {posts.length === 0 ? (
          <p className="text-muted">{dict.latest.empty}</p>
        ) : (
          <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-3">
            {posts.map((post) => (
              <PostCard key={post.slug} post={post} locale={locale} />
            ))}
          </div>
        )}
      </section>
    </>
  );
}
