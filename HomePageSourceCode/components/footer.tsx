import Link from "next/link";
import { LogoTile } from "./logo";
import { localePath, type Locale } from "@/lib/i18n";
import type { Dictionary } from "@/lib/dictionaries";

export function Footer({
  locale,
  dict,
}: {
  locale: Locale;
  dict: Dictionary;
}) {
  const year = 2026;

  function resolve(href: string): string {
    if (href.startsWith("http") || href.startsWith("mailto:")) return href;
    return localePath(locale, href);
  }

  return (
    <footer className="border-t border-border bg-paper-2">
      <div className="mx-auto max-w-6xl px-5 py-14 sm:px-8">
        <div className="grid gap-10 lg:grid-cols-4">
          <div className="lg:col-span-2">
            <div className="flex items-center gap-2.5">
              <LogoTile className="h-7 w-7" />
              <span className="font-serif text-xl tracking-[-0.01em] text-ink">
                diaugeia<span className="text-accent">.ai</span>
              </span>
            </div>
            <p className="mt-4 max-w-xs text-sm leading-relaxed text-muted">
              {dict.footer.tagline}
            </p>
            <p className="mt-4 font-serif text-sm italic text-faint">
              διαύγεια
            </p>
          </div>

          {/* Explore + Community stay paired in a 2-col block, so on narrow
              screens they drop to the next row together, below the intro. */}
          <div className="grid grid-cols-2 gap-10 lg:col-span-2">
            {dict.footer.sections.map((section) => (
              <div key={section.title}>
                <h3 className="text-xs font-semibold uppercase tracking-wider text-faint">
                  {section.title}
                </h3>
                <ul className="mt-4 space-y-2.5">
                  {section.links.map((link) => (
                    <li key={link.label}>
                      <Link
                        href={resolve(link.href)}
                        className="text-sm text-muted transition-colors hover:text-ink"
                      >
                        {link.label}
                      </Link>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </div>

        <div className="mt-12 flex flex-col gap-2 border-t border-border pt-6 text-xs text-faint sm:flex-row sm:items-center sm:justify-between">
          <p>
            © {year} diaugeia. {dict.footer.rights}
          </p>
          <p>{dict.footer.openSource}</p>
        </div>
      </div>
    </footer>
  );
}
