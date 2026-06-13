"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Menu, X } from "lucide-react";
import { Logo } from "./logo";
import { ThemeToggle } from "./theme-toggle";
import { LanguageSwitcher } from "./language-switcher";
import { localePath, type Locale } from "@/lib/i18n";
import type { Dictionary } from "@/lib/dictionaries";

export function Nav({
  locale,
  dict,
}: {
  locale: Locale;
  dict: Dictionary;
}) {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);
  const home = localePath(locale);

  const links = [
    { href: localePath(locale, "/tseval"), label: dict.nav.tseval },
    { href: localePath(locale, "/research"), label: dict.nav.research },
    { href: localePath(locale, "/news"), label: dict.nav.news },
    { href: localePath(locale, "/join"), label: dict.nav.join },
  ];

  // Close the mobile menu whenever the route changes.
  useEffect(() => setOpen(false), [pathname]);

  return (
    <header className="sticky top-0 z-50 border-b border-border bg-paper/80 backdrop-blur-md">
      <div className="mx-auto flex h-16 max-w-6xl items-center justify-between px-5 sm:px-8">
        <Link href={home} aria-label="diaugeia — home" className="shrink-0">
          <Logo />
        </Link>

        {/* Desktop nav (≥ sm) */}
        <nav className="hidden items-center gap-1 sm:flex sm:gap-2">
          <div className="flex items-center">
            {links.map((link) => {
              const active = pathname.startsWith(link.href);
              return (
                <Link
                  key={link.href}
                  href={link.href}
                  className={`relative px-3 py-2 text-[0.95rem] transition-colors sm:px-4 ${
                    active ? "text-ink" : "text-muted hover:text-ink"
                  }`}
                >
                  {link.label}
                  {active && (
                    <span className="absolute inset-x-3 -bottom-px h-px bg-accent sm:inset-x-4" />
                  )}
                </Link>
              );
            })}
          </div>

          <span className="mx-1 hidden h-5 w-px bg-border sm:block" />

          <div className="flex items-center gap-2">
            <LanguageSwitcher current={locale} />
            <ThemeToggle labels={dict.themeToggle} />
          </div>
        </nav>

        {/* Mobile controls (< sm): theme toggle + hamburger */}
        <div className="flex items-center gap-1.5 sm:hidden">
          <ThemeToggle labels={dict.themeToggle} />
          <button
            type="button"
            aria-label="Menu"
            aria-expanded={open}
            aria-controls="mobile-menu"
            onClick={() => setOpen((v) => !v)}
            className="inline-flex h-9 w-9 items-center justify-center rounded-full border border-border text-ink transition-colors hover:border-border-strong"
          >
            {open ? (
              <X className="h-[1.1rem] w-[1.1rem]" />
            ) : (
              <Menu className="h-[1.1rem] w-[1.1rem]" />
            )}
          </button>
        </div>
      </div>

      {/* Mobile dropdown panel — floats over the page (absolute), so it does
          not push the content down. */}
      {open && (
        <div
          id="mobile-menu"
          className="absolute inset-x-0 top-full border-b border-border bg-paper shadow-[0_14px_28px_-14px_rgba(0,0,0,0.3)] sm:hidden"
        >
          <div className="mx-auto max-w-6xl px-5 py-3">
            <nav className="flex flex-col">
              {links.map((link) => {
                const active = pathname.startsWith(link.href);
                return (
                  <Link
                    key={link.href}
                    href={link.href}
                    className={`border-l-2 py-2.5 pl-3 text-[1.05rem] transition-colors ${
                      active
                        ? "border-accent font-medium text-ink"
                        : "border-transparent text-muted hover:text-ink"
                    }`}
                  >
                    {link.label}
                  </Link>
                );
              })}
            </nav>
            <div className="mt-3 border-t border-border pt-3">
              <LanguageSwitcher current={locale} />
            </div>
          </div>
        </div>
      )}
    </header>
  );
}
