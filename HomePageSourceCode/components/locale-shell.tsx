import { Nav } from "./nav";
import { Footer } from "./footer";
import { HtmlLang } from "./html-lang";
import { getDictionary } from "@/lib/dictionaries";
import type { Locale } from "@/lib/i18n";

/** Per-locale page chrome: nav + footer + lang sync. Used by both locale
    layouts so the page logic stays shared. */
export function LocaleShell({
  locale,
  children,
}: {
  locale: Locale;
  children: React.ReactNode;
}) {
  const dict = getDictionary(locale);
  return (
    <>
      <HtmlLang lang={locale} />
      <Nav locale={locale} dict={dict} />
      <main className="flex-1">{children}</main>
      <Footer locale={locale} dict={dict} />
    </>
  );
}
