import type { Metadata } from "next";
import { Inter, Newsreader, Fira_Code, Noto_Serif_SC } from "next/font/google";
import "./globals.css";
import { ThemeProvider } from "@/components/theme-provider";

// All self-hosted by next/font at build time — fetched from our own origin at
// runtime, never from Google (so they load fine in mainland China).
const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

const newsreader = Newsreader({
  subsets: ["latin"],
  variable: "--font-newsreader",
  display: "swap",
  style: ["normal", "italic"],
});

const firaCode = Fira_Code({
  subsets: ["latin"],
  variable: "--font-fira-code",
  display: "swap",
});

// Chinese serif for headings (思源宋体). CJK has no small subset, so disable
// preload — the browser still loads only the slices for characters in use.
const notoSerifSC = Noto_Serif_SC({
  weight: ["400", "600", "700"],
  variable: "--font-noto-serif-sc",
  display: "swap",
  preload: false,
});

const title =
  "Diaugeia.AI — Transparent, open-source infrastructure for AI research";
const description =
  "Diaugeia.AI builds the open, reproducible infrastructure that modern AI research depends on.";

export const metadata: Metadata = {
  title: {
    default: title,
    template: "%s · Diaugeia.AI",
  },
  description,
  metadataBase: new URL("https://diaugeia.ai"),
  manifest: "/manifest.webmanifest",
  openGraph: { title, description, type: "website" },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html
      lang="en"
      suppressHydrationWarning
      className={`${inter.variable} ${newsreader.variable} ${firaCode.variable} ${notoSerifSC.variable}`}
    >
      <body className="flex min-h-screen flex-col">
        <ThemeProvider
          attribute="class"
          defaultTheme="light"
          enableSystem
          disableTransitionOnChange
        >
          {children}
        </ThemeProvider>
      </body>
    </html>
  );
}
