import { LocaleShell } from "@/components/locale-shell";

export default function EnLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return <LocaleShell locale="en">{children}</LocaleShell>;
}
