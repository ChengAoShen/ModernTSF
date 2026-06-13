import type { Metadata } from "next";
import { LocaleShell } from "@/components/locale-shell";

const title = "Diaugeia.AI — 面向 AI 研究的透明、开源基础设施";
const description =
  "Diaugeia.AI 构建现代 AI 研究所依赖的开放基础设施——可复现、开源、透明。";

export const metadata: Metadata = {
  title: { default: title, template: "%s · Diaugeia.AI" },
  description,
  openGraph: { title, description, type: "website" },
};

export default function ZhLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return <LocaleShell locale="zh">{children}</LocaleShell>;
}
