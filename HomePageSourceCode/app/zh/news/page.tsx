import type { Metadata } from "next";
import { CategoryPage } from "@/components/category-page";
import { getDictionary } from "@/lib/dictionaries";

export const metadata: Metadata = { title: getDictionary("zh").news.title };

export default function Page() {
  return <CategoryPage category="news" locale="zh" />;
}
