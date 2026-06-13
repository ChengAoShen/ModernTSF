import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { Article } from "@/components/article";
import { getPost, getAllSlugs } from "@/lib/content";

export function generateStaticParams() {
  return getAllSlugs("research").map((slug) => ({ slug }));
}

export async function generateMetadata({
  params,
}: {
  params: Promise<{ slug: string }>;
}): Promise<Metadata> {
  const { slug } = await params;
  const post = getPost("research", "zh", slug);
  if (!post) return {};
  return { title: post.title, description: post.summary };
}

export default async function Page({
  params,
}: {
  params: Promise<{ slug: string }>;
}) {
  const { slug } = await params;
  const post = getPost("research", "zh", slug);
  if (!post) notFound();
  return <Article post={post} locale="zh" />;
}
