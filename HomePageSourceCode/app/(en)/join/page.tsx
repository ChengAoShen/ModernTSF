import type { Metadata } from "next";
import { JoinPage } from "@/components/join-page";
import { getDictionary } from "@/lib/dictionaries";

export const metadata: Metadata = { title: getDictionary("en").join.title };

export default function Page() {
  return <JoinPage locale="en" />;
}
