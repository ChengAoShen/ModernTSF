import type { Metadata } from "next";
import { TSEvalPage } from "@/components/tseval-page";
import { getDictionary } from "@/lib/dictionaries";

export const metadata: Metadata = { title: getDictionary("en").tseval.title };

export default function Page() {
  return <TSEvalPage locale="en" />;
}
