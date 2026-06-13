import Image from "next/image";

/**
 * diaugeia wordmark — the light-beam mark in a dark rounded tile, paired with
 * the lowercase logotype. The ".ai" suffix is set in the gold accent.
 */
export function Logo({ className = "" }: { className?: string }) {
  return (
    <span className={`inline-flex items-center gap-2.5 ${className}`}>
      <LogoTile className="h-8 w-8" />
      <span className="font-serif text-[1.35rem] leading-none tracking-[-0.01em] text-ink">
        diaugeia<span className="text-accent">.ai</span>
      </span>
    </span>
  );
}

/** The square logo mark — gold light beam on a dark tile. */
export function LogoTile({ className = "" }: { className?: string }) {
  return (
    <span
      className={`relative inline-block shrink-0 overflow-hidden rounded-lg bg-black ring-1 ring-border-strong ${className}`}
    >
      <Image
        src="/diaugeia.png"
        alt="diaugeia logo"
        fill
        sizes="48px"
        className="object-cover"
        priority
      />
    </span>
  );
}
