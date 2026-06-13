import { EmailMessage } from "cloudflare:email";
import { createMimeMessage, Mailbox } from "mimetext";

/**
 * Thin Cloudflare Worker that fronts the statically-exported site.
 *
 *   POST /api/submit  → validate the Contact / Join form and email it natively
 *                       through the Email Routing send_email binding (env.SEB).
 *   everything else   → served from the static assets in out/ (env.ASSETS).
 *
 * No third-party service, no API key: the binding (wrangler.jsonc) is locked to
 * one verified recipient. Reply-To is the submitter, so a reply just works.
 */

const FROM = "noreply@diaugeia.ai";
const TO = "chengao_shen@ieee.org";
const MAX_BODY = 64 * 1024; // reject obviously-oversized payloads
const MAX_LINE = 200; // single-line fields (name, email, role, …)
const MAX_TEXT = 5000; // multi-line fields (message, motivation)
const MAX_INTERESTS = 20;

const json = (data, status = 200) =>
  new Response(JSON.stringify(data), {
    status,
    headers: { "content-type": "application/json" },
  });

const isEmail = (v) => /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(v);
// Single-line field: collapse CR/LF (so values can't smuggle mail headers) + cap.
const line = (v) =>
  String(v ?? "")
    .replace(/[\r\n]+/g, " ")
    .trim()
    .slice(0, MAX_LINE);
// Multi-line field: kept in the body only (never a header), so newlines are fine.
const text = (v) => String(v ?? "").trim().slice(0, MAX_TEXT);

export default {
  async fetch(request, env) {
    const { pathname } = new URL(request.url);
    if (pathname === "/api/submit" || pathname === "/api/submit/") {
      if (request.method !== "POST") {
        return json({ ok: false, error: "method_not_allowed" }, 405);
      }
      return handleSubmit(request, env);
    }
    return env.ASSETS.fetch(request);
  },
};

async function handleSubmit(request, env) {
  if (Number(request.headers.get("content-length") || 0) > MAX_BODY) {
    return json({ ok: false, error: "too_large" }, 413);
  }

  let body;
  try {
    body = await request.json();
  } catch {
    return json({ ok: false, error: "invalid_json" }, 400);
  }

  // Honeypot: real users leave this blank. Return ok so bots don't learn they
  // were dropped — but send nothing.
  if (typeof body.company === "string" && body.company.trim() !== "") {
    return json({ ok: true });
  }

  const kind = body.kind;
  if (kind !== "contact" && kind !== "join") {
    return json({ ok: false, error: "invalid_kind" }, 400);
  }

  const name = line(body.name);
  const email = line(body.email);
  if (!name || !isEmail(email)) {
    return json({ ok: false, error: "invalid_fields" }, 400);
  }

  let subject;
  const rows = [
    ["Name", name],
    ["Email", email],
  ];

  if (kind === "contact") {
    const message = text(body.message);
    if (!message) return json({ ok: false, error: "invalid_fields" }, 400);
    subject = `[Diaugeia] Contact — ${name}`;
    rows.push(["Message", message]);
  } else {
    const affiliation = line(body.affiliation);
    const role = line(body.role);
    const motivation = text(body.motivation);
    if (!affiliation || !role || !motivation) {
      return json({ ok: false, error: "invalid_fields" }, 400);
    }
    const interests = Array.isArray(body.interests)
      ? body.interests.slice(0, MAX_INTERESTS).map(line).filter(Boolean).join(", ")
      : "";
    const link = line(body.link);
    subject = `[Diaugeia] Join application — ${name}`;
    rows.push(
      ["Affiliation", affiliation],
      ["Role", role],
      ["Interests", interests || "—"],
      ["Link", link || "—"],
      ["Motivation", motivation],
    );
  }

  try {
    const msg = createMimeMessage();
    msg.setSender({ name: "Diaugeia", addr: FROM });
    msg.setRecipient(TO);
    msg.setSubject(subject);
    msg.setHeader("Reply-To", new Mailbox(email));
    msg.addMessage({
      contentType: "text/plain",
      data: rows.map(([k, v]) => `${k}: ${v}`).join("\n\n"),
    });
    await env.SEB.send(new EmailMessage(FROM, TO, msg.asRaw()));
  } catch (e) {
    console.error("[submit] send failed:", e instanceof Error ? e.stack : e);
    return json({ ok: false, error: "send_failed" }, 502);
  }

  return json({ ok: true });
}
