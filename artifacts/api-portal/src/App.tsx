import { useState, useEffect, useCallback } from "react";

const BG = "hsl(222,47%,11%)";
const CARD = "hsl(222,47%,15%)";
const BORDER = "hsl(222,47%,22%)";
const TEXT = "hsl(210,40%,92%)";
const MUTED = "hsl(215,20%,55%)";

const OPENAI_BLUE = "#3B82F6";
const ANTHROPIC_ORANGE = "#F97316";
const BOTH_GRAY = "#6B7280";
const GET_GREEN = "#22C55E";
const POST_PURPLE = "#A855F7";

const MODELS = [
  { id: "gpt-5.2", provider: "OpenAI" },
  { id: "gpt-5-mini", provider: "OpenAI" },
  { id: "gpt-5-nano", provider: "OpenAI" },
  { id: "o4-mini", provider: "OpenAI" },
  { id: "o3", provider: "OpenAI" },
  { id: "claude-opus-4-6", provider: "Anthropic" },
  { id: "claude-sonnet-4-6", provider: "Anthropic" },
  { id: "claude-haiku-4-5", provider: "Anthropic" },
];

const ENDPOINTS = [
  {
    method: "GET",
    path: "/v1/models",
    label: "List Models",
    compat: "Both",
    description: "Returns all available OpenAI and Anthropic models.",
  },
  {
    method: "POST",
    path: "/v1/chat/completions",
    label: "Chat Completions",
    compat: "OpenAI",
    description: "OpenAI-compatible chat completions. Supports streaming, tool calls, and all listed models (auto-routes claude- models via format conversion).",
  },
  {
    method: "POST",
    path: "/v1/messages",
    label: "Messages",
    compat: "Anthropic",
    description: "Anthropic Messages native API. Supports streaming and tool calls. OpenAI models are automatically converted to Anthropic response format.",
  },
];

const STEPS = [
  {
    title: "Open CherryStudio Settings",
    desc: "Go to Settings → Model Providers → Add Provider.",
  },
  {
    title: "Choose Provider Type",
    desc: "Select OpenAI-compatible or Anthropic — both work with this proxy.",
  },
  {
    title: "Enter Base URL & Key",
    desc: "Set the Base URL to this site's origin and the API Key to your PROXY_API_KEY.",
  },
  {
    title: "Select a Model & Chat",
    desc: "Pick any listed model (OpenAI or Anthropic) and start chatting.",
  },
];

function useCopy() {
  const [copied, setCopied] = useState<string | null>(null);
  const copy = useCallback((id: string, text: string) => {
    if (navigator.clipboard) {
      navigator.clipboard.writeText(text).catch(() => fallbackCopy(text));
    } else {
      fallbackCopy(text);
    }
    setCopied(id);
    setTimeout(() => setCopied(null), 2000);
  }, []);
  return { copied, copy };
}

function fallbackCopy(text: string) {
  const el = document.createElement("textarea");
  el.value = text;
  el.style.position = "fixed";
  el.style.opacity = "0";
  document.body.appendChild(el);
  el.focus();
  el.select();
  try { document.execCommand("copy"); } catch { /* ignore */ }
  document.body.removeChild(el);
}

function CopyBtn({ id, text, copied, copy }: { id: string; text: string; copied: string | null; copy: (id: string, text: string) => void }) {
  const isCopied = copied === id;
  return (
    <button
      onClick={() => copy(id, text)}
      style={{
        background: isCopied ? "#22C55E22" : "hsl(222,47%,20%)",
        border: `1px solid ${isCopied ? "#22C55E66" : BORDER}`,
        color: isCopied ? "#22C55E" : MUTED,
        borderRadius: 6,
        padding: "3px 10px",
        fontSize: 12,
        cursor: "pointer",
        transition: "all 0.15s",
        flexShrink: 0,
      }}
    >
      {isCopied ? "Copied!" : "Copy"}
    </button>
  );
}

function ProviderBadge({ provider }: { provider: string }) {
  const color = provider === "OpenAI" ? OPENAI_BLUE : provider === "Anthropic" ? ANTHROPIC_ORANGE : BOTH_GRAY;
  return (
    <span style={{
      background: color + "22",
      border: `1px solid ${color}44`,
      color,
      borderRadius: 4,
      padding: "2px 7px",
      fontSize: 11,
      fontWeight: 600,
      letterSpacing: "0.02em",
    }}>{provider}</span>
  );
}

function MethodBadge({ method }: { method: string }) {
  const color = method === "GET" ? GET_GREEN : POST_PURPLE;
  return (
    <span style={{
      background: color + "22",
      border: `1px solid ${color}44`,
      color,
      borderRadius: 4,
      padding: "2px 8px",
      fontSize: 11,
      fontWeight: 700,
      fontFamily: "monospace",
      minWidth: 44,
      textAlign: "center",
    }}>{method}</span>
  );
}

function StatusIndicator() {
  const [status, setStatus] = useState<"checking" | "online" | "offline">("checking");
  useEffect(() => {
    fetch("/api/healthz")
      .then((r) => setStatus(r.ok ? "online" : "offline"))
      .catch(() => setStatus("offline"));
  }, []);
  const color = status === "online" ? "#22C55E" : status === "offline" ? "#EF4444" : "#F59E0B";
  const label = status === "checking" ? "Checking…" : status === "online" ? "Online" : "Offline";
  return (
    <span style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 13, color }}>
      <span style={{
        width: 8, height: 8, borderRadius: "50%", background: color,
        boxShadow: status === "online" ? `0 0 0 3px ${color}33` : undefined,
        display: "inline-block",
      }} />
      {label}
    </span>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section style={{ marginBottom: 32 }}>
      <h2 style={{ color: TEXT, fontSize: 16, fontWeight: 700, marginBottom: 14, letterSpacing: "0.01em" }}>{title}</h2>
      {children}
    </section>
  );
}

function Card({ children, style }: { children: React.ReactNode; style?: React.CSSProperties }) {
  return (
    <div style={{
      background: CARD,
      border: `1px solid ${BORDER}`,
      borderRadius: 10,
      padding: "16px 20px",
      ...style,
    }}>
      {children}
    </div>
  );
}

export default function App() {
  const { copied, copy } = useCopy();
  const baseUrl = window.location.origin;
  const authHeader = `Authorization: Bearer YOUR_PROXY_API_KEY`;
  const curlExample = `curl ${baseUrl}/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer YOUR_PROXY_API_KEY" \\
  -d '{
    "model": "gpt-5.2",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'`;

  return (
    <div style={{ minHeight: "100vh", background: BG, color: TEXT, fontFamily: "system-ui, -apple-system, sans-serif" }}>
      {/* Header */}
      <header style={{
        background: CARD,
        borderBottom: `1px solid ${BORDER}`,
        padding: "16px 24px",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        position: "sticky",
        top: 0,
        zIndex: 10,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <span style={{
            width: 34, height: 34, borderRadius: 8,
            background: "linear-gradient(135deg, #3B82F6, #A855F7)",
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: 16, fontWeight: 700, color: "#fff",
          }}>⚡</span>
          <div>
            <div style={{ fontWeight: 700, fontSize: 16, letterSpacing: "-0.01em" }}>AI Proxy Gateway</div>
            <div style={{ fontSize: 12, color: MUTED }}>OpenAI + Anthropic unified API</div>
          </div>
        </div>
        <StatusIndicator />
      </header>

      <main style={{ maxWidth: 860, margin: "0 auto", padding: "32px 20px" }}>

        {/* Connection Details */}
        <Section title="Connection Details">
          <Card>
            <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
              <div>
                <div style={{ fontSize: 12, color: MUTED, marginBottom: 6, fontWeight: 500 }}>Base URL</div>
                <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                  <code style={{
                    flex: 1, background: "hsl(222,47%,10%)", border: `1px solid ${BORDER}`,
                    borderRadius: 6, padding: "8px 12px", fontSize: 13, color: "#93C5FD",
                    wordBreak: "break-all",
                  }}>{baseUrl}</code>
                  <CopyBtn id="baseurl" text={baseUrl} copied={copied} copy={copy} />
                </div>
              </div>
              <div>
                <div style={{ fontSize: 12, color: MUTED, marginBottom: 6, fontWeight: 500 }}>Authentication — OpenAI-style clients</div>
                <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                  <code style={{
                    flex: 1, background: "hsl(222,47%,10%)", border: `1px solid ${BORDER}`,
                    borderRadius: 6, padding: "8px 12px", fontSize: 13, color: "#C4B5FD",
                  }}>{authHeader}</code>
                  <CopyBtn id="authheader" text={authHeader} copied={copied} copy={copy} />
                </div>
              </div>
              <div>
                <div style={{ fontSize: 12, color: MUTED, marginBottom: 6, fontWeight: 500 }}>Authentication — Anthropic-style clients (x-api-key)</div>
                <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                  <code style={{
                    flex: 1, background: "hsl(222,47%,10%)", border: `1px solid ${BORDER}`,
                    borderRadius: 6, padding: "8px 12px", fontSize: 13, color: "#C4B5FD",
                  }}>x-api-key: YOUR_PROXY_API_KEY</code>
                  <CopyBtn id="xapikey" text="x-api-key: YOUR_PROXY_API_KEY" copied={copied} copy={copy} />
                </div>
                <div style={{ fontSize: 12, color: MUTED, marginTop: 6 }}>
                  Both formats are accepted. Use <code style={{ color: "#FCD34D" }}>x-api-key</code> when configuring Anthropic SDK clients (e.g. VSCode extensions using Anthropic provider type). Use <code style={{ color: "#FCD34D" }}>Authorization: Bearer</code> for OpenAI-compatible clients.
                </div>
              </div>
            </div>
          </Card>
        </Section>

        {/* API Endpoints */}
        <Section title="API Endpoints">
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            {ENDPOINTS.map((ep) => (
              <Card key={ep.path} style={{ padding: "14px 18px" }}>
                <div style={{ display: "flex", alignItems: "flex-start", gap: 12 }}>
                  <MethodBadge method={ep.method} />
                  <div style={{ flex: 1 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap", marginBottom: 4 }}>
                      <code style={{ fontSize: 14, color: "#93C5FD", fontWeight: 600 }}>{ep.path}</code>
                      <ProviderBadge provider={ep.compat} />
                    </div>
                    <div style={{ fontSize: 13, color: MUTED }}>{ep.description}</div>
                  </div>
                  <CopyBtn id={`ep-${ep.path}`} text={`${baseUrl}${ep.path}`} copied={copied} copy={copy} />
                </div>
              </Card>
            ))}
          </div>
        </Section>

        {/* Available Models */}
        <Section title="Available Models">
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))", gap: 10 }}>
            {MODELS.map((m) => (
              <Card key={m.id} style={{ padding: "12px 16px", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                <div>
                  <div style={{ fontSize: 13, fontWeight: 600, color: TEXT, marginBottom: 4 }}>{m.id}</div>
                  <ProviderBadge provider={m.provider} />
                </div>
                <CopyBtn id={`model-${m.id}`} text={m.id} copied={copied} copy={copy} />
              </Card>
            ))}
          </div>
        </Section>

        {/* CherryStudio Setup */}
        <Section title="CherryStudio Setup Guide">
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            {STEPS.map((step, i) => (
              <Card key={i} style={{ display: "flex", alignItems: "flex-start", gap: 14, padding: "14px 18px" }}>
                <span style={{
                  width: 28, height: 28, borderRadius: "50%", flexShrink: 0,
                  background: "linear-gradient(135deg, #3B82F6, #A855F7)",
                  display: "flex", alignItems: "center", justifyContent: "center",
                  fontSize: 13, fontWeight: 700, color: "#fff",
                }}>{i + 1}</span>
                <div>
                  <div style={{ fontWeight: 600, fontSize: 14, color: TEXT, marginBottom: 3 }}>{step.title}</div>
                  <div style={{ fontSize: 13, color: MUTED }}>{step.desc}</div>
                </div>
              </Card>
            ))}
          </div>
        </Section>

        {/* Quick Test */}
        <Section title="Quick Test (curl)">
          <Card style={{ padding: 0, overflow: "hidden" }}>
            <div style={{
              background: "hsl(222,47%,10%)",
              borderBottom: `1px solid ${BORDER}`,
              padding: "10px 16px",
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
            }}>
              <span style={{ fontSize: 12, color: MUTED, fontWeight: 500 }}>bash</span>
              <CopyBtn id="curl" text={curlExample} copied={copied} copy={copy} />
            </div>
            <pre style={{
              margin: 0,
              padding: "16px 20px",
              fontSize: 12.5,
              lineHeight: 1.7,
              overflowX: "auto",
              color: "#E2E8F0",
            }}>
              <span style={{ color: "#94A3B8" }}>curl </span>
              <span style={{ color: "#93C5FD" }}>{baseUrl}/v1/chat/completions</span>
              {` \\\n  `}
              <span style={{ color: "#94A3B8" }}>-H </span>
              <span style={{ color: "#86EFAC" }}>"Content-Type: application/json"</span>
              {` \\\n  `}
              <span style={{ color: "#94A3B8" }}>-H </span>
              <span style={{ color: "#86EFAC" }}>"Authorization: Bearer YOUR_PROXY_API_KEY"</span>
              {` \\\n  `}
              <span style={{ color: "#94A3B8" }}>-d </span>
              <span style={{ color: "#FCD34D" }}>{`'{\n    "model": "gpt-5.2",\n    "messages": [{"role": "user", "content": "Hello!"}]\n  }'`}</span>
            </pre>
          </Card>
        </Section>

        {/* Footer */}
        <footer style={{ textAlign: "center", fontSize: 12, color: MUTED, paddingTop: 16, borderTop: `1px solid ${BORDER}` }}>
          Powered by Replit AI Integrations · OpenAI SDK + Anthropic SDK · No API keys required
        </footer>
      </main>
    </div>
  );
}
