import { Router } from "express";
import OpenAI from "openai";
import Anthropic from "@anthropic-ai/sdk";
import type { Request, Response } from "express";
import { logger } from "../lib/logger";

const router = Router();

const openai = new OpenAI({
  baseURL: process.env.AI_INTEGRATIONS_OPENAI_BASE_URL,
  apiKey: process.env.AI_INTEGRATIONS_OPENAI_API_KEY ?? "dummy",
});

const anthropic = new Anthropic({
  baseURL: process.env.AI_INTEGRATIONS_ANTHROPIC_BASE_URL,
  apiKey: process.env.AI_INTEGRATIONS_ANTHROPIC_API_KEY ?? "dummy",
});

const OPENAI_MODELS = [
  { id: "gpt-5.2", provider: "openai" },
  { id: "gpt-5-mini", provider: "openai" },
  { id: "gpt-5-nano", provider: "openai" },
  { id: "o4-mini", provider: "openai" },
  { id: "o3", provider: "openai" },
];

const ANTHROPIC_MODELS = [
  { id: "claude-opus-4-6", provider: "anthropic" },
  { id: "claude-sonnet-4-6", provider: "anthropic" },
  { id: "claude-haiku-4-5", provider: "anthropic" },
];

const ALL_MODELS = [...OPENAI_MODELS, ...ANTHROPIC_MODELS];

function verifyBearer(req: Request, res: Response): boolean {
  const proxyKey = process.env.PROXY_API_KEY;
  if (!proxyKey) {
    res.status(500).json({ error: { message: "PROXY_API_KEY not configured", type: "api_error", code: 500 } });
    return false;
  }
  const authHeader = req.headers["authorization"];
  const xApiKey = req.headers["x-api-key"];
  const validBearer = authHeader === `Bearer ${proxyKey}`;
  const validApiKey = xApiKey === proxyKey;
  if (!validBearer && !validApiKey) {
    res.status(401).json({ error: { message: "Unauthorized", type: "authentication_error", code: 401 } });
    return false;
  }
  return true;
}

/** Strip provider prefix: "replit/claude-sonnet-4-6" → "claude-sonnet-4-6" */
function normalizeModel(model: string): string {
  const slash = model.indexOf("/");
  return slash >= 0 ? model.slice(slash + 1) : model;
}

function isOpenAIModel(model: string): boolean {
  return model.startsWith("gpt-") || model.startsWith("o");
}

function isAnthropicModel(model: string): boolean {
  return model.startsWith("claude-");
}

function flush(res: Response) {
  (res as unknown as { flush?: () => void }).flush?.();
}

/** Send JSON error only if headers haven't been sent. Logs every error for debugging. */
function sendError(res: Response, status: number, message: string, type = "api_error", context?: Record<string, unknown>) {
  if (res.headersSent) return;
  logger.error({ status, type, ...context }, `proxy error: ${message}`);
  res.status(status).json({ error: { message, type, code: status } });
}

/**
 * Safely write an SSE error event. Checks writableEnded / destroyed before writing.
 * Swallows all write errors — never throws.
 */
function writeStreamError(res: Response, message: string) {
  if (res.writableEnded || res.destroyed) return;
  try {
    const chunk = {
      id: `chatcmpl-err-${Date.now()}`,
      object: "chat.completion.chunk",
      created: Math.floor(Date.now() / 1000),
      model: "",
      choices: [{ index: 0, delta: {}, finish_reason: "stop", logprobs: null }],
      error: { message, type: "api_error" },
    };
    res.write(`data: ${JSON.stringify(chunk)}\n\n`);
    res.write("data: [DONE]\n\n");
  } catch { /* socket already gone */ }
}

/** Safely end the response — only if not already ended. */
function safeEnd(res: Response) {
  if (!res.writableEnded && !res.destroyed) {
    try { res.end(); } catch { /* ignore */ }
  }
}

// ── Tool format converters ───────────────────────────────────────────────────

type OAITool = {
  type: "function";
  function: { name: string; description?: string; parameters?: Record<string, unknown> };
};

type AnthropicTool = {
  name: string;
  description?: string;
  input_schema: Record<string, unknown>;
};

/** Sanitize a tool name to match Anthropic's pattern ^[a-zA-Z0-9_-]{1,128}$ */
function sanitizeToolName(name: string): string {
  const sanitized = (name ?? "tool").replace(/[^a-zA-Z0-9_-]/g, "_").slice(0, 128);
  return sanitized.length > 0 ? sanitized : "tool";
}

function oaiToolsToAnthropic(tools: OAITool[]): AnthropicTool[] {
  return tools.map((t) => ({
    name: sanitizeToolName(t.function.name),
    description: t.function.description,
    input_schema: (t.function.parameters as Record<string, unknown>) ?? { type: "object", properties: {} },
  }));
}

/** Sanitize all tool names in an Anthropic-format tools array */
function sanitizeAnthropicTools(tools: AnthropicTool[]): AnthropicTool[] {
  return tools.map((t) => ({ ...t, name: sanitizeToolName(t.name) }));
}

function anthropicToolsToOAI(tools: AnthropicTool[]): OAITool[] {
  return tools.map((t) => ({
    type: "function" as const,
    function: { name: t.name, description: t.description, parameters: t.input_schema },
  }));
}

type OAIToolChoice = string | { type: "function"; function: { name: string } };
type AnthropicToolChoice = { type: string; name?: string };

function oaiToolChoiceToAnthropic(tc: OAIToolChoice): AnthropicToolChoice {
  if (typeof tc === "string") {
    if (tc === "required") return { type: "any" };
    if (tc === "none") return { type: "none" };
    return { type: "auto" };
  }
  return { type: "tool", name: tc.function.name };
}

function anthropicToolChoiceToOAI(tc: AnthropicToolChoice): OAIToolChoice {
  if (tc.type === "any") return "required";
  if (tc.type === "none") return "none";
  if (tc.type === "tool" && tc.name) return { type: "function", function: { name: tc.name } };
  return "auto";
}

// ── Anthropic message repair ─────────────────────────────────────────────────

/**
 * Anthropic requires that every `tool_use` block in an assistant message is
 * immediately followed by a user message containing a `tool_result` block for
 * each `tool_use` id.  Clients (especially IDE extensions) often omit these
 * when the user continues a conversation without waiting for tool execution.
 * This function scans the messages array and injects synthetic empty
 * `tool_result` blocks wherever they are missing so Anthropic accepts the request.
 */
function repairToolResults(messages: AnthropicMessage[]): AnthropicMessage[] {
  type Block = { type: string; id?: string; tool_use_id?: string; [k: string]: unknown };

  const result: AnthropicMessage[] = [];
  let i = 0;

  while (i < messages.length) {
    const msg = messages[i];
    result.push(msg);
    i++;

    if (msg.role !== "assistant") continue;
    const blocks = Array.isArray(msg.content) ? (msg.content as Block[]) : [];
    const toolUseIds = blocks.filter((b) => b.type === "tool_use" && b.id).map((b) => b.id as string);
    if (toolUseIds.length === 0) continue;

    // Find which ids already have a tool_result in the next user message
    const next = messages[i];
    const nextBlocks: Block[] = Array.isArray(next?.content) ? (next.content as Block[]) : [];
    const coveredIds = new Set(
      nextBlocks.filter((b) => b.type === "tool_result" && b.tool_use_id).map((b) => b.tool_use_id as string),
    );
    const missingIds = toolUseIds.filter((id) => !coveredIds.has(id));
    if (missingIds.length === 0) continue;

    const syntheticResults: Block[] = missingIds.map((id) => ({
      type: "tool_result",
      tool_use_id: id,
      content: "",
    }));

    if (next?.role === "user") {
      // Prepend synthetic results to the existing user message and skip it
      result.push({ role: "user", content: [...syntheticResults, ...nextBlocks] } as AnthropicMessage);
      i++; // consume the original next message
    } else {
      // No following user message — inject a synthetic one
      result.push({ role: "user", content: syntheticResults } as AnthropicMessage);
    }
  }

  return result;
}

// ── Message converters (OAI → Anthropic) ────────────────────────────────────

type OAIMessage = {
  role: string;
  content: unknown;
  tool_calls?: Array<{ id: string; function: { name: string; arguments: string }; type: string }>;
  tool_call_id?: string;
  name?: string;
};

type AnthropicMessage = {
  role: "user" | "assistant";
  content: unknown;
};

function oaiMessagesToAnthropic(messages: OAIMessage[]): {
  system: string | undefined;
  messages: AnthropicMessage[];
} {
  let system: string | undefined;
  const result: AnthropicMessage[] = [];

  for (const msg of messages) {
    if (msg.role === "system") {
      system = typeof msg.content === "string" ? msg.content : JSON.stringify(msg.content);
      continue;
    }

    if (msg.role === "assistant") {
      const contentBlocks: unknown[] = [];
      if (msg.content && msg.content !== "") {
        contentBlocks.push({
          type: "text",
          text: typeof msg.content === "string" ? msg.content : JSON.stringify(msg.content),
        });
      }
      if (msg.tool_calls) {
        for (const tc of msg.tool_calls) {
          let input: unknown = {};
          try { input = JSON.parse(tc.function.arguments); } catch { input = {}; }
          contentBlocks.push({ type: "tool_use", id: tc.id, name: tc.function.name, input });
        }
      }
      result.push({
        role: "assistant",
        content: contentBlocks.length === 1 && !msg.tool_calls ? contentBlocks[0] : contentBlocks,
      });
      continue;
    }

    if (msg.role === "tool") {
      const lastUser = result[result.length - 1];
      const toolBlock = {
        type: "tool_result",
        tool_use_id: msg.tool_call_id ?? "",
        content: typeof msg.content === "string" ? msg.content : JSON.stringify(msg.content),
      };
      if (lastUser && lastUser.role === "user") {
        if (Array.isArray(lastUser.content)) {
          (lastUser.content as unknown[]).push(toolBlock);
        } else {
          lastUser.content = [lastUser.content, toolBlock];
        }
      } else {
        result.push({ role: "user", content: [toolBlock] });
      }
      continue;
    }

    result.push({ role: "user", content: typeof msg.content === "string" ? msg.content : msg.content });
  }

  return { system, messages: result };
}

// ── Anthropic Message → OAI response ────────────────────────────────────────

type AnthropicFinalMessage = {
  id: string;
  model: string;
  stop_reason: string | null;
  usage: { input_tokens: number; output_tokens: number };
  content: Array<{ type: string; text?: string; id?: string; name?: string; input?: unknown }>;
};

function anthropicToOAIResponse(msg: AnthropicFinalMessage, requestedModel: string): Record<string, unknown> {
  const toolCalls: unknown[] = [];
  const textParts: string[] = [];

  for (const block of msg.content) {
    if (block.type === "text") textParts.push(block.text ?? "");
    if (block.type === "tool_use") {
      toolCalls.push({
        id: block.id,
        type: "function",
        function: { name: block.name, arguments: JSON.stringify(block.input ?? {}) },
      });
    }
  }

  const finishReason = msg.stop_reason === "tool_use" ? "tool_calls" : "stop";
  const message: Record<string, unknown> = { role: "assistant", content: textParts.join("") || null };
  if (toolCalls.length > 0) {
    message.tool_calls = toolCalls;
    message.content = textParts.join("") || null;
  }

  return {
    id: `chatcmpl-${msg.id}`,
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model: requestedModel,
    choices: [{ index: 0, message, finish_reason: finishReason, logprobs: null }],
    usage: {
      prompt_tokens: msg.usage.input_tokens,
      completion_tokens: msg.usage.output_tokens,
      total_tokens: msg.usage.input_tokens + msg.usage.output_tokens,
    },
  };
}

// ── GET /v1/models ───────────────────────────────────────────────────────────

router.get("/models", (req: Request, res: Response) => {
  if (!verifyBearer(req, res)) return;
  const data = ALL_MODELS.map((m) => ({
    id: m.id,
    object: "model",
    created: 1700000000,
    owned_by: m.provider,
  }));
  res.json({ object: "list", data });
});

// ── POST /v1/chat/completions ────────────────────────────────────────────────

router.post("/chat/completions", async (req: Request, res: Response) => {
  if (!verifyBearer(req, res)) return;

  const body = req.body as {
    model: string;
    messages: OAIMessage[];
    stream?: boolean;
    tools?: OAITool[];
    tool_choice?: OAIToolChoice;
    max_tokens?: number;
    max_completion_tokens?: number;
    temperature?: number;
    [key: string]: unknown;
  };

  const model = normalizeModel(body.model ?? "");
  const { messages, stream, tools, tool_choice, ...rest } = body;
  delete (rest as Record<string, unknown>).model;

  if (!model) {
    sendError(res, 400, "model is required", "invalid_request_error", { rawModel: body.model });
    return;
  }

  logger.info({ model, stream: body.stream, endpoint: "chat/completions" }, "proxy request");

  // ── OpenAI path ──────────────────────────────────────────────────────────
  if (isOpenAIModel(model)) {
    const params: Record<string, unknown> = { model, messages, ...rest };
    if (tools) params.tools = tools;
    if (tool_choice) params.tool_choice = tool_choice;

    if (stream) {
      res.setHeader("Content-Type", "text/event-stream");
      res.setHeader("Cache-Control", "no-cache");
      res.setHeader("Connection", "keep-alive");
      res.setHeader("X-Accel-Buffering", "no");
      res.flushHeaders();

      const keepalive = setInterval(() => {
        if (!res.writableEnded) try { res.write(": keepalive\n\n"); } catch { /* ignore */ }
      }, 5000);

      try {
        const oaiStream = await openai.chat.completions.create({
          ...params,
          stream: true,
        } as Parameters<typeof openai.chat.completions.create>[0]);

        for await (const chunk of oaiStream) {
          if (res.writableEnded) break;
          res.write(`data: ${JSON.stringify(chunk)}\n\n`);
          flush(res);
        }
        if (!res.writableEnded) res.write("data: [DONE]\n\n");
      } catch (err) {
        const e = err as { message?: string };
        writeStreamError(res, e.message ?? "Stream error");
      } finally {
        clearInterval(keepalive);
        safeEnd(res);
      }
      return;
    }

    try {
      const response = await openai.chat.completions.create({
        ...params,
        stream: false,
      } as Parameters<typeof openai.chat.completions.create>[0]);
      res.json(response);
    } catch (err) {
      const e = err as { status?: number; message?: string };
      sendError(res, e.status ?? 500, e.message ?? "OpenAI error");
    }
    return;
  }

  // ── Anthropic path ───────────────────────────────────────────────────────
  if (isAnthropicModel(model)) {
    const { system, messages: anthropicMessages } = oaiMessagesToAnthropic(messages);
    const anthropicParams: Record<string, unknown> = {
      model,
      max_tokens: body.max_tokens ?? body.max_completion_tokens ?? 8192,
      messages: repairToolResults(anthropicMessages),
    };
    if (system) anthropicParams.system = system;
    if (tools) anthropicParams.tools = oaiToolsToAnthropic(tools);
    if (tool_choice) anthropicParams.tool_choice = oaiToolChoiceToAnthropic(tool_choice);

    if (stream) {
      res.setHeader("Content-Type", "text/event-stream");
      res.setHeader("Cache-Control", "no-cache");
      res.setHeader("Connection", "keep-alive");
      res.setHeader("X-Accel-Buffering", "no");
      res.flushHeaders();

      const keepalive = setInterval(() => {
        if (!res.writableEnded) try { res.write(": keepalive\n\n"); } catch { /* ignore */ }
      }, 5000);

      req.on("close", () => clearInterval(keepalive));

      try {
        const chatId = `chatcmpl-${Date.now()}`;
        let toolIndex = -1;

        const sendChunk = (delta: Record<string, unknown>, finishReason: string | null = null) => {
          if (res.writableEnded) return;
          const chunk = {
            id: chatId,
            object: "chat.completion.chunk",
            created: Math.floor(Date.now() / 1000),
            model,
            choices: [{ index: 0, delta, finish_reason: finishReason, logprobs: null }],
          };
          res.write(`data: ${JSON.stringify(chunk)}\n\n`);
          flush(res);
        };

        sendChunk({ role: "assistant", content: "" });

        const anthropicStream = anthropic.messages.stream(
          anthropicParams as Parameters<typeof anthropic.messages.stream>[0],
        );

        for await (const event of anthropicStream) {
          if (res.writableEnded) break;
          const e = event as Record<string, unknown>;
          const type = e.type as string;

          if (type === "content_block_start") {
            const block = e.content_block as { type: string; id?: string; name?: string };
            if (block.type === "tool_use") {
              toolIndex++;
              sendChunk({
                tool_calls: [{
                  index: toolIndex,
                  id: block.id ?? "",
                  type: "function",
                  function: { name: block.name ?? "", arguments: "" },
                }],
              });
            }
          } else if (type === "content_block_delta") {
            const delta = e.delta as { type: string; text?: string; partial_json?: string };
            if (delta.type === "text_delta" && delta.text) {
              sendChunk({ content: delta.text });
            } else if (delta.type === "input_json_delta" && delta.partial_json !== undefined) {
              sendChunk({
                tool_calls: [{ index: toolIndex, function: { arguments: delta.partial_json } }],
              });
            }
          } else if (type === "message_delta") {
            const d = e.delta as { stop_reason?: string };
            if (d.stop_reason) {
              sendChunk({}, d.stop_reason === "tool_use" ? "tool_calls" : "stop");
            }
          }
        }

        if (!res.writableEnded) res.write("data: [DONE]\n\n");
      } catch (err) {
        const e = err as { message?: string };
        writeStreamError(res, e.message ?? "Stream error");
      } finally {
        clearInterval(keepalive);
        safeEnd(res);
      }
      return;
    }

    // Non-stream: buffer internally via stream() to avoid upstream timeouts
    try {
      const finalMsg = await anthropic.messages.stream(
        anthropicParams as Parameters<typeof anthropic.messages.stream>[0],
      ).finalMessage();
      res.json(anthropicToOAIResponse(finalMsg as AnthropicFinalMessage, model));
    } catch (err) {
      const e = err as { status?: number; message?: string };
      sendError(res, e.status ?? 500, e.message ?? "Anthropic error");
    }
    return;
  }

  sendError(res, 400, `Unknown model: ${model}. Supported: ${ALL_MODELS.map((m) => m.id).join(", ")}`, "invalid_request_error", { model });
});

// ── POST /v1/messages (Anthropic native) ─────────────────────────────────────

router.post("/messages", async (req: Request, res: Response) => {
  if (!verifyBearer(req, res)) return;

  const body = req.body as {
    model: string;
    messages: AnthropicMessage[];
    system?: string;
    tools?: AnthropicTool[];
    tool_choice?: AnthropicToolChoice;
    max_tokens?: number;
    stream?: boolean;
    [key: string]: unknown;
  };

  const model = normalizeModel(body.model ?? "");
  const { messages, system, tools, tool_choice, max_tokens, stream, ...rest } = body;
  delete (rest as Record<string, unknown>).model;

  if (!model) {
    sendError(res, 400, "model is required", "invalid_request_error", { rawModel: body.model });
    return;
  }

  logger.info({ model, stream: body.stream, endpoint: "messages" }, "proxy request");

  // ── Claude model — forward to Anthropic ──────────────────────────────────
  if (isAnthropicModel(model)) {
    const params: Record<string, unknown> = {
      model,
      messages: repairToolResults(messages),
      max_tokens: max_tokens ?? 8192,
      ...rest,
    };
    if (system) params.system = system;
    if (tools) params.tools = sanitizeAnthropicTools(tools);
    if (tool_choice) params.tool_choice = tool_choice?.name ? { ...tool_choice, name: sanitizeToolName(tool_choice.name) } : tool_choice;

    if (stream) {
      res.setHeader("Content-Type", "text/event-stream");
      res.setHeader("Cache-Control", "no-cache");
      res.setHeader("Connection", "keep-alive");
      res.setHeader("X-Accel-Buffering", "no");
      res.flushHeaders();

      const keepalive = setInterval(() => {
        if (!res.writableEnded) try { res.write(": keepalive\n\n"); } catch { /* ignore */ }
      }, 5000);
      req.on("close", () => clearInterval(keepalive));

      try {
        const anthropicStream = anthropic.messages.stream(
          params as Parameters<typeof anthropic.messages.stream>[0],
        );
        for await (const event of anthropicStream) {
          if (res.writableEnded) break;
          const e = event as { type: string };
          res.write(`event: ${e.type}\ndata: ${JSON.stringify(event)}\n\n`);
          flush(res);
        }
      } catch (err) {
        const e = err as { status?: number; message?: string; error?: unknown };
        logger.error({ model, status: e.status, upstream: e.error ?? e.message }, "anthropic stream error");
        if (!res.writableEnded) {
          try { res.write(`event: error\ndata: ${JSON.stringify({ type: "error", error: { type: "api_error", message: e.message ?? "Stream error" } })}\n\n`); } catch { /* ignore */ }
        }
      } finally {
        clearInterval(keepalive);
        safeEnd(res);
      }
      return;
    }

    try {
      const msg = await anthropic.messages.stream(
        params as Parameters<typeof anthropic.messages.stream>[0],
      ).finalMessage();
      res.json(msg);
    } catch (err) {
      const e = err as { status?: number; message?: string; error?: unknown };
      sendError(res, e.status ?? 500, e.message ?? "Anthropic error", "api_error", { model, upstream: e.error ?? e.message });
    }
    return;
  }

  // ── OpenAI model via /v1/messages — convert Anthropic → OAI → back ───────
  if (isOpenAIModel(model)) {
    const oaiMessages: OAIMessage[] = [];
    if (system) oaiMessages.push({ role: "system", content: system });

    for (const msg of messages) {
      if (msg.role === "user") {
        if (Array.isArray(msg.content)) {
          const toolResults = (msg.content as Array<{ type: string; tool_use_id?: string; content?: string }>)
            .filter((b) => b.type === "tool_result");
          const textBlocks = (msg.content as Array<{ type: string; text?: string }>)
            .filter((b) => b.type === "text");

          for (const tr of toolResults) {
            oaiMessages.push({ role: "tool", tool_call_id: tr.tool_use_id ?? "", content: tr.content ?? "" });
          }
          if (textBlocks.length > 0) {
            oaiMessages.push({ role: "user", content: textBlocks.map((b) => b.text).join("") });
          }
        } else {
          oaiMessages.push({ role: "user", content: msg.content as string });
        }
      } else if (msg.role === "assistant") {
        if (Array.isArray(msg.content)) {
          const textParts = (msg.content as Array<{ type: string; text?: string }>)
            .filter((b) => b.type === "text").map((b) => b.text).join("");
          const toolUses = (msg.content as Array<{ type: string; id?: string; name?: string; input?: unknown }>)
            .filter((b) => b.type === "tool_use");
          const oaiMsg: OAIMessage = { role: "assistant", content: textParts || null };
          if (toolUses.length > 0) {
            oaiMsg.tool_calls = toolUses.map((tu) => ({
              id: tu.id ?? "",
              type: "function",
              function: { name: tu.name ?? "", arguments: JSON.stringify(tu.input ?? {}) },
            }));
          }
          oaiMessages.push(oaiMsg);
        } else {
          oaiMessages.push({ role: "assistant", content: msg.content as string });
        }
      }
    }

    const oaiParams: Record<string, unknown> = { model, messages: oaiMessages };
    if (tools) oaiParams.tools = anthropicToolsToOAI(tools);
    if (tool_choice) oaiParams.tool_choice = anthropicToolChoiceToOAI(tool_choice);
    if (max_tokens) oaiParams.max_completion_tokens = max_tokens;

    if (stream) {
      res.setHeader("Content-Type", "text/event-stream");
      res.setHeader("Cache-Control", "no-cache");
      res.setHeader("Connection", "keep-alive");
      res.setHeader("X-Accel-Buffering", "no");
      res.flushHeaders();

      const keepalive = setInterval(() => {
        if (!res.writableEnded) try { res.write(": keepalive\n\n"); } catch { /* ignore */ }
      }, 5000);
      req.on("close", () => clearInterval(keepalive));

      const msgId = `msg_${Date.now()}`;
      let outputTokens = 0;

      const sendEvent = (type: string, data: unknown) => {
        if (res.writableEnded) return;
        res.write(`event: ${type}\ndata: ${JSON.stringify(data)}\n\n`);
        flush(res);
      };

      try {
        const oaiStream = await openai.chat.completions.create({
          ...oaiParams,
          stream: true,
        } as Parameters<typeof openai.chat.completions.create>[0]);

        sendEvent("message_start", {
          type: "message_start",
          message: {
            id: msgId,
            type: "message",
            role: "assistant",
            model,
            content: [],
            stop_reason: null,
            stop_sequence: null,
            usage: { input_tokens: 0, output_tokens: 0 },
          },
        });

        let textBlockIdx = -1;
        const toolBlockIndices: Record<string, number> = {};
        let blockIdx = -1;
        let hasText = false;

        for await (const chunk of oaiStream) {
          if (res.writableEnded) break;
          const delta = chunk.choices[0]?.delta;
          if (!delta) continue;

          if (delta.content) {
            if (!hasText) {
              hasText = true;
              blockIdx++;
              textBlockIdx = blockIdx;
              sendEvent("content_block_start", {
                type: "content_block_start",
                index: textBlockIdx,
                content_block: { type: "text", text: "" },
              });
              sendEvent("ping", { type: "ping" });
            }
            sendEvent("content_block_delta", {
              type: "content_block_delta",
              index: textBlockIdx,
              delta: { type: "text_delta", text: delta.content },
            });
            outputTokens++;
          }

          if (delta.tool_calls) {
            for (const tc of delta.tool_calls) {
              const tcId = tc.id ?? "";
              if (tcId && !(tcId in toolBlockIndices)) {
                blockIdx++;
                toolBlockIndices[tcId] = blockIdx;
                sendEvent("content_block_start", {
                  type: "content_block_start",
                  index: blockIdx,
                  content_block: { type: "tool_use", id: tcId, name: tc.function?.name ?? "", input: {} },
                });
              }
              const idx = tc.id
                ? toolBlockIndices[tc.id]
                : Object.values(toolBlockIndices).at(-1) ?? blockIdx;
              if (tc.function?.arguments) {
                sendEvent("content_block_delta", {
                  type: "content_block_delta",
                  index: idx,
                  delta: { type: "input_json_delta", partial_json: tc.function.arguments },
                });
              }
            }
          }

          if (chunk.choices[0]?.finish_reason) {
            if (hasText) sendEvent("content_block_stop", { type: "content_block_stop", index: textBlockIdx });
            for (const [, i] of Object.entries(toolBlockIndices)) {
              sendEvent("content_block_stop", { type: "content_block_stop", index: i });
            }
            const stopReason = chunk.choices[0].finish_reason === "tool_calls" ? "tool_use" : "end_turn";
            sendEvent("message_delta", {
              type: "message_delta",
              delta: { stop_reason: stopReason, stop_sequence: null },
              usage: { output_tokens: outputTokens },
            });
            sendEvent("message_stop", { type: "message_stop" });
          }

          if (chunk.usage) outputTokens = chunk.usage.completion_tokens ?? 0;
        }
      } catch (err) {
        const e = err as { message?: string };
        if (!res.writableEnded) {
          try { res.write(`event: error\ndata: ${JSON.stringify({ type: "error", error: { type: "api_error", message: e.message ?? "Stream error" } })}\n\n`); } catch { /* ignore */ }
        }
      } finally {
        clearInterval(keepalive);
        safeEnd(res);
      }
      return;
    }

    // Non-stream OAI → Anthropic format
    try {
      const oaiResponse = await openai.chat.completions.create(
        oaiParams as Parameters<typeof openai.chat.completions.create>[0],
      );
      const choice = oaiResponse.choices[0];
      const contentBlocks: unknown[] = [];
      if (choice?.message?.content) contentBlocks.push({ type: "text", text: choice.message.content });
      if (choice?.message?.tool_calls) {
        for (const tc of choice.message.tool_calls) {
          let input: unknown = {};
          try { input = JSON.parse(tc.function.arguments); } catch { input = {}; }
          contentBlocks.push({ type: "tool_use", id: tc.id, name: tc.function.name, input });
        }
      }
      res.json({
        id: `msg_${oaiResponse.id}`,
        type: "message",
        role: "assistant",
        model,
        content: contentBlocks,
        stop_reason: choice?.finish_reason === "tool_calls" ? "tool_use" : "end_turn",
        stop_sequence: null,
        usage: {
          input_tokens: oaiResponse.usage?.prompt_tokens ?? 0,
          output_tokens: oaiResponse.usage?.completion_tokens ?? 0,
        },
      });
    } catch (err) {
      const e = err as { status?: number; message?: string };
      sendError(res, e.status ?? 500, e.message ?? "OpenAI error");
    }
    return;
  }

  sendError(res, 400, `Unknown model: ${model}. Supported: ${ALL_MODELS.map((m) => m.id).join(", ")}`, "invalid_request_error", { model });
});

export default router;
