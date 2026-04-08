import { Router } from "express";
import OpenAI from "openai";
import Anthropic from "@anthropic-ai/sdk";
import type { Request, Response } from "express";

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
  const auth = req.headers["authorization"];
  if (!proxyKey || !auth || auth !== `Bearer ${proxyKey}`) {
    res.status(401).json({ error: { message: "Unauthorized", type: "authentication_error", code: 401 } });
    return false;
  }
  return true;
}

/**
 * Strip provider prefix from model names.
 * e.g. "replit/claude-sonnet-4-6" → "claude-sonnet-4-6"
 *      "openai/gpt-5.2"          → "gpt-5.2"
 */
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

/** Send a JSON error only if headers haven't been sent yet. */
function sendError(res: Response, status: number, message: string, type = "api_error") {
  if (res.headersSent) return;
  res.status(status).json({ error: { message, type, code: status } });
}

/** Write an SSE error event when we are already streaming. */
function writeStreamError(res: Response, message: string) {
  try {
    const errChunk = {
      id: `chatcmpl-err-${Date.now()}`,
      object: "chat.completion.chunk",
      created: Math.floor(Date.now() / 1000),
      model: "",
      choices: [{ index: 0, delta: {}, finish_reason: "stop", logprobs: null }],
      error: { message, type: "api_error" },
    };
    res.write(`data: ${JSON.stringify(errChunk)}\n\n`);
    res.write("data: [DONE]\n\n");
  } catch { /* ignore write errors on a closed socket */ }
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

function oaiToolsToAnthropic(tools: OAITool[]): AnthropicTool[] {
  return tools.map((t) => ({
    name: t.function.name,
    description: t.function.description,
    input_schema: (t.function.parameters as Record<string, unknown>) ?? { type: "object", properties: {} },
  }));
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
        contentBlocks.push({ type: "text", text: typeof msg.content === "string" ? msg.content : JSON.stringify(msg.content) });
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
  const message: Record<string, unknown> = {
    role: "assistant",
    content: textParts.join("") || null,
  };
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

function flush(res: Response) {
  (res as unknown as { flush?: () => void }).flush?.();
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

  // Normalize model name — strip provider prefix (e.g. "replit/claude-sonnet-4-6")
  const rawModel = body.model;
  const model = normalizeModel(rawModel ?? "");
  const { messages, stream, tools, tool_choice, ...rest } = body;
  delete (rest as Record<string, unknown>).model;

  if (!model) {
    sendError(res, 400, "model is required", "invalid_request_error");
    return;
  }

  let streamStarted = false;

  try {
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
        streamStarted = true;

        const keepalive = setInterval(() => {
          try { res.write(": keepalive\n\n"); } catch { /* ignore */ }
        }, 5000);

        try {
          const oaiStream = await openai.chat.completions.create({
            ...params,
            stream: true,
          } as Parameters<typeof openai.chat.completions.create>[0]);

          for await (const chunk of oaiStream) {
            res.write(`data: ${JSON.stringify(chunk)}\n\n`);
            flush(res);
          }
          res.write("data: [DONE]\n\n");
        } finally {
          clearInterval(keepalive);
          res.end();
        }
        return;
      }

      const response = await openai.chat.completions.create({
        ...params,
        stream: false,
      } as Parameters<typeof openai.chat.completions.create>[0]);
      res.json(response);
      return;
    }

    // ── Anthropic path ───────────────────────────────────────────────────────
    if (isAnthropicModel(model)) {
      const { system, messages: anthropicMessages } = oaiMessagesToAnthropic(messages);
      const anthropicParams: Record<string, unknown> = {
        model,
        max_tokens: body.max_tokens ?? body.max_completion_tokens ?? 8192,
        messages: anthropicMessages,
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
        streamStarted = true;

        const keepalive = setInterval(() => {
          try { res.write(": keepalive\n\n"); } catch { /* ignore */ }
        }, 5000);

        req.on("close", () => clearInterval(keepalive));

        try {
          let currentToolId = "";
          let currentToolName = "";
          let toolIndex = -1;
          let blockIdx = -1;
          let hasText = false;
          const chatId = `chatcmpl-${Date.now()}`;

          const sendChunk = (delta: Record<string, unknown>, finishReason: string | null = null) => {
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

          // Send the initial role chunk so clients know content is coming
          sendChunk({ role: "assistant", content: "" });

          const anthropicStream = anthropic.messages.stream(
            anthropicParams as Parameters<typeof anthropic.messages.stream>[0],
          );

          for await (const event of anthropicStream) {
            const e = event as Record<string, unknown>;
            const type = e.type as string;

            if (type === "content_block_start") {
              const block = e.content_block as { type: string; id?: string; name?: string };
              if (block.type === "text") {
                blockIdx++;
                hasText = true;
              } else if (block.type === "tool_use") {
                toolIndex++;
                blockIdx++;
                currentToolId = block.id ?? "";
                currentToolName = block.name ?? "";
                sendChunk({
                  tool_calls: [{
                    index: toolIndex,
                    id: currentToolId,
                    type: "function",
                    function: { name: currentToolName, arguments: "" },
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
                const finishReason = d.stop_reason === "tool_use" ? "tool_calls" : "stop";
                sendChunk({}, finishReason);
              }
            }
          }

          void hasText; void currentToolId; void currentToolName; void blockIdx;

          res.write("data: [DONE]\n\n");
        } finally {
          clearInterval(keepalive);
          res.end();
        }
        return;
      }

      // Non-stream: always buffer internally via stream to avoid upstream timeouts
      const finalMsg = await anthropic.messages.stream(
        anthropicParams as Parameters<typeof anthropic.messages.stream>[0],
      ).finalMessage();
      res.json(anthropicToOAIResponse(finalMsg as AnthropicFinalMessage, model));
      return;
    }

    sendError(res, 400, `Unknown model: ${model}. Supported: ${ALL_MODELS.map((m) => m.id).join(", ")}`, "invalid_request_error");
  } catch (err: unknown) {
    const e = err as { status?: number; message?: string };
    if (streamStarted) {
      writeStreamError(res, e.message ?? "Internal server error");
      res.end();
    } else {
      sendError(res, e.status ?? 500, e.message ?? "Internal server error");
    }
  }
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

  // Normalize model name
  const rawModel = body.model;
  const model = normalizeModel(rawModel ?? "");
  const { messages, system, tools, tool_choice, max_tokens, stream, ...rest } = body;
  delete (rest as Record<string, unknown>).model;

  if (!model) {
    sendError(res, 400, "model is required", "invalid_request_error");
    return;
  }

  let streamStarted = false;

  try {
    // ── Claude model — forward to Anthropic ──────────────────────────────────
    if (isAnthropicModel(model)) {
      const params: Record<string, unknown> = {
        model,
        messages,
        max_tokens: max_tokens ?? 8192,
        ...rest,
      };
      if (system) params.system = system;
      if (tools) params.tools = tools;
      if (tool_choice) params.tool_choice = tool_choice;

      if (stream) {
        res.setHeader("Content-Type", "text/event-stream");
        res.setHeader("Cache-Control", "no-cache");
        res.setHeader("Connection", "keep-alive");
        res.setHeader("X-Accel-Buffering", "no");
        res.flushHeaders();
        streamStarted = true;

        const keepalive = setInterval(() => {
          try { res.write(": keepalive\n\n"); } catch { /* ignore */ }
        }, 5000);
        req.on("close", () => clearInterval(keepalive));

        try {
          const anthropicStream = anthropic.messages.stream(
            params as Parameters<typeof anthropic.messages.stream>[0],
          );
          for await (const event of anthropicStream) {
            const e = event as { type: string };
            res.write(`event: ${e.type}\ndata: ${JSON.stringify(event)}\n\n`);
            flush(res);
          }
        } finally {
          clearInterval(keepalive);
          res.end();
        }
        return;
      }

      const msg = await anthropic.messages.stream(
        params as Parameters<typeof anthropic.messages.stream>[0],
      ).finalMessage();
      res.json(msg);
      return;
    }

    // ── OpenAI model — convert Anthropic → OAI → back ───────────────────────
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
        streamStarted = true;

        const keepalive = setInterval(() => {
          try { res.write(": keepalive\n\n"); } catch { /* ignore */ }
        }, 5000);
        req.on("close", () => clearInterval(keepalive));

        const msgId = `msg_${Date.now()}`;
        let outputTokens = 0;

        const sendEvent = (type: string, data: unknown) => {
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
              if (hasText) {
                sendEvent("content_block_stop", { type: "content_block_stop", index: textBlockIdx });
              }
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

            if (chunk.usage) {
              outputTokens = chunk.usage.completion_tokens ?? 0;
            }
          }

          void hasText;
        } finally {
          clearInterval(keepalive);
          res.end();
        }
        return;
      }

      // Non-stream OAI
      const oaiResponse = await openai.chat.completions.create(
        oaiParams as Parameters<typeof openai.chat.completions.create>[0],
      );
      const choice = oaiResponse.choices[0];
      const contentBlocks: unknown[] = [];
      if (choice?.message?.content) {
        contentBlocks.push({ type: "text", text: choice.message.content });
      }
      if (choice?.message?.tool_calls) {
        for (const tc of choice.message.tool_calls) {
          let input: unknown = {};
          try { input = JSON.parse(tc.function.arguments); } catch { input = {}; }
          contentBlocks.push({ type: "tool_use", id: tc.id, name: tc.function.name, input });
        }
      }
      const stopReason = choice?.finish_reason === "tool_calls" ? "tool_use" : "end_turn";
      res.json({
        id: `msg_${oaiResponse.id}`,
        type: "message",
        role: "assistant",
        model,
        content: contentBlocks,
        stop_reason: stopReason,
        stop_sequence: null,
        usage: {
          input_tokens: oaiResponse.usage?.prompt_tokens ?? 0,
          output_tokens: oaiResponse.usage?.completion_tokens ?? 0,
        },
      });
      return;
    }

    sendError(res, 400, `Unknown model: ${model}. Supported: ${ALL_MODELS.map((m) => m.id).join(", ")}`, "invalid_request_error");
  } catch (err: unknown) {
    const e = err as { status?: number; message?: string };
    if (streamStarted) {
      writeStreamError(res, e.message ?? "Internal server error");
      res.end();
    } else {
      sendError(res, e.status ?? 500, e.message ?? "Internal server error");
    }
  }
});

export default router;
