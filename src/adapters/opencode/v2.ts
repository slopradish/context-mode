/**
 * OpenCode 2 (Effect plugin API) adapter for context-mode.
 *
 * OpenCode 2 intentionally broke the plugin API (docs/build/plugins/migrate-v1):
 * plugins now default-export `Plugin.define({ id, setup })` instead of the V1
 * hooks-factory contract. Loading the V1 module fails with
 * `Plugin must export a default definition with an id and an effect or setup
 * function` — the loader never calls `server()`.
 *
 * This module implements the V2 `setup(ctx)` against the same shared state
 * bootstrap as the V1 adapter (core.ts), with one hook-registration per V1
 * hook, per the official migration table:
 *
 *   V1                                    V2
 *   event                                 ctx.event.subscribe()
 *   tool map                              ctx.tool.transform(...)
 *   tool.execute.before                   ctx.tool.hook("execute.before", ...)
 *   tool.execute.after                    ctx.tool.hook("execute.after", ...)
 *   chat.message                          ctx.session.hook("prompt", ...)
 *   experimental.chat.system.transform    ctx.session.hook("context", ...) + event.system
 *   experimental.session.compacting       ctx.session.hook("compaction", ...)
 *
 * Deliberate differences:
 *   - V1 pushed the compaction snapshot into `output.context` (the compaction
 *     request's own system strings). V2 has no per-request context mutation,
 *     so the snapshot rides into the summarizer request via `event.system`
 *     instead — the summary then carries it into the compacted history.
 *   - V1's generic `event` hook delivered `message.updated` with cumulative
 *     `.cost` and a last-step token snapshot. V2 has no message bus; usage is
 *     taken from the `session.usage.updated` event (sessionID + cost + tokens).
 *
 * Event-shape references: @opencode/plugin 2.2.5
 *   dist/promise/tool.d.ts   (execute.before / execute.after)
 *   dist/promise/session.d.ts (SessionPrompt / SessionContext / SessionCompaction)
 *   @opencode/schema/dist/session-event.d.ts (UsageUpdated)
 */

import z from "zod/v4";

import {
  createCore,
  isSyntheticMessage,
  systemHasRoutingInstructions,
  type ContextModeCore,
  type NativeToolDefinition,
  type SafeLog,
} from "./core.js";
import { extractEvents, extractUserEvents, parseOpencodeUsage, buildAgentUsageEvent } from "../../session/extract.js";
import type { HookInput } from "../../session/extract.js";
import { buildResumeSnapshot } from "../../session/snapshot.js";
import type { SessionEvent } from "../../types.js";
import { zod3ShapeToV4 } from "./zod3tov4.js";

// ── V2 context (structural subset of @opencode/plugin Context) ───────────
// Kept minimal and loosely typed on purpose: depending on `@opencode/plugin`
// would drag the Effect runtime (effect, @opencode/ai, @opencode/client —
// >100 MB of transitive deps) into context-mode's install for every other
// harness. OpenCode's V2 loader only validates `default.id` and
// `default.setup`, so a plain function suffices at runtime.

interface V2PluginContext {
  readonly location: { readonly directory: string };
  readonly tool: {
    transform(callback: (editor: V2ToolEditor) => void): Promise<unknown>;
    hook(name: "execute.before", callback: (event: V2ToolBefore) => Promise<void> | void): Promise<unknown>;
    hook(name: "execute.after", callback: (event: V2ToolAfter) => Promise<void> | void): Promise<unknown>;
  };
  readonly session: {
    hook(name: "prompt", callback: (event: V2SessionPrompt) => void): Promise<unknown>;
    hook(name: "compaction", callback: (event: V2SessionContext) => void): Promise<unknown>;
    hook(name: "context", callback: (event: V2SessionContext) => void): Promise<unknown>;
  };
  readonly event: {
    subscribe(options?: {
      types?: string[];
      signal?: AbortSignal;
    }): AsyncIterable<{ type: string; data?: Record<string, unknown> }>;
  };
}

interface V2ToolEditor {
  add(tool: {
    name: string;
    description: string;
    input: Record<string, unknown>;
    execute(input: unknown, context: V2ToolContext): Promise<unknown>;
  }): void;
}

interface V2ToolContext {
  readonly sessionID: string;
  readonly agent: string;
  readonly messageID: string;
  readonly id: string;
  readonly progress: (update: Record<string, unknown>) => Promise<void>;
  readonly signal: AbortSignal;
}

interface V2ToolBefore {
  tool: string;
  readonly sessionID: string;
  readonly id: string;
  input: unknown;
}

type V2ToolAfter = {
  readonly tool: string;
  readonly sessionID: string;
  readonly id: string;
  readonly input: unknown;
} & (
  | {
      readonly status: "completed";
      result?: {
        content?: string | ReadonlyArray<{ type?: string; text?: string }>;
        output?: unknown;
        metadata?: Record<string, unknown>;
      };
    }
  | {
      readonly status: "error";
      readonly error?: { message?: string };
    }
);

interface V2SessionPrompt {
  readonly sessionID: string;
  readonly prompt: { text?: string };
}

/** SessionContext: a V2 `SystemPart` is `{ type: "text", text: string, ... }`. */
interface V2SessionContext {
  readonly sessionID: string;
  readonly agent: string;
  system: Array<{ type?: string; text?: string }>;
  options?: Record<string, unknown>;
}

/** Adapter-internal logger (debug only, never breaks the turn). */
const silentLog: SafeLog = async () => {};

// ── Helpers ───────────────────────────────────────────────

function systemPartText(part: unknown): string {
  if (typeof part === "string") return part;
  const p = part as { text?: unknown } | null;
  return p && typeof p.text === "string" ? p.text : "";
}

/**
 * @opencode/schema Tool.Result → flat text, for the V1-shaped event capture
 * path (extractEvents reads `tool_response`).
 */
function resultToText(result: unknown): string {
  if (typeof result === "string") return result;
  if (result == null) return "";
  const r = result as {
    content?: string | ReadonlyArray<{ type?: string; text?: string }>;
    output?: unknown;
  };
  if (typeof r?.content === "string") return r.content;
  if (Array.isArray(r?.content)) {
    return r.content
      .filter((c) => c?.type === "text" && typeof c.text === "string")
      .map((c) => c.text as string)
      .join("\n");
  }
  if (r?.output !== undefined) {
    return typeof r.output === "string" ? r.output : JSON.stringify(r.output ?? "");
  }
  return "";
}

/**
 * V1 native `args` are zod-v4 shapes; V2 tool `input` is JSON Schema.
 * Convert via zod v4's own emitter, with a permissive fallback — the
 * runtime handler still does real validation through the registered zod
 * schema (core.ts buildNativeTools).
 */
function zodShapeToJsonSchema(shape: Record<string, unknown>): Record<string, unknown> {
  try {
    const v4 = zod3ShapeToV4(shape);
    const json = z.toJSONSchema(z.object(v4 as Record<string, z.ZodType>), { io: "input" });
    return json as Record<string, unknown>;
  } catch {
    // Schemas zod can't express as JSON Schema (z.coerce.*, exotic refinements)
    // degrade to an open object; validation happens in execute() via zod.
    return { type: "object", properties: {}, additionalProperties: true };
  }
}

// ── Setup ─────────────────────────────────────────────────


/**
 * V2 `setup(ctx)`. Returns the cleanup function (event subscription abort),
 * per the V2 plugin lifecycle contract.
 */
export async function openCode2Setup(ctx: V2PluginContext): Promise<() => void> {
  const projectDir = ctx?.location?.directory ?? process.cwd();
  const logger: SafeLog = async (message, extra) => {
    try {
      console.log(`[context-mode] ${typeof message === "string" ? message : ""}`, extra ?? "");
    } catch {
      // console can be dead in some host contexts — ignore
    }
  };

  const core: ContextModeCore = await createCore({
    projectDir,
    logger: process.env.OPENCODE_DEBUG ? logger : silentLog,
  });
  const { db, routing, routingBlock, autoInjection, safeLog, captureAgentsMd } = core;
  const platform = core.platform;

  // ── Native ctx_* tools (V1 `tool` map → ctx.tool.transform) ───
  await ctx.tool.transform((editor) => {
    for (const name of Object.keys(core.nativeTools)) {
      const tool: NativeToolDefinition = core.nativeTools[name];
      editor.add({
        name,
        description: tool.description,
        input: zodShapeToJsonSchema(tool.args),
        execute: async (args, toolCtx) => {
          const result = await tool.execute(
            (args ?? {}) as Record<string, unknown>,
            {
              sessionID: toolCtx.sessionID,
              messageID: toolCtx.messageID,
              agent: toolCtx.agent,
              directory: projectDir,
              metadata: (input) => void toolCtx.progress(input).catch(() => {}),
            },
          );
          return { content: typeof result === "string" ? result : result.output };
        },
      });
    }
  });

  // ── Routing enforcement (V1 tool.execute.before) ─────────
  // Deny semantics: throwing from execute.before goes through V2's only
  // failure channel ("Only tool execute.before may fail") and rejects the
  // call before it runs — same net effect as the V1 contract.
  await ctx.tool.hook("execute.before", (event) => {
    const toolInput = event.input && typeof event.input === "object" ? event.input : {};

    let decision;
    try {
      decision = routing.routePreToolUse(event.tool ?? "", toolInput, projectDir, platform);
    } catch {
      return; // Routing failure → allow passthrough
    }
    if (!decision) return; // No routing match → passthrough

    const d = decision as { action?: string; reason?: string; updatedInput?: unknown; additionalContext?: string };
    if (d.action === "deny" || d.action === "ask") {
      throw new Error(d.reason ?? "Blocked by context-mode");
    }
    if (d.action === "modify" && d.updatedInput && event.input && typeof event.input === "object") {
      Object.assign(event.input, d.updatedInput);
    }
    if (d.action === "context" && d.additionalContext && event.input && typeof event.input === "object") {
      (event.input as Record<string, unknown>).additionalContext = d.additionalContext;
    }
  });

  // ── Session event capture (V1 tool.execute.after) ────────
  await ctx.tool.hook("execute.after", (event) => {
    const sessionId = event.sessionID;
    if (!sessionId) return;
    if (event.status !== "completed") return; // failed calls are not captured
    try {
      db.ensureSession(sessionId, projectDir);
      // OC-4 (#487 follow-up): AGENTS.md → rule_content capture for snapshot
      // and auto-memory parity. Idempotent per-session via Set guard.
      captureAgentsMd(sessionId);

      const hookInput: HookInput = {
        tool_name: event.tool ?? "",
        tool_input: (event.input ?? {}) as Record<string, unknown>,
        tool_response: resultToText(event.result),
        tool_output: undefined, // V2 completed events don't carry isError
      };

      const events = extractEvents(hookInput);
      for (const ev of events) {
        db.insertEvent(sessionId, ev as SessionEvent, "PostToolUse");
      }
    } catch (e) {
      // Silent — session capture must never break the tool call
    }
  });

  // ── User-prompt capture (V1 chat.message → prompt hook) ──
  await ctx.session.hook("prompt", (event) => {
    const sessionId = event?.sessionID;
    if (!sessionId) return;
    try {
      const message = typeof event?.prompt?.text === "string" ? event.prompt.text : "";
      if (!message) return;
      // CCv2 inline filter: skip synthetic harness messages so the
      // user-prompt event stream stays clean.
      if (isSyntheticMessage(message)) return;

      db.ensureSession(sessionId, projectDir);
      // OC-4 (#487 follow-up): also capture on prompt so sessions that
      // never invoke a tool still seed rule_content events for continuity.
      captureAgentsMd(sessionId);

      // 1. Always save the raw prompt
      db.insertEvent(sessionId, {
        type: "user_prompt",
        category: "user-prompt",
        data: message,
        priority: 1,
      } as SessionEvent, "UserPromptSubmit");

      // 2. Extract role/decision/intent/skill events from the prompt body
      const userEvents = extractUserEvents(message);
      for (const ev of userEvents) {
        db.insertEvent(sessionId, ev as SessionEvent, "UserPromptSubmit");
      }
    } catch {
      // Silent — prompt capture must never break the turn
    }
  });

  // ── Resume + routing injection (V1 experimental.chat.system.transform) ──
  // V2 shape: event.system is Array<SystemPart> (`{ type: "text", text }`).
  // Keep the V1 index-1 splice reasoning: inserting at index 1 (after the
  // header) preserves the header invariant that OpenCode's cache-fold relies
  // on; unshift would invalidate the provider prompt cache on every injection.
  await ctx.session.hook("context", (event) => {
    try {
      const sessionId = event?.sessionID;
      if (!sessionId) return;

      const system = Array.isArray(event?.system) ? event.system : [];
      const texts = system.map(systemPartText);

      // OC-1 / CCv1: ROUTING_BLOCK injection (independent of the resume path).
      if (!systemHasRoutingInstructions(texts)) {
        try {
          system.splice(1, 0, { type: "text", text: routingBlock });
          texts.splice(1, 0, routingBlock);
        } catch {
          // Never break the chat turn on routing-block injection failure.
        }
        void safeLog(routingBlock, { sessionId, source: "context-hook routing block" });
      }

      // Resume snapshot claim (race-safe across concurrent processes).
      const row = db.claimLatestUnconsumedResume(sessionId);
      if (!row || !row.snapshot) return; // no row → retry on next turn

      if (Array.isArray(event?.system)) {
        event.system.splice(1, 0, { type: "text", text: row.snapshot });
        // Mark consumed only AFTER successful splice so failed paths can retry.
        void safeLog(row.snapshot, { sessionId, source: "context-hook resume" });
      }
    } catch {
      // Silent — never break the chat turn
    }
  });

  // ── Compaction snapshot (V1 experimental.session.compacting) ──
  // V1 pushed snapshot + auto-injection into the compaction request context;
  // V2 exposes the summarizer request as event.system, so the values ride
  // along there and end up inside the written summary.
  await ctx.session.hook("compaction", (event) => {
    const sessionId = event?.sessionID;
    if (!sessionId) return;
    try {
      db.ensureSession(sessionId, projectDir);
      const events = db.getEvents(sessionId);
      if (events.length === 0) return;

      const stats = db.getSessionStats(sessionId);
      const snapshot = buildResumeSnapshot(events, {
        compactCount: (stats?.compact_count ?? 0) + 1,
      });

      db.upsertResume(sessionId, snapshot, events.length);
      db.incrementCompactCount(sessionId);

      if (Array.isArray(event?.system)) {
        event.system.push({ type: "text", text: snapshot });
        void safeLog(snapshot, { sessionId, source: "compaction-hook snapshot" });

        // OC-3 / Z3: budget-capped auto-injection (≤500 tokens / ~2000 chars),
        // pushed separately so the summary folds it independently.
        try {
          const autoBlock: string = autoInjection.buildAutoInjection(events);
          if (autoBlock && autoBlock.length > 0) {
            event.system.push({ type: "text", text: autoBlock });
            void safeLog(autoBlock, { sessionId, source: "compaction-hook auto-injection" });
          }
        } catch {
          // Auto-injection failure must NOT break the snapshot path.
        }
      }
    } catch {
      return; // Silent — snapshot must never break compaction
    }
  });

  // ── Usage capture (V1 generic `event` hook → session.usage.updated) ──
  const controller = new AbortController();
  void (async () => {
    try {
      for await (const busEvent of ctx.event.subscribe({
        types: ["session.usage.updated"],
        signal: controller.signal,
      })) {
        try {
          const data = busEvent?.data as
            | {
                sessionID?: string;
                cost?: number;
                tokens?: {
                  input?: number;
                  output?: number;
                  reasoning?: number;
                  cache?: { read?: number; write?: number };
                };
              }
            | undefined;
          const sessionId = data?.sessionID;
          if (!sessionId || typeof sessionId !== "string") continue;

          // Feed the V1-shaped parser: assistant role, token columns, and the
          // native cumulative cost (exact $. The V2 event leaves modelID off;
          // the builder falls back to the pricing catalog when absent.
          const counts = parseOpencodeUsage({
            role: "assistant",
            tokens: {
              input: data.tokens?.input,
              output: data.tokens?.output,
              cache: { read: data.tokens?.cache?.read, write: data.tokens?.cache?.write },
            },
            cost: data.cost,
          });
          if (!counts) continue;
          const usageEvent = buildAgentUsageEvent(counts);
          if (!usageEvent) continue;

          db.ensureSession(sessionId, projectDir);
          db.insertEvent(sessionId, usageEvent, "UsageUpdated");
        } catch {
          // Silent — usage capture must never break the session.
        }
      }
    } catch {
      // Subscription failure is non-fatal.
    }
  })();

  return () => controller.abort();
}
