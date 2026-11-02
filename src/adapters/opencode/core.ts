/**
 * OpenCode adapter core — shared bootstrap between the V1 and V2 plugin
 * entrypoints.
 *
 * OpenCode 2 intentionally broke the plugin API (docs/migrate-v1): plugins now
 * export `Plugin.define({ id, setup })` (or an Effect) instead of the V1
 * hooks-factory contract. context-mode v1.0.x keeps its existing V1
 * implementation (`plugin.ts`); this module holds the state bootstrap both
 * implementations share, plus the V2 setup factory (`v2.ts`).
 *
 * Moved here verbatim from plugin.ts ( Tyler / PR add-v2 follow-up):
 *   - shared request/context types (PluginContext, NativeTool*)
 *   - platform detection (getPlatform)
 *   - synthetic-message filter
 *   - routing instructions quorum (ROUTING_MARKERS / systemHasRoutingInstructions)
 *   - native tool registry bridge (buildNativeTools)
 *   - plugin init sequence (createCore)
 */

import { dirname, resolve, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { existsSync, readFileSync } from "node:fs";

import { resolveSessionDbPath, SessionDB } from "../../session/db.js";
import type { SessionEvent } from "../../types.js";
import { AdapterPlatformType, OpenCodeAdapter } from "./index.js";
import { PLATFORM_ENV_VARS } from "../detect.js";
import { zod3ShapeToV4 } from "./zod3tov4.js";

// ── Shared types ──────────────────────────────────────────

/** KiloCode/OpenCode plugin input — both platforms pass at least `directory`. */
export type PluginClientAppLogBodyExtra = {
  sessionId?: string;
  source?: string;
};

export type PluginClientAppLogBody = {
  service: string;
  level: "info" | "warn" | "error" | "debug"; // Strict union for log levels
  message: string;
  extra?: PluginClientAppLogBodyExtra;
};

export type PluginClientAppLogOptions = {
  body: PluginClientAppLogBody;
};

export type PluginClientApp = {
  log: (options: PluginClientAppLogOptions) => Promise<void>;
};

export type PluginClient = {
  app: PluginClientApp;
};

export type PluginContext = {
  client: PluginClient;
  directory: string;
};

export type NativeToolContext = {
  sessionID: string;
  messageID: string;
  agent: string;
  directory: string;
  worktree?: string;
  abort?: AbortSignal;
  metadata?: (input: { title?: string; metadata?: Record<string, unknown> }) => void;
};

export type NativeToolDefinition = {
  description: string;
  args: Record<string, unknown>;
  execute: (
    args: Record<string, unknown>,
    ctx: NativeToolContext,
  ) => Promise<string | { title?: string; output: string; metadata?: Record<string, unknown> }>;
};

/** Best-effort debug log — never rejects (#448). No-op without a logger. */
export type SafeLog = (message?: string, extra?: PluginClientAppLogBodyExtra) => Promise<void>;

// ── Synthetic message filter ──────────────────────────────

// Synthetic message tags emitted by harnesses (CCv2 inline filter). When the
// user "message" is actually a system-generated nudge (e.g. tool-result, system
// reminder), capturing it as user_prompt would flood the DB with noise.
const SYNTHETIC_MESSAGE_PREFIXES = [
  "<task-notification>",
  "<system-reminder>",
  "<context_guidance>",
  "<tool-result>",
];

export function isSyntheticMessage(text: string): boolean {
  const trimmed = text.trim();
  return SYNTHETIC_MESSAGE_PREFIXES.some((p) => trimmed.startsWith(p));
}

// ── Routing instructions quorum ───────────────────────────

// Quorum markers — must NOT be substrings of each other (#487).
// Each token uniquely identifies the routing block / context-mode rules
// without overlapping any other marker. The XML tag is the primary signal;
// the two distinctive bare tool names are the secondary signals. Together
// any 2 of 3 confirm the system prompt already carries routing instructions.
export const ROUTING_MARKERS = [
  "<context_window_protection>",
  "ctx_search",
  "ctx_index",
];

export function systemHasRoutingInstructions(system: string[]): boolean {
  const text = system.join("\n");
  // Word-boundary check guards against unrelated identifiers that happen to
  // share a prefix/suffix (e.g. a hypothetical `ctx_search_v2`).
  const wordBoundary = (m: string) => {
    if (m.startsWith("<")) return text.includes(m);
    const re = new RegExp(`(?:^|\\W)${m.replace(/[.*+?^${}()|[\\]\\\\]/g, "\\$&")}(?:\\W|$)`);
    return re.test(text);
  };
  return ROUTING_MARKERS.filter(wordBoundary).length >= 2;
}

/**
 * Detect whether the plugin is running under KiloCode or OpenCode.
 *
 * Reuses the canonical PLATFORM_ENV_VARS list (src/adapters/detect.ts) instead
 * of hardcoding env var names — single source of truth, future-proof if Kilo
 * or OpenCode add/rename env vars upstream.
 *
 * Order matters: KiloCode is an OpenCode fork and sets `OPENCODE=1` in
 * addition to `KILO_PID`. PLATFORM_ENV_VARS lists `kilo` BEFORE `opencode`
 * so KILO_PID wins the iteration.
 *
 * Pre-fix version was `return process.env.KILO_PID ? "kilo" : "opencode";` —
 */
export function getPlatform(): AdapterPlatformType {
  for (const [platform, vars] of PLATFORM_ENV_VARS) {
    if (platform !== "kilo" && platform !== "opencode") continue;
    if (vars.some((v) => process.env[v.name])) {
      return platform as AdapterPlatformType;
    }
  }
  // Plugin host should always set one of the env vars. Fallback to opencode
  // (the wider ecosystem) when neither is set, for predictable behavior.
  return "opencode";
}

// ── Shared plugin state ───────────────────────────────────

export interface RoutingDecision {
  action?: string;
  reason?: string;
  updatedInput?: unknown;
  additionalContext?: string;
}

export type RoutingModuleLike = {
  initSecurity(buildRoot: string): Promise<unknown>;
  routePreToolUse(
    toolName: string,
    toolInput: unknown,
    projectDir: string,
    platform: AdapterPlatformType,
  ): false | undefined | RoutingDecision;
};

export type AutoInjectionModuleLike = {
  buildAutoInjection(events: SessionEvent[]): string;
};

export interface ContextModeCore {
  platform: AdapterPlatformType;
  projectDir: string;
  db: SessionDB;
  routing: RoutingModuleLike;
  /** Pre-built platform-specific routing block (cached per process). */
  routingBlock: string;
  autoInjection: AutoInjectionModuleLike;
  /** Native in-process ctx_* tools bridged from the MCP server registry. */
  nativeTools: Record<string, NativeToolDefinition>;
  /** OC-4: persist AGENTS.md/CLAUDE.md/CONTEXT.md as rule events, once per session. */
  captureAgentsMd(sessionId: string): void;
  safeLog: SafeLog;
}

/**
 * Initialize everything the V1 hooks factory and the V2 setup function both
 * need. Moved verbatim from the V1 factory prelude so both entrypoints share
 * one init path (paths, env, config changes must not drift).
 */
export async function createCore(opts: {
  projectDir: string;
  /** Host logger; omit for a silent no-op (debug logging is best-effort). */
  logger?: (message?: string, extra?: PluginClientAppLogBodyExtra) => Promise<void>;
}): Promise<ContextModeCore> {
  const projectDir = opts.projectDir;
  const platform = getPlatform();
  const adapter = new OpenCodeAdapter(platform);
  // initSecurity() looks for `<dir>/security.js`, which lives at the
  // top of build/ — two levels up from this adapter directory.
  const selfDir = dirname(fileURLToPath(import.meta.url));
  const buildRoot = resolve(selfDir, "..", "..");

  // Load routing module (ESM .mjs, lives outside build/ in hooks/)
  const routingPath = resolve(selfDir, "..", "..", "..", "hooks", "core", "routing.mjs");
  const routing = (await import(pathToFileURL(routingPath).href)) as unknown as RoutingModuleLike;
  await routing.initSecurity(buildRoot);

  // OC-1 / OC-3: Load hook helpers once at plugin init. Dynamic import keeps
  // the .mjs ESM islands isolated from the .ts compile graph.
  const routingBlockPath = resolve(selfDir, "..", "..", "..", "hooks", "routing-block.mjs");
  const routingBlockMod = await import(pathToFileURL(routingBlockPath).href);
  const toolNamingPath = resolve(selfDir, "..", "..", "..", "hooks", "core", "tool-naming.mjs");
  const toolNamingMod = await import(pathToFileURL(toolNamingPath).href);
  const autoInjectionPath = resolve(selfDir, "..", "..", "..", "hooks", "auto-injection.mjs");
  const autoInjectionMod = (await import(pathToFileURL(autoInjectionPath).href)) as unknown as AutoInjectionModuleLike;

  // Pre-build the routing block once per process — it is platform-specific
  // (tool naming differs between opencode and kilo) but does NOT depend on
  // sessionID, so we cache it. createToolNamer accepts both "opencode" and
  // "kilo" per hooks/core/tool-naming.mjs:25-26.
  const toolNamer = toolNamingMod.createToolNamer(platform);
  const routingBlock: string = routingBlockMod.createRoutingBlock(toolNamer);

  // Initialize per-process state. We do NOT fabricate a sessionId here —
  // OpenCode/Kilo provide the real `input.sessionID` on every hook, and a
  // process-global UUID would (a) never match prior-session resume rows and
  // (b) collide across multi-session reuse (Mickey / PR #376 root cause).
  // C2 narrowing: resolve DB path through the canonical helper directly.
  // BaseAdapter no longer exposes getSessionDBPath; the adapter only owns
  // the sessions DIR (per-platform), the helper owns the per-project FILE
  // (case-fold + worktree-suffix + one-shot legacy migration).
  const db = new SessionDB({
    dbPath: resolveSessionDbPath({ projectDir, sessionsDir: adapter.getSessionDir() }),
  });

  // Clean up old sessions on startup (no SessionStart hook to do this).
  db.cleanupOldSessions(7);

  // OC-4 (#487 follow-up): per-session capture gate. PR #487 trusted the host
  // to deliver AGENTS.md events, but OpenCode only fires `rule_content` events
  // when the user explicitly reads the file. snapshot.ts:172 + analytics.ts:152
  // CONSUME `rule_content` to render rules into the resume snapshot — without
  // this capture path, AGENTS.md is silently absent from continuity output.
  // Keyed by sessionId (NOT projectDir) so multi-session reuse within a long-
  // lived plugin process still gets per-session capture exactly once.
  const agentsMdCaptured = new Set<string>();

  /**
   * OC-4: Read AGENTS.md (with CLAUDE.md / CONTEXT.md fallbacks) from the
   * project directory and persist as `rule` + `rule_content` events. Mirrors
   * the CC SessionStart pattern at hooks/sessionstart.mjs:121-132 and the
   * OpenCode instruction.ts FILES order. Idempotent via `agentsMdCaptured`
   * Set keyed by sessionId. Fail-soft: missing/unreadable files do not throw.
   */
  function captureAgentsMd(sessionId: string): void {
    if (agentsMdCaptured.has(sessionId)) return;
    agentsMdCaptured.add(sessionId);
    const candidates = ["AGENTS.md", "CLAUDE.md", "CONTEXT.md"];
    for (const name of candidates) {
      try {
        const p = join(projectDir, name);
        if (!existsSync(p)) continue;
        const content = readFileSync(p, "utf-8");
        if (!content.trim()) continue;
        db.insertEvent(sessionId, {
          type: "rule",
          category: "rule",
          data: p,
          priority: 1,
        } as SessionEvent, "PluginInit");
        db.insertEvent(sessionId, {
          type: "rule_content",
          category: "rule",
          data: content,
          priority: 1,
        } as SessionEvent, "PluginInit");
      } catch {
        // file missing or unreadable — skip silently
      }
    }
  }

  // Drop-in wrapper for debug logging that NEVER rejects (#448).
  const safeLog: SafeLog = async (message?: string, extra?: PluginClientAppLogBodyExtra) => {
    if (!opts.logger) return;
    try {
      await opts.logger(message, extra);
    } catch {
      // Never break the turn on debug-log failure.
    }
  };

  const nativeTools = await buildNativeTools(projectDir);

  return {
    platform,
    projectDir,
    db,
    routing,
    routingBlock,
    autoInjection: autoInjectionMod,
    nativeTools,
    captureAgentsMd,
    safeLog,
  };
}

/**
 * In-process bridge from the MCP server registry to the OpenCode native tool
 * layer (#574): ctx_* tools run in the plugin host process instead of
 * spawning a separate MCP child per session.
 */
export async function buildNativeTools(projectDir: string): Promise<Record<string, NativeToolDefinition>> {
  // Import the existing MCP server registry without starting its stdio
  // transport. This is the plugin-only bridge for #574: OpenCode/Kilo
  // call ctx_* tools in-process through Hooks.tool instead of spawning
  // a separate MCP child per session.
  const prevEmbedded = process.env.CONTEXT_MODE_EMBEDDED_PLUGIN_TOOLS;
  process.env.CONTEXT_MODE_EMBEDDED_PLUGIN_TOOLS = "1";
  let mod: typeof import("../../server.js");
  try {
    mod = await import("../../server.js");
  } finally {
    if (prevEmbedded === undefined) delete process.env.CONTEXT_MODE_EMBEDDED_PLUGIN_TOOLS;
    else process.env.CONTEXT_MODE_EMBEDDED_PLUGIN_TOOLS = prevEmbedded;
  }
  const tools: Record<string, NativeToolDefinition> = {};

  for (const registered of mod.REGISTERED_CTX_TOOLS) {
    const config = registered.config as Record<string, unknown>;
    // Zod schema object that the MCP framework normally calls
    // safeParseAsync() on before invoking the handler. The native
    // OpenCode plugin path bypasses MCP's transport layer entirely
    // (refs/platforms/opencode/packages/opencode/src/tool/registry.ts:127),
    // so we must parse args here too — otherwise z.preprocess() coercions
    // (coerceCommandsArray / coerceJsonArray in server.ts) and defaults
    // never fire. Fixes #621.
    const inputSchema = config.inputSchema as
      | { shape?: unknown; _def?: { shape?: unknown }; parse?: (input: unknown) => unknown }
      | undefined;
    const shape =
      typeof inputSchema?.shape === "object" && inputSchema.shape !== null
        ? inputSchema.shape
        : typeof inputSchema?._def?.shape === "function"
          ? (inputSchema._def.shape as () => unknown)()
          : {};

    // Both KiloCode and recent OpenCode bundle Zod v4 in-host; v3 schemas
    // crash with `n._zod.def` undefined. Gate widened from kilo-only (#632)
    // because every consumer of this file is an OpenCode-family host.
    const argsForHost = zod3ShapeToV4(shape as Record<string, unknown>);

    tools[registered.name] = {
      description: String(config.description ?? ""),
      args: argsForHost,
      async execute(args: Record<string, unknown>, toolCtx: NativeToolContext) {
        toolCtx.metadata?.({ title: String(config.title ?? registered.name) });
        const project = toolCtx.directory || projectDir;

        // Run the registered Zod schema BEFORE the handler — same contract
        // as the MCP SDK (server/mcp.js safeParseAsync at line 174). This
        // applies z.preprocess() coercions, populates .default() values,
        // and produces the validation error the handler expects (#621).
        let parsedArgs: Record<string, unknown> = args ?? {};
        if (typeof inputSchema?.parse === "function") {
          try {
            parsedArgs = inputSchema.parse(args ?? {}) as Record<string, unknown>;
          } catch (err) {
            // Surface validation failures with a clear, actionable message
            // (mirrors MCP SDK error format) instead of a downstream
            // "x.map is not a function" crash.
            const message = err instanceof Error ? err.message : String(err);
            throw new Error(
              `Invalid arguments for ${registered.name}: ${message}`,
            );
          }
        }

        const result = await mod.withProjectDirOverride({ projectDir: project, sessionId: toolCtx.sessionID }, async () =>
          registered.handler(parsedArgs),
        );

        const r = result as {
          content?: Array<{ type?: string; text?: string }>;
          isError?: boolean;
        };
        const text = Array.isArray(r?.content)
          ? r.content
              .filter((c) => c?.type === "text" && typeof c.text === "string")
              .map((c) => c.text)
              .join("\n")
          : typeof result === "string"
            ? result
            : JSON.stringify(result ?? "");

        if (r?.isError) throw new Error(text || `${registered.name} returned an error`);
        return { title: String(config.title ?? registered.name), output: text };
      },
    };
  }

  return tools;
}
