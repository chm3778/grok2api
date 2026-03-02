import type { GrokSettings, GlobalSettings } from "../settings";

type GrokNdjson = Record<string, unknown>;

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

async function readWithTimeout(
  reader: ReadableStreamDefaultReader<Uint8Array>,
  ms: number,
): Promise<ReadableStreamReadResult<Uint8Array> | { timeout: true }> {
  if (ms <= 0) return { timeout: true };
  return Promise.race([
    reader.read(),
    sleep(ms).then(() => ({ timeout: true }) as const),
  ]);
}

function makeChunk(
  id: string,
  created: number,
  model: string,
  content: string,
  finish_reason?: "stop" | "error" | null,
  reasoning_content?: string,
  includeRole = false,
): string {
  const delta: Record<string, unknown> = {};
  if (includeRole) delta.role = "assistant";
  if (content) delta.content = content;
  if (typeof reasoning_content === "string" && reasoning_content) {
    delta.reasoning_content = reasoning_content;
    delta.reasoning = reasoning_content;
  }

  const payload: Record<string, unknown> = {
    id,
    object: "chat.completion.chunk",
    created,
    model,
    choices: [
      {
        index: 0,
        delta,
        finish_reason: finish_reason ?? null,
      },
    ],
  };
  return `data: ${JSON.stringify(payload)}\n\n`;
}

function makeDone(): string {
  return "data: [DONE]\n\n";
}

function toImgProxyUrl(globalCfg: GlobalSettings, origin: string, path: string): string {
  const baseUrl = (globalCfg.base_url ?? "").trim() || origin;
  return `${baseUrl}/images/${path}`;
}

function buildVideoTag(src: string): string {
  return `<video src="${src}" controls="controls" width="500" height="300"></video>\n`;
}

function buildVideoPosterPreview(videoUrl: string, posterUrl?: string): string {
  const href = String(videoUrl || "").replace(/"/g, "&quot;");
  const poster = String(posterUrl || "").replace(/"/g, "&quot;");
  if (!href) return "";
  if (!poster) return `<a href="${href}" target="_blank" rel="noopener noreferrer">${href}</a>\n`;
  return `<a href="${href}" target="_blank" rel="noopener noreferrer" style="display:inline-block;position:relative;max-width:100%;text-decoration:none;">
  <img src="${poster}" alt="video" style="max-width:100%;height:auto;border-radius:12px;display:block;" />
  <span style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;">
    <span style="width:64px;height:64px;border-radius:9999px;background:rgba(0,0,0,.55);display:flex;align-items:center;justify-content:center;">
      <span style="width:0;height:0;border-top:12px solid transparent;border-bottom:12px solid transparent;border-left:18px solid #fff;margin-left:4px;"></span>
    </span>
  </span>
</a>\n`;
}

function buildVideoHtml(args: { videoUrl: string; posterUrl?: string; posterPreview: boolean }): string {
  if (args.posterPreview) return buildVideoPosterPreview(args.videoUrl, args.posterUrl);
  return buildVideoTag(args.videoUrl);
}

function base64UrlEncode(input: string): string {
  const bytes = new TextEncoder().encode(input);
  let binary = "";
  for (const b of bytes) binary += String.fromCharCode(b);
  return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/g, "");
}

function encodeAssetPath(raw: string): string {
  try {
    const u = new URL(raw);
    // Keep full URL (query etc.) to avoid lossy pathname-only encoding (some URLs may encode the real path in query).
    return `u_${base64UrlEncode(u.toString())}`;
  } catch {
    const p = raw.startsWith("/") ? raw : `/${raw}`;
    return `p_${base64UrlEncode(p)}`;
  }
}

function normalizeGeneratedAssetUrls(input: unknown): string[] {
  if (!Array.isArray(input)) return [];

  const out: string[] = [];
  for (const v of input) {
    if (typeof v !== "string") continue;
    const s = v.trim();
    if (!s) continue;
    if (s === "/") continue;

    try {
      const u = new URL(s);
      if (u.pathname === "/" && !u.search && !u.hash) continue;
    } catch {
      // ignore (path-style strings are allowed)
    }

    out.push(s);
  }

  return out;
}

function extractReasoningText(grok: any): string | undefined {
  const directCandidates: unknown[] = [
    grok?.reasoning_content,
    grok?.reasoningContent,
    grok?.reasoning,
    grok?.thinking,
  ];

  for (const candidate of directCandidates) {
    if (typeof candidate === "string" && candidate.trim()) return candidate;
  }

  const modelResp = grok?.modelResponse;
  const modelCandidates: unknown[] = [
    modelResp?.reasoning_content,
    modelResp?.reasoningContent,
    modelResp?.reasoning,
    modelResp?.thinking,
  ];
  for (const candidate of modelCandidates) {
    if (typeof candidate === "string" && candidate.trim()) return candidate;
  }

  return undefined;
}

function asRecord(value: unknown): Record<string, unknown> | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  return value as Record<string, unknown>;
}

function nonEmptyText(value: unknown): string | undefined {
  if (typeof value !== "string") return undefined;
  const trimmed = value.trim();
  return trimmed || undefined;
}

function tokenMatchesFilteredTag(token: string, filteredTags: string[]): boolean {
  if (!token) return false;
  return filteredTags.some((tag) => tag && token.includes(tag));
}

function splitGluedReasoningMarkers(value: string): string {
  if (!value) return value;
  return value.replace(
    /([^\n])[ \t]*(?=\[(?:Agent\s*\d+|Grok)\]\s+\[(?:WebSearch|AgentThink|SearchImage)\])/g,
    "$1\n\n",
  );
}

function startsWithReasoningMarker(value: string): boolean {
  return /^\s*\[(?:Agent\s*\d+|Grok)\]\s+\[(?:WebSearch|AgentThink|SearchImage)\]/.test(value);
}

function extractToolUsageReasoning(grok: any): string | undefined {
  const card = asRecord(grok?.toolUsageCard);
  if (!card) return undefined;

  const rolloutId = nonEmptyText(grok?.rolloutId);
  const prefix = rolloutId ? `[${rolloutId}] ` : "";

  const webSearchArgs = asRecord(asRecord(card.webSearch)?.args);
  const webSearchQuery = nonEmptyText(webSearchArgs?.query) ?? nonEmptyText(webSearchArgs?.q);
  if (webSearchQuery) return `${prefix}[WebSearch] ${webSearchQuery}`;

  const chatroomSendArgs = asRecord(asRecord(card.chatroomSend)?.args);
  const chatroomMessage = nonEmptyText(chatroomSendArgs?.message);
  if (chatroomMessage) return `${prefix}[AgentThink] ${chatroomMessage}`;

  const searchImageArgs = asRecord(asRecord(card.searchImages)?.args);
  const searchImageQuery =
    nonEmptyText(searchImageArgs?.image_description) ??
    nonEmptyText(searchImageArgs?.description) ??
    nonEmptyText(searchImageArgs?.query);
  if (searchImageQuery) return `${prefix}[SearchImage] ${searchImageQuery}`;

  return undefined;
}

export function createOpenAiStreamFromGrokNdjson(
  grokResp: Response,
  opts: {
    cookie: string;
    settings: GrokSettings;
    global: GlobalSettings;
    origin: string;
    requestedModel: string;
    onFinish?: (result: { status: number; duration: number }) => Promise<void> | void;
  },
): ReadableStream<Uint8Array> {
  const { settings, global, origin } = opts;
  const fallbackModel =
    typeof opts.requestedModel === "string" && opts.requestedModel.trim()
      ? opts.requestedModel.trim()
      : "grok-4";
  const decoder = new TextDecoder();
  const encoder = new TextEncoder();

  const id = `chatcmpl-${crypto.randomUUID()}`;
  const created = Math.floor(Date.now() / 1000);

  const filteredTags = (settings.filtered_tags ?? "")
    .split(",")
    .map((t) => t.trim())
    .filter(Boolean);
  const showThinking = settings.show_thinking !== false;

  const firstTimeoutMs = Math.max(0, (settings.stream_first_response_timeout ?? 30) * 1000);
  const chunkTimeoutMs = Math.max(0, (settings.stream_chunk_timeout ?? 120) * 1000);
  const totalTimeoutMs = Math.max(0, (settings.stream_total_timeout ?? 600) * 1000);

  return new ReadableStream<Uint8Array>({
    async start(controller) {
      const body = grokResp.body;
      if (!body) {
        controller.enqueue(encoder.encode(makeChunk(id, created, fallbackModel, "Empty response", "error")));
        controller.enqueue(encoder.encode(makeDone()));
        controller.close();
        return;
      }

      const reader = body.getReader();
      const startTime = Date.now();
      let finalStatus = 200;
      let lastChunkTime = startTime;
      let firstReceived = false;

      let currentModel = fallbackModel;
      let isImage = false;
      let isThinking = false;
      let thinkingFinished = false;
      let lastVideoProgress = -1;
      let roleSent = false;
      let hasEmittedReasoning = false;

      let buffer = "";

      const emitChunk = (
        content: string,
        finishReason?: "stop" | "error" | null,
        reasoningContent?: string,
      ) => {
        const includeRole = !roleSent && (Boolean(content) || Boolean(reasoningContent));
        controller.enqueue(
          encoder.encode(makeChunk(id, created, currentModel, content, finishReason, reasoningContent, includeRole)),
        );
        if (includeRole) roleSent = true;
      };

      const flushStop = () => {
        emitChunk("", "stop");
        controller.enqueue(encoder.encode(makeDone()));
      };

      try {
        // eslint-disable-next-line no-constant-condition
        while (true) {
          const now = Date.now();
          const elapsed = now - startTime;
          if (!firstReceived && elapsed > firstTimeoutMs) {
            flushStop();
            if (opts.onFinish) await opts.onFinish({ status: finalStatus, duration: (Date.now() - startTime) / 1000 });
            controller.close();
            return;
          }
          if (totalTimeoutMs > 0 && elapsed > totalTimeoutMs) {
            flushStop();
            if (opts.onFinish) await opts.onFinish({ status: finalStatus, duration: (Date.now() - startTime) / 1000 });
            controller.close();
            return;
          }
          const idle = now - lastChunkTime;
          if (firstReceived && idle > chunkTimeoutMs) {
            flushStop();
            if (opts.onFinish) await opts.onFinish({ status: finalStatus, duration: (Date.now() - startTime) / 1000 });
            controller.close();
            return;
          }

          const perReadTimeout = Math.min(
            firstReceived ? chunkTimeoutMs : firstTimeoutMs,
            totalTimeoutMs > 0 ? Math.max(0, totalTimeoutMs - elapsed) : Number.POSITIVE_INFINITY,
          );

          const res = await readWithTimeout(reader, perReadTimeout);
          if ("timeout" in res) {
            flushStop();
            if (opts.onFinish) await opts.onFinish({ status: finalStatus, duration: (Date.now() - startTime) / 1000 });
            controller.close();
            return;
          }

          const { value, done } = res;
          if (done) break;
          if (!value) continue;
          buffer += decoder.decode(value, { stream: true });

          for (let idx = buffer.indexOf("\n"); idx !== -1; idx = buffer.indexOf("\n")) {
            const line = buffer.slice(0, idx).trim();
            buffer = buffer.slice(idx + 1);
            if (!line) continue;

            let data: GrokNdjson;
            try {
              data = JSON.parse(line) as GrokNdjson;
            } catch {
              continue;
            }

            firstReceived = true;
            lastChunkTime = Date.now();

            const err = (data as any).error;
            if (err?.message) {
              finalStatus = 500;
              emitChunk(`Error: ${String(err.message)}`, "stop");
              controller.enqueue(encoder.encode(makeDone()));
              if (opts.onFinish) await opts.onFinish({ status: finalStatus, duration: (Date.now() - startTime) / 1000 });
              controller.close();
              return;
            }

            const grok = (data as any).result?.response;
            if (!grok) continue;

            const userRespModel = grok.userResponse?.model;
            if (typeof userRespModel === "string" && userRespModel.trim()) currentModel = userRespModel.trim();

            // Video generation stream
            const videoResp = grok.streamingVideoGenerationResponse;
            if (videoResp) {
              const progress = typeof videoResp.progress === "number" ? videoResp.progress : 0;
              const videoUrl = typeof videoResp.videoUrl === "string" ? videoResp.videoUrl : "";
              const thumbUrl = typeof videoResp.thumbnailImageUrl === "string" ? videoResp.thumbnailImageUrl : "";

              if (progress > lastVideoProgress) {
                lastVideoProgress = progress;
                if (showThinking) {
                  const msg = `视频已生成${progress}%`;
                  emitChunk("", undefined, msg);
                }
              }

              if (videoUrl) {
                const videoPath = encodeAssetPath(videoUrl);
                const src = toImgProxyUrl(global, origin, videoPath);

                let poster: string | undefined;
                if (thumbUrl) {
                  const thumbPath = encodeAssetPath(thumbUrl);
                  poster = toImgProxyUrl(global, origin, thumbPath);
                }

                emitChunk(
                  buildVideoHtml({
                    videoUrl: src,
                    posterPreview: settings.video_poster_preview === true,
                    ...(poster ? { posterUrl: poster } : {}),
                  }),
                );
              }
              continue;
            }

            if (grok.imageAttachmentInfo) isImage = true;
            const rawToken = grok.token;
            const explicitReasoningText = extractReasoningText(grok);
            const toolUsageReasoningText = extractToolUsageReasoning(grok);
            const currentIsThinking = Boolean(grok.isThinking);
            const messageTag = grok.messageTag;

            if (isImage) {
              const modelResp = grok.modelResponse;
              if (modelResp) {
                const urls = normalizeGeneratedAssetUrls(modelResp.generatedImageUrls);
                if (urls.length) {
                  const linesOut: string[] = [];
                  for (const u of urls) {
                    const imgPath = encodeAssetPath(u);
                    const imgUrl = toImgProxyUrl(global, origin, imgPath);
                    linesOut.push(`![Generated Image](${imgUrl})`);
                  }
                  emitChunk(linesOut.join("\n"), "stop");
                  controller.enqueue(encoder.encode(makeDone()));
                  if (opts.onFinish) await opts.onFinish({ status: finalStatus, duration: (Date.now() - startTime) / 1000 });
                  controller.close();
                  return;
                }
              } else if (typeof rawToken === "string" && rawToken) {
                emitChunk(rawToken);
              }
              continue;
            }

            // Text chat stream
            const hasTextToken = typeof rawToken === "string" && rawToken.length > 0;
            if (Array.isArray(rawToken) && !explicitReasoningText && !toolUsageReasoningText) continue;
            if (!hasTextToken && !explicitReasoningText && !toolUsageReasoningText) continue;
            let token = hasTextToken ? rawToken : "";

            const tokenFiltered = tokenMatchesFilteredTag(token, filteredTags);
            if (tokenFiltered) {
              if (!currentIsThinking) continue;
              token = "";
            }

            if (thinkingFinished && currentIsThinking) continue;

            if (grok.toolUsageCardId && grok.webSearchResults?.results && Array.isArray(grok.webSearchResults.results)) {
              if (currentIsThinking) {
                if (showThinking) {
                  let appended = "";
                  for (const r of grok.webSearchResults.results) {
                    const title = typeof r.title === "string" ? r.title : "";
                    const url = typeof r.url === "string" ? r.url : "";
                    const preview = typeof r.preview === "string" ? r.preview.replace(/\n/g, "") : "";
                    appended += `\n- [${title}](${url} \"${preview}\")`;
                  }
                  token += `${appended}\n`;
                } else {
                  continue;
                }
              } else {
                continue;
              }
            }

            let content = currentIsThinking ? "" : token;
            if (!currentIsThinking && messageTag === "header") content = `\n\n${token}\n\n`;

            let shouldSkip = false;
            if (isThinking && !currentIsThinking) {
              thinkingFinished = true;
            } else if (currentIsThinking && !showThinking) {
              shouldSkip = true;
            }

            let reasoningContent = showThinking
              ? explicitReasoningText ?? toolUsageReasoningText ?? (currentIsThinking ? token : undefined)
              : undefined;
            if (typeof reasoningContent === "string" && reasoningContent) {
              reasoningContent = splitGluedReasoningMarkers(reasoningContent);
              if (
                hasEmittedReasoning &&
                startsWithReasoningMarker(reasoningContent) &&
                !reasoningContent.startsWith("\n")
              ) {
                reasoningContent = `\n\n${reasoningContent}`;
              }
            }
            if (!content && !reasoningContent) shouldSkip = true;
            if (!shouldSkip) {
              emitChunk(content, undefined, reasoningContent);
              if (reasoningContent) hasEmittedReasoning = true;
            }
            isThinking = currentIsThinking;
          }
        }

        emitChunk("", "stop");
        controller.enqueue(encoder.encode(makeDone()));
        if (opts.onFinish) await opts.onFinish({ status: finalStatus, duration: (Date.now() - startTime) / 1000 });
        controller.close();
      } catch (e) {
        finalStatus = 500;
        emitChunk(`处理错误: ${e instanceof Error ? e.message : String(e)}`, "error");
        controller.enqueue(encoder.encode(makeDone()));
        if (opts.onFinish) await opts.onFinish({ status: finalStatus, duration: (Date.now() - startTime) / 1000 });
        controller.close();
      } finally {
        try {
          reader.releaseLock();
        } catch {
          // ignore
        }
      }
    },
  });
}

export async function parseOpenAiFromGrokNdjson(
  grokResp: Response,
  opts: { cookie: string; settings: GrokSettings; global: GlobalSettings; origin: string; requestedModel: string },
): Promise<Record<string, unknown>> {
  const { global, origin, requestedModel, settings } = opts;
  const showThinking = settings.show_thinking !== false;
  const filteredTags = (settings.filtered_tags ?? "")
    .split(",")
    .map((t) => t.trim())
    .filter(Boolean);
  const text = await grokResp.text();
  const lines = text.split("\n").map((l) => l.trim()).filter(Boolean);

  let content = "";
  let model = requestedModel;
  const reasoningParts: string[] = [];
  for (const line of lines) {
    let data: GrokNdjson;
    try {
      data = JSON.parse(line) as GrokNdjson;
    } catch {
      continue;
    }

    const err = (data as any).error;
    if (err?.message) throw new Error(String(err.message));

    const grok = (data as any).result?.response;
    if (!grok) continue;

    const currentIsThinking = Boolean(grok.isThinking);

    const explicitReasoningText = extractReasoningText(grok);
    const toolUsageReasoningText = extractToolUsageReasoning(grok);
    const rawToken = typeof grok.token === "string" ? grok.token : "";
    const tokenReasoningText = tokenMatchesFilteredTag(rawToken, filteredTags) ? "" : rawToken;
    const frameReasoning = showThinking
      ? explicitReasoningText ?? toolUsageReasoningText ?? (currentIsThinking ? tokenReasoningText : undefined)
      : undefined;
    if (frameReasoning) reasoningParts.push(frameReasoning);

    const videoResp = grok.streamingVideoGenerationResponse;
    if (videoResp?.videoUrl && typeof videoResp.videoUrl === "string") {
      const videoPath = encodeAssetPath(videoResp.videoUrl);
      const src = toImgProxyUrl(global, origin, videoPath);

      let poster: string | undefined;
      if (typeof videoResp.thumbnailImageUrl === "string" && videoResp.thumbnailImageUrl) {
        const thumbPath = encodeAssetPath(videoResp.thumbnailImageUrl);
        poster = toImgProxyUrl(global, origin, thumbPath);
      }

      content = buildVideoHtml({
        videoUrl: src,
        posterPreview: settings.video_poster_preview === true,
        ...(poster ? { posterUrl: poster } : {}),
      });
      model = requestedModel;
      break;
    }

    const modelResp = grok.modelResponse;
    if (!modelResp) continue;
    if (typeof modelResp.error === "string" && modelResp.error) throw new Error(modelResp.error);

    if (typeof modelResp.model === "string" && modelResp.model) model = modelResp.model;
    if (typeof modelResp.message === "string") content = modelResp.message;

    const rawUrls = modelResp.generatedImageUrls;
    const urls = normalizeGeneratedAssetUrls(rawUrls);
    if (urls.length) {
      for (const u of urls) {
        const imgPath = encodeAssetPath(u);
        const imgUrl = toImgProxyUrl(global, origin, imgPath);
        content += `\n![Generated Image](${imgUrl})`;
      }
      break;
    }

    // If upstream emits placeholder/empty generatedImageUrls in intermediate frames, keep scanning.
    if (Array.isArray(rawUrls)) continue;

    // For normal chat replies, the first modelResponse is enough.
    break;
  }

  const reasoningText = splitGluedReasoningMarkers(reasoningParts.join("").trim());
  const message: Record<string, unknown> = { role: "assistant", content };
  if (showThinking && reasoningText) {
    message.reasoning_content = reasoningText;
    message.reasoning = reasoningText;
  }

  return {
    id: `chatcmpl-${crypto.randomUUID()}`,
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model,
    choices: [
      {
        index: 0,
        message,
        finish_reason: "stop",
      },
    ],
    usage: null,
  };
}
