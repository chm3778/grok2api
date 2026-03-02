import test from 'node:test';
import assert from 'node:assert/strict';

import { buildConversationPayload } from '../src/grok/conversation.ts';
import { createOpenAiStreamFromGrokNdjson, parseOpenAiFromGrokNdjson } from '../src/grok/processor.ts';

test('show_thinking=false disables isReasoning even for thinking model', () => {
  const { payload } = buildConversationPayload({
    requestModel: 'grok-4-thinking',
    content: 'hello',
    imgIds: [],
    imgUris: [],
    settings: {
      temporary: true,
      show_thinking: false,
    },
  });

  assert.equal(payload.isReasoning, false);
});

test('show_thinking=true enables isReasoning even for non-thinking model', () => {
  const { payload } = buildConversationPayload({
    requestModel: 'grok-4.20-beta',
    content: 'hello',
    imgIds: [],
    imgUris: [],
    settings: {
      temporary: true,
      show_thinking: true,
    },
  });

  assert.equal(payload.isReasoning, true);
});

test('reasoning effort override is not present in upstream payload', () => {
  const { payload } = buildConversationPayload({
    requestModel: 'grok-4-thinking',
    content: 'hello',
    imgIds: [],
    imgUris: [],
    settings: {
      temporary: true,
      show_thinking: true,
    },
  });

  const responseMetadata = payload.responseMetadata;
  assert.ok(responseMetadata && typeof responseMetadata === 'object');
  assert.equal('modelConfigOverride' in responseMetadata, false);
});

test('non-stream parser skips reasoning fields when no extractable reasoning text', async () => {
  const lines = [
    {
      result: {
        response: {
          token: '<xai:tool_usage_card><xai:tool_name>web_search</xai:tool_name></xai:tool_usage_card>',
          isThinking: true,
        },
      },
    },
    {
      result: {
        response: {
          modelResponse: {
            responseId: 'resp-1',
            model: 'grok-420',
            message: 'final',
          },
          isThinking: false,
        },
      },
    },
  ];

  const parsed = await parseOpenAiFromGrokNdjson(new Response(lines.map((x) => JSON.stringify(x)).join('\n')), {
    cookie: '',
    settings: {
      filtered_tags: 'xaiartifact,xai:tool_usage_card',
      show_thinking: true,
      video_poster_preview: false,
    },
    global: {
      base_url: '',
      log_level: 'INFO',
    },
    origin: 'http://127.0.0.1',
    requestedModel: 'grok-4.20-beta',
  });

  const message = parsed.choices?.[0]?.message;
  assert.ok(message && typeof message === 'object');
  assert.equal('reasoning_content' in message, false);
  assert.equal('reasoning' in message, false);
});

test('stream format emits role once and avoids think-tag wrappers when reasoning fields are present', async () => {
  const lines = [
    {
      result: {
        response: {
          token:
            '<xai:tool_usage_card><xai:tool_name>web_search</xai:tool_name><xai:tool_args><![CDATA[{"query":"昨天的当天是明天的什么"}]]></xai:tool_args></xai:tool_usage_card>',
          isThinking: true,
          rolloutId: 'Agent 2',
          toolUsageCard: {
            webSearch: { args: { query: '昨天的当天是明天的什么' } },
          },
        },
      },
    },
    {
      result: {
        response: {
          token: '前天。',
          isThinking: false,
          messageTag: 'final',
        },
      },
    },
  ];

  const stream = createOpenAiStreamFromGrokNdjson(new Response(lines.map((x) => JSON.stringify(x)).join('\n')), {
    cookie: '',
    settings: {
      filtered_tags: 'xaiartifact,xai:tool_usage_card',
      show_thinking: true,
      video_poster_preview: false,
    },
    global: {
      base_url: '',
      log_level: 'INFO',
    },
    origin: 'http://127.0.0.1',
    requestedModel: 'grok-4.20-beta',
  });

  const sseText = await new Response(stream).text();

  assert.equal(sseText.includes('"reasoning_content"'), true);
  assert.equal(sseText.includes('<think>'), false);
  assert.equal(sseText.includes('</think>'), false);

  const roleMatches = sseText.match(/"role":"assistant"/g) ?? [];
  assert.equal(roleMatches.length, 1);
});

test('non-stream parser splits glued agent reasoning markers with blank lines', async () => {
  const gluedReasoning =
    '[Agent 3] [WebSearch] Jio SIM cost[Agent 3] [WebSearch] Jio eSIM activation[Agent 1] [AgentThink] summarize findings';
  const lines = [
    {
      result: {
        response: {
          reasoning_content: gluedReasoning,
          isThinking: true,
        },
      },
    },
    {
      result: {
        response: {
          modelResponse: {
            responseId: 'resp-2',
            model: 'grok-420',
            message: 'final',
          },
          isThinking: false,
        },
      },
    },
  ];

  const parsed = await parseOpenAiFromGrokNdjson(new Response(lines.map((x) => JSON.stringify(x)).join('\n')), {
    cookie: '',
    settings: {
      filtered_tags: 'xaiartifact,xai:tool_usage_card',
      show_thinking: true,
      video_poster_preview: false,
    },
    global: {
      base_url: '',
      log_level: 'INFO',
    },
    origin: 'http://127.0.0.1',
    requestedModel: 'grok-4.20-beta',
  });

  const message = parsed.choices?.[0]?.message;
  assert.ok(message && typeof message === 'object');
  assert.equal(
    message.reasoning_content,
    '[Agent 3] [WebSearch] Jio SIM cost\n\n[Agent 3] [WebSearch] Jio eSIM activation\n\n[Agent 1] [AgentThink] summarize findings',
  );
  assert.equal(message.reasoning, message.reasoning_content);
});
