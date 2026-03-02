import test from 'node:test';
import assert from 'node:assert/strict';

import { buildConversationPayload } from '../src/grok/conversation.ts';
import { parseOpenAiFromGrokNdjson } from '../src/grok/processor.ts';

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
