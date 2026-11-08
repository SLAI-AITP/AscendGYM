// Presentation names only. Raw IDs remain the keys for measurements, pricing,
// imports and filters; a display correction must never change those joins.
import { SOTA_MODELS } from './vendors.js';

const harnessNames = {
  'claude-code': 'Claude Code',
  claudecode: 'Claude Code',
  codex: 'Codex',
  opencode: 'OpenCode',
  'open-code': 'OpenCode',
  'default-warm': 'Default Warm',
  'default-cold': 'Default Cold',
  'default-code-only': 'Default Code Only',
  'default-no-lessons': 'Default No Lessons'
};

const modelNames = {
  'claude-opus-5': 'Claude-Opus-5',
  'gpt-5.6-terra': 'GPT-5.6 Terra',
  'gpt-5.6-sol': 'GPT-5.6 Sol',
  'gpt-5.6-luna': 'GPT-5.6 Luna',
  'glm-5.3': 'GLM-5.3',
  'deepseek-v4-flash-bf16': 'DeepSeek-V4-Flash',
  'deepseek-v4-flash': 'DeepSeek-V4-Flash',
  // Full names are confirmed by the run IDs in the supplementary source
  // (qwen3.5-122b-a10b / qwen3.6-35b-a3b) and Qwen's model cards.
  'qwen3.5-122b': 'Qwen3.5-122B-A10B',
  'qwen3.6-35b': 'Qwen3.6-35B-A3B'
};

// Apply the same spelling to IDs, already formatted names and imported aliases.
// Unknown identities retain their supplied name rather than inventing a model.
const nameKey = value => value.trim().toLowerCase().replace(/[\s_-]+/g, '-');
function nameRegistry(pairs) {
  const registry = new Map();
  for (const [id, name] of pairs) {
    registry.set(nameKey(id), name);
    registry.set(nameKey(name), name);
  }
  return registry;
}
const harnessRegistry = nameRegistry(Object.entries(harnessNames));
const modelRegistry = nameRegistry([
  ...SOTA_MODELS.map(model => [model.id, model.id]),
  ...Object.entries(modelNames)
]);

function nameFor(value, registry) {
  const id = typeof value === 'string' ? value : value?.id;
  return (typeof id === 'string' ? registry.get(nameKey(id)) : undefined)
    ?? value?.display_name ?? id ?? '';
}

export const harnessDisplayName = value => nameFor(value, harnessRegistry);
export const modelDisplayName = value => nameFor(value, modelRegistry);
export const configurationDisplayName = entry =>
  `${harnessDisplayName(entry.agent)} × ${modelDisplayName(entry.engine_model)}`;
