const OpenAI = require('openai');
const { Logger } = require('./logger');

const GEMINI_MODELS = [
  'gemini-3.7-flash',
  'gemini-3.1-pro-preview',
  'gemini-3.5-flash-lite',
];
const GEMINI_DEFAULT_MODEL = GEMINI_MODELS[0];

const PROVIDERS = {
  openai: {
    name: 'OpenAI',
    baseURL: 'https://api.openai.com/v1',
    defaultModel: 'gpt-5.6',
    models: ['gpt-5.6', 'gpt-5.6-terra', 'gpt-5.6-luna'],
    envKey: 'OPENAI_API_KEY',
  },
  openrouter: {
    name: 'OpenRouter',
    baseURL: 'https://openrouter.ai/api/v1',
    defaultModel: 'openai/gpt-5.6-sol',
    models: ['openai/gpt-5.6-sol', 'anthropic/claude-fable-5', 'google/gemini-3.7-flash', 'moonshotai/kimi-k3', 'z-ai/glm-5.3'],
    envKey: 'OPENROUTER_API_KEY',
  },
  kimi: {
    name: 'Kimi (Moonshot AI)',
    baseURL: 'https://api.moonshot.ai/v1',
    defaultModel: 'kimi-k3',
    models: ['kimi-k3', 'kimi-k2.7-code', 'kimi-k2.6'],
    envKey: 'MOONSHOT_API_KEY',
  },
  mimo: {
    name: 'MiMo (Xiaomi)',
    baseURL: 'https://api.xiaomimimo.com/v1',
    defaultModel: 'mimo-v2.5-pro',
    models: ['mimo-v2.5-pro', 'mimo-v2.5'],
    envKey: 'MIMO_API_KEY',
  },
  glm: {
    name: 'GLM (Zhipu AI)',
    baseURL: 'https://api.z.ai/api/paas/v4/',
    defaultModel: 'glm-5.3',
    models: ['glm-5.3', 'glm-5.2', 'glm-5.1'],
    envKey: 'GLM_API_KEY',
  },
};

class AITextService {
  constructor(credentials = {}) {
    this.logger = new Logger('AITextService');
    this.client = null;
    this.gemini = null;
    this.model = null;
    this.providerName = null;
    this.isCustomProvider = false;
    this.fallbackModel = null;
    this._customApiKey = null;

    this._init(credentials);
  }

  _init(credentials) {
    const customConfig = this._getCustomProviderConfig();
    if (customConfig) {
      return this._initCustomProvider(customConfig);
    }

    const provider = credentials.aiProvider?.provider;
    const apiKey = credentials.aiProvider?.apiKey;
    const model = credentials.aiProvider?.model;

    if (provider && PROVIDERS[provider] && apiKey) {
      return this._initOpenAICompatible(PROVIDERS[provider], apiKey, model);
    }

    for (const [, preset] of Object.entries(PROVIDERS)) {
      const key = process.env[preset.envKey];
      if (key) {
        return this._initOpenAICompatible(preset, key);
      }
    }

    const geminiKey = credentials.gemini?.apiKey || process.env.GEMINI_API_KEY;
    if (geminiKey) {
      return this._initGemini(geminiKey, credentials.gemini?.model);
    }

    this.logger.warn('No AI text provider configured — text generation unavailable');
  }

  _initOpenAICompatible(preset, apiKey, model) {
    this.client = new OpenAI({ apiKey, baseURL: preset.baseURL });
    this.model = model || preset.defaultModel;
    this.providerName = preset.name;
    this.logger.info(`${preset.name} initialized (model: ${this.model})`);
  }

  // Self-hosted OpenAI-compatible proxy (e.g. 9router). Master-switched via
  // CUSTOM_AI_PROVIDER_ENABLED so the built-in provider selection above stays
  // untouched when it's off.
  _getCustomProviderConfig() {
    const enabled = /^(1|true|yes)$/i.test(String(process.env.CUSTOM_AI_PROVIDER_ENABLED || '').trim());
    if (!enabled) return null;

    const baseURL = String(process.env.CUSTOM_AI_BASE_URL || '').trim();
    const model = String(process.env.CUSTOM_AI_MODEL || '').trim();
    if (!baseURL || !model) {
      this.logger.warn('CUSTOM_AI_PROVIDER_ENABLED is true but CUSTOM_AI_BASE_URL or CUSTOM_AI_MODEL is missing — ignoring custom provider');
      return null;
    }

    const fallbackModel = String(process.env.CUSTOM_AI_MODEL_FALLBACK || '').trim() || null;
    const timeoutMs = Math.max(1000, Number(process.env.CUSTOM_AI_TIMEOUT_MS) || 60000);
    return { apiKey: process.env.CUSTOM_AI_API_KEY || '', baseURL, model, fallbackModel, timeoutMs };
  }

  _initCustomProvider(config) {
    try {
      this.client = new OpenAI({
        apiKey: config.apiKey || 'unset',
        baseURL: config.baseURL,
        timeout: config.timeoutMs,
      });
      this.model = config.model;
      this.fallbackModel = config.fallbackModel && config.fallbackModel !== config.model ? config.fallbackModel : null;
      this.providerName = 'Custom AI provider';
      this.isCustomProvider = true;
      // Kept only to strip literal occurrences from error/log text — the key
      // itself is never logged (FR5).
      this._customApiKey = config.apiKey || null;
      this.logger.info(
        `Custom AI provider initialized (baseURL: ${config.baseURL}, model: ${this.model}${this.fallbackModel ? `, fallback: ${this.fallbackModel}` : ''})`
      );
    } catch (error) {
      this.logger.error('Failed to initialize custom AI provider:', error.message);
    }
  }

  _initGemini(apiKey, model) {
    try {
      const { GoogleGenAI } = require('@google/genai');
      this.gemini = new GoogleGenAI({ apiKey });
      this.model = model || GEMINI_DEFAULT_MODEL;
      this.providerName = 'Google Gemini';
      this.logger.info(`Gemini initialized (model: ${this.model})`);
    } catch (error) {
      this.logger.error('Failed to initialize Gemini:', error.message);
    }
  }

  async generateText(prompt, options = {}) {
    const model = options.model || this.model;
    const maxTokens = options.maxTokens || 2048;
    const temperature = options.temperature ?? 0.7;

    if (this.gemini) {
      const config = { maxOutputTokens: maxTokens };
      if (!/^gemini-3\.(?:[5-9]|\d{2,})-/.test(model)) config.temperature = temperature;
      const response = await this.gemini.models.generateContent({
        model,
        contents: prompt,
        config,
      });
      const text = response && response.text;
      if (typeof text !== 'string' || !text.trim()) {
        throw new Error(
          `${this.providerName} returned an empty response. Check the API key and model quota — free-tier Gemini keys are rate-limited and can return empty output.`
        );
      }
      return text;
    }

    if (!this.client) {
      throw new Error('No AI text provider configured');
    }

    if (this.isCustomProvider && !options.model) {
      return this._generateWithFallback(prompt, model, maxTokens, temperature);
    }

    return this._chatComplete(model, prompt, maxTokens, temperature);
  }

  // Retries once against CUSTOM_AI_MODEL_FALLBACK when the primary custom-provider
  // model errors, mirroring the bounded-retry shape used for generation stages
  // (see GENERATION_STAGE_MAX_ATTEMPTS in generation-recovery-service.js).
  async _generateWithFallback(prompt, primaryModel, maxTokens, temperature) {
    try {
      return await this._chatComplete(primaryModel, prompt, maxTokens, temperature);
    } catch (error) {
      if (!this.fallbackModel) {
        throw this._describeCustomProviderError(error);
      }
      this.logger.warn(
        `Custom AI provider model "${primaryModel}" failed (${this._safeErrorSummary(error)}); retrying with fallback "${this.fallbackModel}"`
      );
      try {
        return await this._chatComplete(this.fallbackModel, prompt, maxTokens, temperature);
      } catch (fallbackError) {
        throw this._describeCustomProviderError(fallbackError, true);
      }
    }
  }

  async _chatComplete(model, prompt, maxTokens, temperature) {
    const params = {
      model,
      messages: [{ role: 'user', content: prompt }],
      temperature,
    };

    try {
      // Newer OpenAI models (gpt-5.x and later) reject the legacy max_tokens
      // parameter with a 400 error and require max_completion_tokens instead.
      const response = await this.client.chat.completions.create({
        ...params,
        max_completion_tokens: maxTokens,
      });
      return this._extractContent(response);
    } catch (error) {
      // Older models and some providers reject max_completion_tokens with a 400;
      // retry the same request using the legacy max_tokens spelling.
      if (
        error &&
        error.status === 400 &&
        /max(_completion)?_tokens/i.test(error.message || '')
      ) {
        const response = await this.client.chat.completions.create({
          ...params,
          max_tokens: maxTokens,
        });
        return this._extractContent(response);
      }
      throw error;
    }
  }

  // Redacts anything that looks like a credential before it can reach logs or
  // a thrown error message (FR5: never write the API key itself into logs).
  _redact(text) {
    let out = String(text || '')
      .replace(/Bearer\s+[A-Za-z0-9._~-]+/gi, 'Bearer [redacted]')
      .replace(/"?api[_-]?key"?\s*[:=]\s*"?[A-Za-z0-9._~-]+"?/gi, 'api_key=[redacted]');
    // Belt-and-suspenders: strip literal occurrences of the configured key too,
    // in case an upstream error echoes it back in free-form text.
    if (this._customApiKey) {
      out = out.split(this._customApiKey).join('[redacted]');
    }
    return out;
  }

  _safeErrorSummary(error) {
    const status = error?.status || error?.response?.status;
    const rawBody =
      (error && typeof error.error === 'object' && error.error) ? JSON.stringify(error.error) : (error?.message || String(error));
    const body = this._redact(rawBody).slice(0, 300);
    return status ? `HTTP ${status}: ${body}` : body;
  }

  // Classifies the failure (tunnel unreachable vs. bad key vs. unknown model
  // alias vs. upstream error) so the readiness check and logs surface an
  // actionable cause instead of a generic "request failed".
  _describeCustomProviderError(error, afterFallback = false) {
    const status = error?.status || error?.response?.status;
    const code = error?.code || error?.cause?.code;
    const message = String(error?.message || '');
    const summary = this._safeErrorSummary(error);

    let category = 'Custom AI provider request failed';
    if (code === 'ECONNREFUSED' || code === 'ENOTFOUND' || code === 'ETIMEDOUT' || code === 'ECONNRESET' || /timeout/i.test(message)) {
      category = 'Custom AI provider unreachable (tunnel or network error)';
    } else if (status === 401 || status === 403) {
      category = 'Custom AI provider rejected the API key';
    } else if (status === 404 || /model/i.test(message)) {
      category = 'Custom AI provider model alias not found';
    } else if (typeof status === 'number' && status >= 500) {
      category = 'Custom AI provider upstream server error';
    }

    const wrapped = new Error(`${category}${afterFallback ? ' (fallback model also failed)' : ''}: ${summary}`);
    wrapped.status = status;
    wrapped.cause = error;
    return wrapped;
  }

  _extractContent(response) {
    const content =
      response &&
      response.choices &&
      response.choices[0] &&
      response.choices[0].message
        ? response.choices[0].message.content
        : null;

    if (typeof content !== 'string' || !content.trim()) {
      // A null/empty body used to surface as cryptic "Unexpected end of JSON input"
      // in the agents' JSON parsers. Report the real cause instead.
      throw new Error(
        `${this.providerName} returned an empty response. Check the API key and model quota.`
      );
    }
    return content;
  }

  isAvailable() {
    return !!(this.client || this.gemini);
  }
}

module.exports = { AITextService, PROVIDERS, GEMINI_MODELS, GEMINI_DEFAULT_MODEL };
