import { config } from "dotenv";
import { readFileSync, writeFileSync, existsSync } from "fs";
import { join } from "path";
import { ConversationData, Message } from "./helpers/types.js";
import {
  GEMINI_API_URL,
  getGeminiApiKey,
  GEMINI_MODEL,
  GEMINI_TEMPERATURE,
  GEMINI_MAX_OUTPUT_TOKENS,
  GEMINI_TOP_P,
  GEMINI_TOP_K,
  GEMINI_CANDIDATE_COUNT,
  getGeminiRequestConfig,
  validateGeminiConfig,
} from "./helpers/gemini-config.js";

config({ path: "../.env" });

interface GeminiResponse {
  candidates: Array<{
    content: {
      parts: Array<{
        text: string;
      }>;
    };
  }>;
}

async function createConversation(userPrompt: string, systemPrompt?: string) {
  const configValidation = validateGeminiConfig();
  if (!configValidation.isValid) {
    throw new Error(configValidation.error);
  }

  // Read system prompt from CLI file (expecting it to be in a separate file for CLI usage)
  let finalSystemPrompt = systemPrompt;
  if (!finalSystemPrompt) {
    const systemPromptPath = "../prompts/system-prompt-cli.md";
    if (existsSync(systemPromptPath)) {
      finalSystemPrompt = readFileSync(systemPromptPath, "utf-8");
    } else {
      throw new Error(
        "system-prompt-cli.md not found, use this script through web interface"
      );
    }
  }

  const userPromptPath = join("../prompts", userPrompt);
  let userPromptContent = "";
  if (existsSync(userPromptPath)) {
    userPromptContent = readFileSync(userPromptPath, "utf-8");
  } else {
    userPromptContent = userPrompt;
  }

  const combinedPrompt = `${finalSystemPrompt}\n\nUser: ${userPromptContent}`;

  const requestConfig = getGeminiRequestConfig(combinedPrompt);

  console.log(`📡 Sending request to: ${GEMINI_API_URL}`);
  console.log(`📊 Model: ${process.env.GEMINI_MODEL}`);
  console.log(`🎛️ Temperature: ${process.env.GEMINI_TEMPERATURE}`);

  const response = await fetch(`${GEMINI_API_URL}?key=${getGeminiApiKey()}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(requestConfig),
  });

  if (!response.ok) {
    const errorBody = await response.text();
    throw new Error(
      `HTTP error ${response.status}: ${response.statusText}\nResponse: ${errorBody}`
    );
  }

  const data: GeminiResponse = await response.json();

  if (!data.candidates || !data.candidates[0]) {
    throw new Error("No response from Gemini API");
  }

  const responseText = data.candidates[0].content.parts[0].text;

  let conversationData;
  try {
    // With responseMimeType: "application/json", the response should already be clean JSON
    conversationData = JSON.parse(responseText);
  } catch (error) {
    console.log("Failed to parse JSON directly, trying to extract from markdown blocks");
    
    // Fallback: try to extract JSON from markdown blocks in case API still wrapped the response
    const jsonMatch = responseText.match(/```json\n([\s\S]*?)\n```/);
    if (jsonMatch) {
      try {
        conversationData = JSON.parse(jsonMatch[1]);
      } catch (innerError) {
        throw new Error(`Failed to parse JSON from markdown: ${innerError}`);
      }
    } else {
      throw new Error(`Failed to parse response as JSON: ${error}`);
    }
  }

  const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
  const filename = `conversation-${timestamp}.json`;
  const filepath = join("../generated-scripts", filename);

  writeFileSync(filepath, JSON.stringify(conversationData, null, 2));

  console.log(`✅ Conversation saved to: ${filepath}`);
  return { conversationData, filepath };
}

// Export function and run if called directly
export { createConversation };

if (import.meta.url === `file://${process.argv[1]}`) {
  const userPrompt = process.argv[2] || "default.md";
  createConversation(userPrompt)
    .then((result) => {
      console.log("Success:", result.filepath);
    })
    .catch((error) => {
      console.error("Error:", error.message);
      process.exit(1);
    });
}
