// Environment variables are already loaded in server.ts, avoiding duplicate loading

// Get parameters from .env - using functions to ensure variables are read at runtime
export const getGeminiApiKey = () => process.env.GEMINI_API_KEY;
export const GEMINI_MODEL = process.env.GEMINI_MODEL || 'gemini-2.5-flash-preview-05-20';
export const GEMINI_TEMPERATURE = parseFloat(process.env.GEMINI_TEMPERATURE || '0.7');
export const GEMINI_MAX_OUTPUT_TOKENS = parseInt(process.env.GEMINI_MAX_OUTPUT_TOKENS || '65536');
export const GEMINI_TOP_P = parseFloat(process.env.GEMINI_TOP_P || '0.95');
export const GEMINI_TOP_K = parseInt(process.env.GEMINI_TOP_K || '40');
export const GEMINI_CANDIDATE_COUNT = parseInt(process.env.GEMINI_CANDIDATE_COUNT || '1');

// Automatically form the URL for the Gemini API
export const GEMINI_API_URL = `https://generativelanguage.googleapis.com/v1beta/models/${GEMINI_MODEL}:generateContent`;

// Security settings
export const SAFETY_SETTINGS = [
  {
    category: "HARM_CATEGORY_HARASSMENT",
    threshold: process.env.GEMINI_HARASSMENT_THRESHOLD || "BLOCK_NONE"
  },
  {
    category: "HARM_CATEGORY_HATE_SPEECH", 
    threshold: process.env.GEMINI_HATE_SPEECH_THRESHOLD || "BLOCK_NONE"
  },
  {
    category: "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    threshold: process.env.GEMINI_SEXUALLY_EXPLICIT_THRESHOLD || "BLOCK_NONE"
  },
  {
    category: "HARM_CATEGORY_DANGEROUS_CONTENT",
    threshold: process.env.GEMINI_DANGEROUS_CONTENT_THRESHOLD || "BLOCK_NONE"
  }
];

// Function to get the full request configuration
export function getGeminiRequestConfig(combinedPrompt: string) {
  return {
    contents: [
      {
        parts: [
          {
            text: combinedPrompt
          }
        ]
      }
    ],
    generationConfig: {
      responseMimeType: "application/json",
      temperature: GEMINI_TEMPERATURE,
      maxOutputTokens: GEMINI_MAX_OUTPUT_TOKENS,
      topP: GEMINI_TOP_P,
      topK: GEMINI_TOP_K,
      candidateCount: GEMINI_CANDIDATE_COUNT
    },
    safetySettings: SAFETY_SETTINGS
  };
}

// Function to check for the presence of an API key
export function validateGeminiConfig(): { isValid: boolean; error?: string } {
  const GEMINI_API_KEY = getGeminiApiKey();
  if (!GEMINI_API_KEY) {
    return {
      isValid: false,
      error: 'GEMINI_API_KEY not found in environment variables. Check .env file'
    };
  }
  
  return { isValid: true };
} 