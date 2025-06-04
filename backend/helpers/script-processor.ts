// @ts-ignore - dirty-json has no types
import dJSON from "dirty-json";
import { ConversationData } from "./types.js";
import { 
  getGeminiApiKey,
  GEMINI_API_URL,
  GEMINI_MODEL,
  GEMINI_TEMPERATURE,
  GEMINI_MAX_OUTPUT_TOKENS,
  GEMINI_TOP_P,
  GEMINI_TOP_K,
  GEMINI_CANDIDATE_COUNT,
  getGeminiRequestConfig,
  validateGeminiConfig,
} from "./gemini-config.js";
import { logToFile } from "./logging.js";

interface GeminiResponse {
  candidates: Array<{
    content: {
      parts: Array<{
        text: string;
      }>;
    };
  }>;
}

export async function generateScript(
  userPrompt: string,
  systemPrompt: string
): Promise<{
  success: boolean;
  conversation?: ConversationData;
  error?: string;
  details?: any;
}> {
  try {
    const configValidation = validateGeminiConfig();
    if (!configValidation.isValid) {
      return {
        success: false,
        error: configValidation.error,
      };
    }

    const combinedPrompt = `${systemPrompt}\n\n=== USER REQUEST ===\n${userPrompt}`;
    const requestBody = getGeminiRequestConfig(combinedPrompt);

    console.log(`Sending request to Gemini API:`);
    console.log(`📡 Model: ${GEMINI_MODEL}`);
    console.log(`🌡️  Temperature: ${GEMINI_TEMPERATURE}`);
    console.log(`📊 Max tokens: ${GEMINI_MAX_OUTPUT_TOKENS}`);
    console.log(`🎯 TopP: ${GEMINI_TOP_P}, TopK: ${GEMINI_TOP_K}`);
    console.log(`🔢 Candidates: ${GEMINI_CANDIDATE_COUNT}`);

    logToFile("api_request", {
      model: GEMINI_MODEL,
      userPrompt: userPrompt,
      systemPrompt: systemPrompt,
      combinedPromptLength: combinedPrompt.length,
      requestBody: requestBody,
    });

    const response = await fetch(`${GEMINI_API_URL}?key=${getGeminiApiKey()}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
      let errorText = "";
      try {
        errorText = await response.text();
        console.error("API Error:", errorText);
      } catch (readError) {
        console.error("Could not read error text:", readError);
      }

      logToFile("error", {
        errorType: "http_error",
        action: "generate_script",
        httpStatus: response.status,
        httpStatusText: response.statusText,
        apiError: errorText,
        userPrompt: userPrompt,
        systemPrompt: systemPrompt,
      });

      return {
        success: false,
        error: `HTTP error when requesting to Gemini API: ${response.status} ${response.statusText}`,
        details: {
          status: response.status,
          statusText: response.statusText,
          apiError: errorText || "Could not get error details",
        },
      };
    }

    // Parse response
    let data: GeminiResponse;
    try {
      data = await response.json();
    } catch (jsonError) {
      console.error("JSON parsing error from API response:", jsonError);

      let responseText = "";
      try {
        responseText = await response.text();
      } catch (textError) {
        console.error("Could not read response text:", textError);
      }

      logToFile("error", {
        errorType: "json_parse_error",
        action: "generate_script",
        jsonError: jsonError instanceof Error ? jsonError.message : "Unknown error",
        responseStatus: response.status,
        responseStatusText: response.statusText,
        responseContentType: response.headers.get("content-type"),
        responsePreview: responseText.substring(0, 1000),
        userPrompt: userPrompt,
        systemPrompt: systemPrompt,
      });

      return {
        success: false,
        error: "Received invalid JSON response from Gemini API",
        details: {
          jsonError: jsonError instanceof Error ? jsonError.message : "Unknown JSON parsing error",
          status: response.status,
          statusText: response.statusText,
          contentType: response.headers.get("content-type"),
          responsePreview: responseText.substring(0, 1000),
        },
      };
    }

    console.log("Checking API response structure...");

    // Validate API response structure
    const structureValidation = validateApiResponseStructure(data);
    if (!structureValidation.isValid) {
      logToFile("error", {
        errorType: "api_structure_error",
        action: "generate_script",
        structureError: structureValidation.error,
        apiResponse: JSON.stringify(data, null, 2),
        userPrompt: userPrompt,
        systemPrompt: systemPrompt,
      });

      return {
        success: false,
        error: structureValidation.error,
        details: { rawResponse: JSON.stringify(data, null, 2) },
      };
    }

    const generatedText = data.candidates[0].content.parts[0].text;

    console.log("Response received, parsing JSON...");
    console.log("Generated text length:", generatedText.length);
    console.log("Approximate token count:", Math.round(generatedText.length / 4));
    console.log("First 500 characters:", generatedText.substring(0, 500));

    logToFile("api_response", {
      model: GEMINI_MODEL,
      responseLength: generatedText.length,
      fullResponse: generatedText,
      apiResponseStructure: {
        hasCandidates: !!data.candidates,
        candidatesCount: data.candidates?.length || 0,
        hasContent: !!data.candidates?.[0]?.content,
        hasParts: !!data.candidates?.[0]?.content?.parts,
        partsCount: data.candidates?.[0]?.content?.parts?.length || 0,
      },
    });

    // Parse conversation from generated text
    const parseResult = parseConversationFromText(generatedText, userPrompt, systemPrompt);
    if (!parseResult.success) {
      return parseResult;
    }

    const conversation = parseResult.conversation!;
    
    console.log("Structure checked. Number of replies:", conversation.conversation.length);

    return {
      success: true,
      conversation,
    };

  } catch (error) {
    console.error("Error generating script:", error);

    let errorDetails: any = {
      message: error instanceof Error ? error.message : "Unknown error",
      stack: error instanceof Error ? error.stack : undefined,
      name: error instanceof Error ? error.name : "UnknownError",
    };

    // Network error classification
    if (error instanceof TypeError && error.message.includes("fetch")) {
      errorDetails.type = "network_error";
    }

    // HTTP error classification
    if (error instanceof Error && error.message.includes("HTTP error")) {
      errorDetails.type = "api_error";
    }

    logToFile("error", {
      errorType: "general_error",
      action: "generate_script",
      error: errorDetails,
      userPrompt: userPrompt,
      systemPrompt: systemPrompt,
    });

    return {
      success: false,
      error: "Error generating script",
      details: errorDetails,
    };
  }
}

function validateApiResponseStructure(data: any): { isValid: boolean; error?: string } {
  if (!data) {
    return { isValid: false, error: "Received empty response from API" };
  }

  if (!data.candidates) {
    return { 
      isValid: false, 
      error: "API returned incorrect data structure: missing candidates field" 
    };
  }

  if (!Array.isArray(data.candidates) || data.candidates.length === 0) {
    return { 
      isValid: false, 
      error: "API returned empty candidates array" 
    };
  }

  if (!data.candidates[0]) {
    return { 
      isValid: false, 
      error: "First candidate element is missing" 
    };
  }

  if (!data.candidates[0].content) {
    return { 
      isValid: false, 
      error: "Missing content field in first candidate" 
    };
  }

  if (!data.candidates[0].content.parts) {
    return { 
      isValid: false, 
      error: "Missing parts field in content" 
    };
  }

  if (!Array.isArray(data.candidates[0].content.parts) || 
      data.candidates[0].content.parts.length === 0) {
    return { 
      isValid: false, 
      error: "Parts array is empty or not an array" 
    };
  }

  if (!data.candidates[0].content.parts[0]) {
    return { 
      isValid: false, 
      error: "First parts element is missing" 
    };
  }

  if (!data.candidates[0].content.parts[0].text) {
    return { 
      isValid: false, 
      error: "Missing text in first parts element" 
    };
  }

  return { isValid: true };
}

function parseConversationFromText(
  generatedText: string,
  userPrompt: string,
  systemPrompt: string
): {
  success: boolean;
  conversation?: ConversationData;
  error?: string;
  details?: any;
} {
  let conversation: ConversationData;
  
  try {
    // Use dirty-json for more robust parsing
    conversation = dJSON.parse(generatedText);
    console.log("✅ JSON successfully parsed with dirty-json");
  } catch (parseError) {
    console.error("❌ dirty-json parsing error:", parseError);

    // Fallback: try extracting JSON from markdown blocks
    try {
      const jsonMatch =
        generatedText.match(/```json\s*(\{[\s\S]*?\})\s*```/) ||
        generatedText.match(/(\{[\s\S]*?\})/);

      if (jsonMatch && jsonMatch[1]) {
        // Try dirty-json for extracted JSON
        try {
          conversation = dJSON.parse(jsonMatch[1]);
          console.log("✅ JSON successfully extracted from markdown and parsed with dirty-json");
        } catch (dirtyJsonError) {
          // Last attempt with standard JSON.parse
          conversation = JSON.parse(jsonMatch[1]);
          console.log("✅ JSON successfully extracted from markdown and parsed with standard parser");
        }
      } else {
        throw new Error("Could not find JSON in response");
      }
    } catch (fallbackError) {
      console.error("Fallback parsing also failed:", fallbackError);

      logToFile("error", {
        errorType: "script_json_parse_error",
        action: "generate_script",
        parsingMethod: "dirty-json + fallback",
        originalError: parseError instanceof Error ? parseError.message : "Unknown parsing error",
        fallbackError: fallbackError instanceof Error ? fallbackError.message : "Unknown fallback error",
        textLength: generatedText.length,
        firstChars: generatedText.substring(0, 1000),
        lastChars: generatedText.length > 1000 ? 
          generatedText.substring(generatedText.length - 500) : "",
        fullResponse: generatedText,
        userPrompt: userPrompt,
        systemPrompt: systemPrompt,
      });

      return {
        success: false,
        error: "Could not parse JSON from model response",
        details: {
          originalError: parseError instanceof Error ? parseError.message : "Unknown parsing error",
          fallbackError: fallbackError instanceof Error ? fallbackError.message : "Unknown fallback error",
          textLength: generatedText.length,
          firstChars: generatedText.substring(0, 1000),
          lastChars: generatedText.length > 1000 ? 
            generatedText.substring(generatedText.length - 500) : "",
        },
      };
    }
  }

  // Validate conversation structure
  const validationResult = validateConversationStructure(conversation, userPrompt, systemPrompt);
  if (!validationResult.success) {
    return validationResult;
  }

  return {
    success: true,
    conversation,
  };
}

function validateConversationStructure(
  conversation: any,
  userPrompt: string,
  systemPrompt: string
): {
  success: boolean;
  conversation?: ConversationData;
  error?: string;
  details?: any;
} {
  console.log("Checking conversation object structure...");
  
  if (!conversation) {
    logToFile("error", {
      errorType: "conversation_structure_error",
      action: "generate_script",
      issue: "conversation_is_null_or_undefined",
      conversationValue: conversation,
      userPrompt: userPrompt,
      systemPrompt: systemPrompt,
    });

    return {
      success: false,
      error: "Received conversation object is empty",
      details: { conversationValue: conversation },
    };
  }

  if (!conversation.conversation) {
    console.log("❌ conversation.conversation field is missing or undefined");
    
    logToFile("error", {
      errorType: "conversation_structure_error",
      action: "generate_script",
      issue: "missing_conversation_array",
      conversationKeys: Object.keys(conversation || {}),
      conversationValue: conversation,
      userPrompt: userPrompt,
      systemPrompt: systemPrompt,
    });

    return {
      success: false,
      error: "Incomplete conversation structure: missing 'conversation' field",
      details: {
        receivedKeys: Object.keys(conversation || {}),
        expectedStructure: {
          podcast_name: "string",
          conversation: "array",
        },
        actualStructure: conversation,
      },
    };
  }

  if (!Array.isArray(conversation.conversation)) {
    console.log("❌ conversation.conversation field is not an array");
    
    logToFile("error", {
      errorType: "conversation_structure_error",
      action: "generate_script",
      issue: "conversation_not_array",
      conversationType: typeof conversation.conversation,
      conversationValue: conversation.conversation,
      userPrompt: userPrompt,
      systemPrompt: systemPrompt,
    });

    return {
      success: false,
      error: "Invalid structure: 'conversation' field should be an array",
      details: {
        actualType: typeof conversation.conversation,
        actualValue: conversation.conversation,
      },
    };
  }

  return {
    success: true,
    conversation,
  };
} 