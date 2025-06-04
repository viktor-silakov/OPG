import express from "express";
import cors from "cors";
import { config } from "dotenv";
import {
  readFileSync,
  writeFileSync,
  existsSync,
  mkdirSync,
  unlinkSync,
} from "fs";
import { join } from "path";
import { fileURLToPath } from "url";
import { dirname } from "path";
import { ConversationData } from "./helpers/types.js";
import { createServer } from "http";
import { WebSocketServer } from "ws";
import multer from "multer";
import {
  handleWebSocketConnection,
  logToFile,
  getPodcastsList,
  getPromptsList,
  getPromptContent,
  savePromptContent,
  deletePrompt,
  getVoicesList,
  createVoiceReference,
  deleteVoice,
  testVoice,
  saveVoiceFiles,
  voiceExists,
  getVoiceAudioPath,
  ensureVoicesDirectory,
  getLogsList,
  generateScript,
  processGeneratePodcastRequest,
  startBackgroundPodcastGeneration,
  getVoiceReferenceText,
} from "./helpers/index.js";

// Get the directory path for ES modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Load environment variables from root folder using absolute path
const envPath = join(__dirname, "..", ".env");
config({ path: envPath });

const app = express();
const server = createServer(app);
const wss = new WebSocketServer({ server });
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json({ limit: "10mb" }));
app.use(express.static("public"));
app.use("/output", express.static("../output"));

// Interfaces
interface GenerateScriptRequest {
  userPrompt: string;
  systemPrompt?: string;
}

interface GeneratePodcastRequest {
  conversationData?: ConversationData;
  conversationFile?: string;
}

// Handle WebSocket connections
handleWebSocketConnection(wss);

// Ensure directories exist
if (!existsSync("../generated-scripts")) {
  mkdirSync("../generated-scripts", { recursive: true });
}

const SCRIPTS_DIR = "../generated-scripts";

// Podcast script generation endpoint
app.post("/generate-script", async (req, res) => {
  try {
    const { userPrompt, systemPrompt }: GenerateScriptRequest = req.body;

    if (!userPrompt) {
      return res.status(400).json({
        error: "userPrompt is required",
      });
    }

    if (!systemPrompt) {
      return res.status(400).json({
        error: "systemPrompt is required. Please pass it from the frontend.",
      });
    }

    const result = await generateScript(userPrompt, systemPrompt);

    if (!result.success) {
      return res.status(500).json({
        error: result.error,
        details: result.details,
        suggestion: getSuggestionForError(result.error),
      });
    }

    const conversation = result.conversation!;

    const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
    const filename = `conversation-${timestamp}.json`;
    const filepath = join(SCRIPTS_DIR, filename);
    writeFileSync(filepath, JSON.stringify(conversation, null, 2), "utf-8");

    console.log(`Conversation successfully saved to ${filepath}`);
    console.log(`Number of replies: ${conversation.conversation.length}`);

    logToFile("success", {
      action: "generate_script",
      filename: filename,
      filepath: filepath,
      conversation: conversation,
      messageCount: conversation.conversation.length,
      userPrompt: userPrompt,
      systemPrompt: systemPrompt,
    });

    res.json({
      success: true,
      message: "Podcast script successfully generated",
      filename: filename,
      conversation: conversation,
      messageCount: conversation.conversation.length,
    });
  } catch (error) {
    console.error("Error generating script:", error);

    let errorDetails: any = {
      message: error instanceof Error ? error.message : "Unknown error",
      stack: error instanceof Error ? error.stack : undefined,
      name: error instanceof Error ? error.name : "UnknownError",
    };

    if (error instanceof TypeError && error.message.includes("fetch")) {
      errorDetails.type = "network_error";
      errorDetails.suggestion = "Check internet connection and Gemini API availability";
    }

    if (error instanceof Error && error.message.includes("HTTP error")) {
      errorDetails.type = "api_error";
      errorDetails.suggestion = "Check API key settings and request limits";
    }

    logToFile("error", {
      errorType: "general_error",
      action: "generate_script",
      error: errorDetails,
      userPrompt: req.body.userPrompt,
      systemPrompt: req.body.systemPrompt,
    });

    res.status(500).json({
      error: "Error generating script",
      details: errorDetails,
      timestamp: new Date().toISOString(),
      requestInfo: {
        hasUserPrompt: !!req.body.userPrompt,
        hasSystemPrompt: !!req.body.systemPrompt,
        userPromptLength: req.body.userPrompt?.length || 0,
        systemPromptLength: req.body.systemPrompt?.length || 0,
      },
    });
  }
});

// Podcast generation endpoint
app.post("/generate-podcast", async (req, res) => {
  try {
    const result = await processGeneratePodcastRequest(req.body);

    if (!result.success) {
      return res.status(404).json({
        success: false,
        message: result.error,
      });
    }

    const { conversation, jobId, messageCount } = result;

    // Send immediate response
    res.json({
      success: true,
      message: "Podcast generation started in background",
      podcast_name: conversation!.podcast_name,
      filename: conversation!.filename,
      messageCount,
      status: "processing",
      jobId,
    });

    // Start background processing
    startBackgroundPodcastGeneration(conversation!, jobId!, messageCount!);
  } catch (error) {
    console.error("Error in /generate-podcast:", error);
    res.status(500).json({
      success: false,
      message: "Server error during podcast generation",
    });
  }
});

// Status check endpoint
app.get("/status", (req, res) => {
  res.json({
    status: "ok",
    message: "Podcast server is running",
    endpoints: {
      "POST /generate-script": "Generate podcast script",
      "POST /generate-podcast": "Generate audio podcast",
      "GET /status": "Check server status",
      "GET /podcasts": "List available podcasts",
      "GET /prompts": "List system prompts",
      "GET /download/:podcastName/:fileName": "Download podcast",
      "GET /logs": "List log files",
      "GET /logs/:filename": "Log file content",
      "GET /api/voices": "List voices",
      "POST /api/voices/create": "Create new voice",
      "DELETE /api/voices/:voiceName": "Delete voice",
      "POST /api/voices/test": "Test voice",
      "GET /api/voices/download/:voiceName": "Download voice",
      "GET /api/voices/:voiceName/text": "Get voice reference text",
    },
  });
});

// Get podcasts list endpoint
app.get("/podcasts", (req, res) => {
  try {
    const podcasts = getPodcastsList();
    res.json({ podcasts });
  } catch (error) {
    console.error("Error getting podcasts list:", error);
    res.status(500).json({
      error: "Error getting podcasts list",
      details: error instanceof Error ? error.message : "Unknown error",
    });
  }
});

// Force download files endpoint
app.get("/download/:podcastName/:fileName", (req, res) => {
  try {
    const { podcastName, fileName } = req.params;
    const filePath = join("../output", podcastName, fileName);

    if (!existsSync(filePath)) {
      return res.status(404).json({
        error: "File not found",
      });
    }

    // Set headers for forced download
    res.setHeader("Content-Disposition", `attachment; filename="${fileName}"`);
    res.setHeader("Content-Type", "audio/wav");

    // Send file
    res.sendFile(join(process.cwd(), filePath));
  } catch (error) {
    console.error("Error downloading file:", error);
    res.status(500).json({
      error: "Server error downloading file",
    });
  }
});

// Get system prompts list endpoint
app.get("/prompts", (req, res) => {
  try {
    const prompts = getPromptsList();
    res.json({ prompts });
  } catch (error) {
    console.error("Error getting prompts list:", error);
    res.status(500).json({
      error: "Error getting prompts list",
      details: error instanceof Error ? error.message : "Unknown error",
    });
  }
});

// Get specific prompt content endpoint
app.get("/prompts/:name", (req, res) => {
  try {
    const { name } = req.params;
    const promptData = getPromptContent(name);
    res.json(promptData);
  } catch (error) {
    if (error instanceof Error && error.message === "Prompt not found") {
      return res.status(404).json({
        error: "Prompt not found",
      });
    }
    
    console.error("Error getting prompt:", error);
    res.status(500).json({
      error: "Error getting prompt",
      details: error instanceof Error ? error.message : "Unknown error",
    });
  }
});

// Save prompt content endpoint
app.put("/prompts/:name", (req, res) => {
  try {
    const { name } = req.params;
    const { content } = req.body;

    if (!content) {
      return res.status(400).json({
        error: "Content is required",
      });
    }

    const result = savePromptContent(name, content);
    res.json(result);
  } catch (error) {
    console.error("Error saving prompt:", error);
    res.status(500).json({
      error: "Error saving prompt",
      details: error instanceof Error ? error.message : "Unknown error",
    });
  }
});

// Delete prompt endpoint
app.delete("/prompts/:name", (req, res) => {
  try {
    const { name } = req.params;
    const result = deletePrompt(name);
    res.json(result);
  } catch (error) {
    if (error instanceof Error && error.message === "Prompt not found") {
      return res.status(404).json({
        error: "Prompt not found",
      });
    }
    
    console.error("Error deleting prompt:", error);
    res.status(500).json({
      error: "Error deleting prompt",
      details: error instanceof Error ? error.message : "Unknown error",
    });
  }
});

// ===== LOGS ENDPOINTS =====

// Get log files list endpoint
app.get("/logs", (req, res) => {
  try {
    console.log("Getting log files list...");
    const logs = getLogsList();
    console.log(`Found log files: ${logs.length}`);
    res.json({ logs });
  } catch (error) {
    console.error("Error getting logs list:", error);
    res.status(500).json({
      error: "Error getting logs list",
      details: error instanceof Error ? error.message : "Unknown error",
    });
  }
});

// Get log file content endpoint
app.get("/logs/:filename", (req, res) => {
  try {
    const { filename } = req.params;
    const { lines } = req.query; // Parameter to limit number of lines

    // Check path security
    if (
      filename.includes("..") ||
      filename.includes("/") ||
      filename.includes("\\")
    ) {
      return res.status(400).json({
        error: "Invalid filename",
      });
    }

    const logPath = join("../logs", filename);

    if (!existsSync(logPath)) {
      return res.status(404).json({
        error: "Log file not found",
      });
    }

    let content = readFileSync(logPath, "utf-8");

    // If line limit requested, take last N lines
    if (lines && !isNaN(Number(lines))) {
      const linesArray = content.split("\n");
      const limitedLines = linesArray.slice(-Number(lines));
      content = limitedLines.join("\n");
    }

    res.json({
      filename,
      content,
      size: content.length,
    });
  } catch (error) {
    console.error("Error getting log file:", error);
    res.status(500).json({
      error: "Error getting log file",
      details: error instanceof Error ? error.message : "Unknown error",
    });
  }
});

// ===== VOICE MANAGEMENT ENDPOINTS =====

// Setup multer for file uploads
const upload = multer({
  dest: "../fs-python/voices/",
  limits: {
    fileSize: 200 * 1024 * 1024, // 200MB max
  },
  fileFilter: (req, file, cb) => {
    if (
      file.mimetype === "audio/wav" ||
      file.originalname.toLowerCase().endsWith(".wav")
    ) {
      cb(null, true);
    } else {
      cb(new Error("Only WAV files are allowed"));
    }
  },
});

// Get voices list endpoint
app.get("/api/voices", (req, res) => {
  try {
    const voices = getVoicesList();
    res.json({ voices });
  } catch (error) {
    console.error("Error getting voices list:", error);
    res.status(500).json({
      error: "Error getting voices list",
      details: error instanceof Error ? error.message : "Unknown error",
    });
  }
});

// Create new voice endpoint
app.post("/api/voices/create", upload.single("audioFile"), async (req, res) => {
  try {
    const { name, referenceText } = req.body;
    const audioFile = req.file;

    if (!name || !referenceText || !audioFile) {
      return res.status(400).json({
        error: "All fields required: name, referenceText, audioFile",
      });
    }

    console.log(`Creating voice: ${name}`);

    ensureVoicesDirectory();

    // Check if voice with this name already exists
    if (voiceExists(name)) {
      // Remove temporary file
      unlinkSync(audioFile.path);
      return res.status(409).json({
        error: "Voice with this name already exists",
      });
    }

    try {
      const { finalAudioPath, finalTextPath, finalTokensPath } = saveVoiceFiles(
        name,
        audioFile.path,
        referenceText
      );

      // Move audio file
      const fs = await import("fs/promises");
      await fs.rename(audioFile.path, finalAudioPath);

      // Create voice reference
      console.log("Creating voice reference...");
      const referenceResult = await createVoiceReference(
        name,
        finalAudioPath,
        finalTokensPath
      );

      if (referenceResult.success) {
        console.log(`Voice ${name} successfully created`);
        res.json({
          success: true,
          message: `Voice "${name}" successfully created`,
          voice: {
            name,
            hasAudio: true,
            hasTokens: true,
            hasText: true,
            textPreview: referenceText.substring(0, 100).trim(),
          },
        });
      } else {
        console.error(`Error creating reference for ${name}:`, referenceResult.error);

        // Clean up partially created files
        try {
          if (existsSync(finalAudioPath)) unlinkSync(finalAudioPath);
          if (existsSync(finalTextPath)) unlinkSync(finalTextPath);
          if (existsSync(finalTokensPath)) unlinkSync(finalTokensPath);
        } catch (cleanupError) {
          console.error("Error cleaning up files:", cleanupError);
        }

        res.status(500).json({
          error: "Error creating voice reference",
          details: referenceResult.error,
        });
      }
    } catch (fileError) {
      // Remove temporary file on error
      try {
        unlinkSync(audioFile.path);
      } catch (unlinkError) {
        console.error("Error removing temporary file:", unlinkError);
      }

      throw fileError;
    }
  } catch (error) {
    console.error("Error creating voice:", error);
    res.status(500).json({
      error: "Error creating voice",
      details: error instanceof Error ? error.message : "Unknown error",
    });
  }
});

// Delete voice endpoint
app.delete("/api/voices/:voiceName", (req, res) => {
  try {
    const { voiceName } = req.params;
    const result = deleteVoice(voiceName);

    if (result.deletedFiles === 0) {
      return res.status(404).json({
        error: "Voice not found",
      });
    }

    res.json({
      success: true,
      message: `Voice "${voiceName}" deleted (deleted files: ${result.deletedFiles})`,
    });
  } catch (error) {
    console.error("Error deleting voice:", error);
    res.status(500).json({
      error: "Error deleting voice",
      details: error instanceof Error ? error.message : "Unknown error",
    });
  }
});

// Test voice endpoint
app.post("/api/voices/test", async (req, res) => {
  try {
    const { voiceName, text } = req.body;

    if (!voiceName || !text) {
      return res.status(400).json({
        error: "Required fields: voiceName, text",
      });
    }

    const result = await testVoice(voiceName, text);

    if (result.success && result.audioPath) {
      console.log(`Test audio created: ${result.audioPath}`);

      // Send audio file
      res.setHeader("Content-Type", "audio/wav");
      res.sendFile(join(process.cwd(), result.audioPath), (err) => {
        if (err) {
          console.error("Error sending file:", err);
        }

        // Remove temporary file
        try {
          unlinkSync(result.audioPath!);
        } catch (cleanupError) {
          console.error("Error removing temporary file:", cleanupError);
        }
      });
    } else {
      res.status(500).json({
        error: result.error || "Error generating test audio",
      });
    }
  } catch (error) {
    console.error("Error testing voice:", error);
    res.status(500).json({
      error: "Error testing voice",
      details: error instanceof Error ? error.message : "Unknown error",
    });
  }
});

// Download voice endpoint
app.get("/api/voices/download/:voiceName", (req, res) => {
  try {
    const { voiceName } = req.params;

    console.log(`Downloading voice: ${voiceName}`);

    const audioPath = getVoiceAudioPath(voiceName);

    if (!audioPath) {
      return res.status(404).json({
        error: "Voice audio file not found",
      });
    }

    // Set download headers
    res.setHeader(
      "Content-Disposition",
      `attachment; filename="${voiceName}.wav"`
    );
    res.setHeader("Content-Type", "audio/wav");

    // Send file
    res.sendFile(join(process.cwd(), audioPath));
  } catch (error) {
    console.error("Error downloading voice:", error);
    res.status(500).json({
      error: "Error downloading voice",
      details: error instanceof Error ? error.message : "Unknown error",
    });
  }
});

// Get voice reference text endpoint
app.get("/api/voices/:voiceName/text", (req, res) => {
  try {
    const { voiceName } = req.params;

    console.log(`Getting reference text for voice: ${voiceName}`);

    const referenceText = getVoiceReferenceText(voiceName);

    if (!referenceText) {
      return res.status(404).json({
        error: "Voice reference text not found",
      });
    }

    res.json({
      voiceName,
      referenceText
    });
  } catch (error) {
    console.error("Error getting voice reference text:", error);
    res.status(500).json({
      error: "Error getting voice reference text",
      details: error instanceof Error ? error.message : "Unknown error",
    });
  }
});

// ===== ENV CONFIGURATION ENDPOINTS =====

// Get .env file content endpoint
app.get("/api/env", (req, res) => {
  try {
    const envPath = "../.env";
    
    if (!existsSync(envPath)) {
      // Return empty content if .env doesn't exist
      return res.json({
        content: "",
        exists: false
      });
    }

    const content = readFileSync(envPath, "utf-8");
    console.log("Retrieved .env file content");
    
    res.json({
      content,
      exists: true
    });
  } catch (error) {
    console.error("Error reading .env file:", error);
    res.status(500).json({
      error: "Error reading .env file",
      details: error instanceof Error ? error.message : "Unknown error",
    });
  }
});

// Save .env file content endpoint
app.put("/api/env", (req, res) => {
  try {
    const { content } = req.body;

    if (typeof content !== "string") {
      return res.status(400).json({
        error: "Content must be a string",
      });
    }

    const envPath = "../.env";
    
    // Create backup of existing .env file
    if (existsSync(envPath)) {
      const backupPath = `../.env.backup.${Date.now()}`;
      const existingContent = readFileSync(envPath, "utf-8");
      writeFileSync(backupPath, existingContent, "utf-8");
      console.log(`Created backup: ${backupPath}`);
    }

    // Write new content
    writeFileSync(envPath, content, "utf-8");
    console.log("Updated .env file successfully");
    
    res.json({
      success: true,
      message: ".env file updated successfully"
    });
  } catch (error) {
    console.error("Error writing .env file:", error);
    res.status(500).json({
      error: "Error writing .env file",
      details: error instanceof Error ? error.message : "Unknown error",
    });
  }
});

// Error handling
app.use(
  (
    error: Error,
    req: express.Request,
    res: express.Response,
    next: express.NextFunction
  ) => {
    console.error("Unhandled error:", error);
    res.status(500).json({
      error: "Internal server error",
      details: error instanceof Error ? error.message : "Unknown error",
    });
  }
);

// Helper function to get error suggestions
function getSuggestionForError(error?: string): string | undefined {
  if (!error) return undefined;
  
  if (error.includes("401")) return "Check API key";
  if (error.includes("429")) return "Rate limit exceeded, try again later";
  if (error.includes("403")) return "Access forbidden, check API settings";
  if (error.includes("HTTP error")) return "Check internet connection and try again";
  
  return undefined;
}

// Start server
server.listen(PORT, () => {
  console.log(`🚀 Podcast server started on port ${PORT}`);
  console.log(`📊 Available endpoints:`);
  console.log(`   GET  http://localhost:${PORT}/status - check status`);
  console.log(`   POST http://localhost:${PORT}/generate-script - generate script`);
  console.log(`   POST http://localhost:${PORT}/generate-podcast - generate podcast`);
  console.log(`   WS   ws://localhost:${PORT} - real-time progress`);
});

export default app; 