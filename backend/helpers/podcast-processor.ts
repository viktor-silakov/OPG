import { spawn } from "child_process";
import { readFileSync } from "fs";
import { ConversationData } from "./types.js";
import { sendProgressUpdate } from "./websocket-manager.js";

export interface PodcastGenerationRequest {
  conversationData?: ConversationData;
  conversationFile?: string;
}

export async function processGeneratePodcastRequest(
  requestBody: PodcastGenerationRequest
): Promise<{
  success: boolean;
  conversation?: ConversationData;
  jobId?: string;
  messageCount?: number;
  error?: string;
}> {
  const { conversationData, conversationFile } = requestBody;
  const jobId = Date.now().toString();

  console.log("Received podcast generation request:");
  console.log("- conversationData present:", !!conversationData);
  console.log("- conversationFile:", conversationFile);

  let conversation: ConversationData;

  if (conversationData) {
    console.log("Using data from request");
    console.log("- Podcast name:", conversationData.podcast_name);
    console.log("- Number of replies:", conversationData.conversation?.length);
    conversation = conversationData;
  } else if (conversationFile) {
    console.log("Reading from file:", conversationFile);
    try {
      const fileContent = readFileSync(conversationFile, "utf-8");
      conversation = JSON.parse(fileContent);
    } catch (err) {
      return {
        success: false,
        error: `File not found: ${conversationFile}`,
      };
    }
  } else {
    console.log("Using default conversation.json");
    try {
      const fileContent = readFileSync("./conversation.json", "utf-8");
      conversation = JSON.parse(fileContent);
    } catch (err) {
      return {
        success: false,
        error: "conversation.json file not found. Generate script first.",
      };
    }
  }

  const messageCount = conversation.conversation.length;

  return {
    success: true,
    conversation,
    jobId,
    messageCount,
  };
}

export function startBackgroundPodcastGeneration(
  conversation: ConversationData,
  jobId: string,
  messageCount: number
) {
  setImmediate(async () => {
    try {
      console.log(`Starting podcast generation for job ${jobId}`);
      sendProgressUpdate(jobId, 0, messageCount, "Starting generation...");

      // Modified spawn to capture progress
      const child = spawn("tsx", ["process.ts"], {
        stdio: ["pipe", "pipe", "pipe"],
        env: {
          ...process.env,
          JOB_ID: jobId,
          CONVERSATION_DATA: JSON.stringify(conversation),
        },
      });

      let currentProgress = 0;

      child.stdout.on("data", (data) => {
        const output = data.toString();
        console.log("Process output:", output);

        // Parse progress from output
        const progressMatch = output.match(/Processing (\d+) of (\d+)/);
        if (progressMatch) {
          const current = parseInt(progressMatch[1]);
          const total = parseInt(progressMatch[2]);
          currentProgress = current;
          sendProgressUpdate(
            jobId,
            current,
            total,
            `Processing ${current} of ${total} replies`
          );
        }
      });

      child.stderr.on("data", (data) => {
        console.error("Process error:", data.toString());
      });

      child.on("close", (code) => {
        if (code === 0) {
          sendProgressUpdate(
            jobId,
            messageCount,
            messageCount,
            "Generation completed!"
          );
          console.log(`Job ${jobId} completed successfully`);
        } else {
          sendProgressUpdate(
            jobId,
            currentProgress,
            messageCount,
            "Generation error"
          );
          console.error(`Job ${jobId} failed with code ${code}`);
        }
      });
    } catch (error) {
      console.error(`Error in job ${jobId}:`, error);
      sendProgressUpdate(jobId, 0, messageCount, "Generation error");
    }
  });
} 