import { writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { Message } from "./types.js";
import { getVoiceCommandArgs, getVoiceSettings, clearVoiceConfigCache } from "./voiceConfig.js";
import { WorkerManager } from "./workerManager.js";
import chalk from "chalk";

// Global WorkerManager instance
let workerManager: WorkerManager | null = null;

// Get configuration from environment variables
const CONCURRENT_REQUESTS = parseInt(process.env.CONCURRENT_REQUESTS || "2");
const DEVICE = process.env.AUDIO_DEVICE || "mps";
const MODEL_VERSION = process.env.MODEL_VERSION || "1.5";
const MODEL_PATH = process.env.MODEL_PATH;
const SEMANTIC_TOKEN_CACHE = process.env.SEMANTIC_TOKEN_CACHE !== "false"; // Default to true

async function initializeWorkerManager(): Promise<WorkerManager> {
  if (!workerManager) {
    console.log(chalk.blue(`🚀 Initializing WorkerManager with ${CONCURRENT_REQUESTS} workers`));
    console.log(chalk.gray(`   Semantic token cache: ${SEMANTIC_TOKEN_CACHE ? "enabled" : "disabled"}`));
    
    workerManager = new WorkerManager(CONCURRENT_REQUESTS, DEVICE, MODEL_VERSION, MODEL_PATH, SEMANTIC_TOKEN_CACHE);
    
    const success = await workerManager.initialize();
    if (!success) {
      throw new Error("Failed to initialize WorkerManager");
    }
    
    // Setup graceful shutdown
    process.on('SIGINT', () => {
      console.log(chalk.yellow('\n🛑 Received SIGINT, shutting down WorkerManager...'));
      workerManager?.shutdown();
      setTimeout(() => process.exit(0), 7000);
    });
    
    process.on('SIGTERM', () => {
      console.log(chalk.yellow('\n🛑 Received SIGTERM, shutting down WorkerManager...'));
      workerManager?.shutdown();
      setTimeout(() => process.exit(0), 7000);
    });
  }
  
  return workerManager;
}

export async function generateAudio(
  text: string,
  speaker: string,
  id: number,
  streamIndex: number,
  outputDir: string
): Promise<string | null> {
  try {
    // Clear cache to ensure fresh config loading
    clearVoiceConfigCache();
    
    const filePath = join(outputDir, `output_${id}.wav`);
    const absoluteFilePath = resolve(process.cwd(), filePath);

    const voiceSettings = getVoiceSettings(speaker);

    console.log(
      chalk.gray(`Generating audio for message ${id} using WorkerManager`)
    );
    console.log(
      chalk.gray(
        `Voice settings: speed=${voiceSettings.speed}, volume=${voiceSettings.volume}dB, pitch=${voiceSettings.pitch}, checkpointPath=${voiceSettings.checkpointPath || 'default'}`
      )
    );

    // Initialize WorkerManager if not already done
    const manager = await initializeWorkerManager();
    
    // Check WorkerManager status
    const status = manager.getStatus();
    console.log(chalk.gray(
      `WorkerManager status: ${status.readyWorkers}/${status.totalWorkers} ready, queue: ${status.queueSize}`
    ));
    
    // Generate audio using WorkerManager
    const result = await manager.generateAudio(
      text,
      speaker,
      id,
      absoluteFilePath,
      voiceSettings
    );
    
    if (result.success && result.output_path) {
      if (existsSync(result.output_path)) {
        console.log(chalk.green(
          `Generated audio for message ${id} in ${result.generation_time?.toFixed(2)}s (${result.file_size_kb?.toFixed(1)}KB)`
        ));
        return filePath;
      } else {
        console.error(`❌ Audio file not created for message ${id}: ${result.output_path}`);
        return null;
      }
    } else {
      console.error(`❌ WorkerManager failed for message ${id}: ${result.error}`);
      return null;
    }
    
  } catch (error) {
    console.error(`❌ Error generating audio for message ${id}: ${error}`);
    console.error(`   Text: "${text.substring(0, 100)}${text.length > 100 ? '...' : ''}"`);
    console.error(`   Speaker: ${speaker}`);
    console.error(`   Output directory: ${outputDir}`);
    console.error(`   Stream index: ${streamIndex}`);
    
    if (error instanceof Error) {
      console.error(`   Error stack: ${error.stack}`);
    }
    
    return null;
  }
}

export function createMessageChunks(
  messages: Message[],
  chunkSize: number
): Message[][] {
  // For WorkerManager, we don't need to create chunks based on chunkSize
  // Instead, we'll distribute evenly across workers
  const numWorkers = CONCURRENT_REQUESTS;
  const chunks: Message[][] = Array.from({ length: numWorkers }, () => []);
  
  // Distribute messages round-robin across workers
  messages.forEach((message, index) => {
    const workerIndex = index % numWorkers;
    chunks[workerIndex].push(message);
  });
  
  // Filter out empty chunks
  return chunks.filter(chunk => chunk.length > 0);
}

export async function processMessageChunks(
  chunks: Message[][],
  outputDir: string
): Promise<string[]> {
  const generatedFiles: string[] = [];
  const totalMessages = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  let totalProcessed = 0;

  console.log(`🎬 Processing ${totalMessages} messages using WorkerManager`);

  // Initialize WorkerManager
  const manager = await initializeWorkerManager();
  
  // Process all messages concurrently using WorkerManager
  const allMessages = chunks.flat();
  
  // Create all generation promises
  const generationPromises = allMessages.map(async (msg, index) => {
    try {
      const filePath = await generateAudio(msg.text, msg.speaker, msg.id, index + 1, outputDir);
      
      totalProcessed++;
      
      if (filePath) {
        console.log(
          `✅ Processed message ${totalProcessed} of ${totalMessages}: ${msg.text.substring(0, 50)}...`
        );
        // Output progress in format that server.ts parses
        console.log(`Processing ${totalProcessed} of ${totalMessages}`);
        return filePath;
      } else {
        console.log(`❌ Error processing message ${totalProcessed} of ${totalMessages}`);
        console.log(`   Message ID: ${msg.id}`);
        console.log(`   Speaker: ${msg.speaker}`);
        console.log(`   Text: "${msg.text.substring(0, 100)}${msg.text.length > 100 ? '...' : ''}"`);
        console.log(`   Text length: ${msg.text.length} characters`);
        console.log(`   Possible causes:`);
        console.log(`     - Voice model issues for ${msg.speaker}`);
        console.log(`     - Processing timeout (>5 minutes)`);
        console.log(`     - Memory or GPU issues`);
        console.log(`     - File system issues in output directory`);
        console.log(`     - Incorrect text for TTS`);
        
        // Output progress in format that server.ts parses
        console.log(`Processing ${totalProcessed} of ${totalMessages}`);
        return null;
      }
    } catch (error) {
      totalProcessed++;
      console.error(`❌ Exception processing message ${msg.id}: ${error}`);
      // Output progress in format that server.ts parses
      console.log(`Processing ${totalProcessed} of ${totalMessages}`);
      return null;
    }
  });

  // Wait for all generations to complete
  const results = await Promise.all(generationPromises);
  
  // Collect successful files
  results.forEach(filePath => {
    if (filePath) {
      generatedFiles.push(filePath);
    }
  });

  console.log(
    `🎉 Processing completed. Successfully: ${generatedFiles.length} of ${totalMessages}`
  );
  
  // Print WorkerManager statistics
  const finalStatus = manager.getStatus();
  console.log(chalk.blue(
    `📊 WorkerManager final status: ${finalStatus.readyWorkers}/${finalStatus.totalWorkers} ready, queue: ${finalStatus.queueSize}`
  ));
  
  return generatedFiles;
}

// Health check function for WorkerManager
export async function healthCheck(): Promise<boolean> {
  try {
    if (!workerManager) {
      return false;
    }
    return await workerManager.healthCheck();
  } catch (error) {
    console.error(`❌ WorkerManager health check failed: ${error}`);
    return false;
  }
}

// Get WorkerManager status
export function getWorkerStatus(): any {
  if (!workerManager) {
    return { status: "not_initialized" };
  }
  return workerManager.getStatus();
}

// Shutdown WorkerManager (for testing or manual shutdown)
export function shutdownWorkerManager(): void {
  if (workerManager) {
    workerManager.shutdown();
    workerManager = null;
  }
}
