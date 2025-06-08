import { writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { Message } from "./types.js";
import { getVoiceCommandArgs, getVoiceSettings, clearVoiceConfigCache } from "./voiceConfig.js";
import chalk from "chalk";

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
    const voiceArgs = getVoiceCommandArgs(speaker);

    console.log(
      chalk.gray(`Generating audio for message ${id} using fs-python CLI`)
    );
    console.log(
      chalk.gray(
        `Voice settings: speed=${voiceSettings.speed}, volume=${voiceSettings.volume}dB, pitch=${voiceSettings.pitch}`
      )
    );

    const voiceFile = `../fs-python/voices/${speaker}.npy`;

    const checkpointPath = voiceSettings.checkpointPath 

    const args = [
      "run",
      "python",
      // "optimized_tts.py", 
      // "cli_tts.py",
      // "optimized_cli_tts.py",
      "flash_optimized_cli.py",
      `"${text.replace(/"/g, '\\"')}"`,
      "-o",
      absoluteFilePath,
      "--device",
      "mps",
      "--model-version",
      "1.5",
      "--monitor",
      ...voiceArgs,
    ];

    if (checkpointPath) {
      args.push("--model-path", checkpointPath);
      console.log(chalk.yellow(`Using checkpoint: ${checkpointPath}`));
    } else {
      console.log(chalk.gray(`No checkpoint specified for ${speaker}`));
    }

    try {
      if (existsSync(voiceFile)) {
        args.push("--voice", speaker);
        console.log(chalk.gray(`Using voice: ${speaker}`));
      }
    } catch (error) {
      console.log(`Voice file not found for ${speaker}, using default voice`);
    }

    const childProcess = spawn("poetry", args, {
      cwd: "../fs-python",
      stdio: ["pipe", "pipe", "pipe"],
      timeout: 5 * 60 * 1000,
      env: {
        ...process.env,
        // Limit multiprocessing usage in Python to prevent resource conflicts
        PYTHONPATH: process.env.PYTHONPATH || "",
        OMP_NUM_THREADS: "1",
        MKL_NUM_THREADS: "1",
        OPENBLAS_NUM_THREADS: "1",
      },
    });

    let stdout = "";
    let stderr = "";

    childProcess.stdout?.on("data", (data) => {
      stdout += data.toString();
    });

    childProcess.stderr?.on("data", (data) => {
      stderr += data.toString();
    });

    return new Promise((resolve, reject) => {
      const cleanup = () => {
        try {
          if (childProcess && !childProcess.killed) {
            childProcess.kill("SIGTERM");
            // Force termination after 3 seconds if process doesn't exit gracefully
            setTimeout(() => {
              if (!childProcess.killed) {
                childProcess.kill("SIGKILL");
              }
            }, 3000);
          }
        } catch (cleanupError) {
          console.warn(`Warning during process cleanup: ${cleanupError}`);
        }
      };

      childProcess.on("close", (code, signal) => {
        if (code === 0) {
          console.log(chalk.green(`Generated audio for message ${id}`));
          try {
            if (existsSync(filePath)) {
              resolve(filePath);
            } else {
              console.error(`❌ Audio file not created for message ${id}: ${filePath}`);
              console.error(`   Command: poetry ${args.join(' ')}`);
              console.error(`   Working directory: ../fs-python`);
              console.error(`   Expected output: ${filePath}`);
              if (stdout) console.error(`   STDOUT: ${stdout}`);
              if (stderr) console.error(`   STDERR: ${stderr}`);
              resolve(null);
            }
          } catch (error) {
            console.error(`❌ Error checking file for message ${id}: ${error}`);
            console.error(`   File path: ${filePath}`);
            resolve(null);
          }
        } else {
          if (signal !== "SIGTERM" && signal !== "SIGKILL") {
            console.error(`❌ Poetry process failed for message ${id}`);
            console.error(`   Exit code: ${code}`);
            console.error(`   Signal: ${signal}`);
            console.error(`   Command: poetry ${args.join(' ')}`);
            console.error(`   Working directory: ../fs-python`);
            console.error(`   Text length: ${text.length} characters`);
            console.error(`   Speaker: ${speaker}`);
            
            if (stderr) {
              console.error(`   STDERR:`);
              stderr.split('\n').forEach(line => {
                if (line.trim()) console.error(`     ${line}`);
              });
            }
            
            if (stdout) {
              console.error(`   STDOUT:`);
              stdout.split('\n').forEach(line => {
                if (line.trim()) console.error(`     ${line}`);
              });
            }
            
            // Analyze possible causes based on exit code
            if (code === 137) {
              console.error(`   Likely cause: Process terminated due to out of memory (OOM)`);
            } else if (code === 1) {
              console.error(`   Likely cause: General Python/CLI error`);
            } else if (code === 2) {
              console.error(`   Likely cause: Invalid command line arguments`);
            } else if (code === 126) {
              console.error(`   Likely cause: File found but cannot be executed`);
            } else if (code === 127) {
              console.error(`   Likely cause: Command not found`);
            }
          }
          resolve(null);
        }
      });

      childProcess.on("error", (error) => {
        console.error(`❌ Process error for message ${id}: ${error.message}`);
        console.error(`   Error name: ${error.name}`);
        console.error(`   Command: poetry ${args.join(' ')}`);
        console.error(`   Working directory: ../fs-python`);
        console.error(`   Text: "${text.substring(0, 100)}${text.length > 100 ? '...' : ''}"`);
        console.error(`   Speaker: ${speaker}`);
        
        // Analyze error type for better debugging
        if (error.message.includes('ENOENT')) {
          console.error(`   Cause: Poetry not found in system or incorrect path`);
        } else if (error.message.includes('EACCES')) {
          console.error(`   Cause: No permission to execute command`);
        } else if (error.message.includes('timeout')) {
          console.error(`   Cause: Execution timeout exceeded (5 minutes)`);
        }
        
        cleanup();
        resolve(null);
      });

      // Automatic cleanup on timeout
      childProcess.on("exit", cleanup);
    });
  } catch (error) {
    console.error(`❌ Initialization error for message ${id}: ${error}`);
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
  const chunks: Message[][] = [];
  for (let i = 0; i < messages.length; i += chunkSize) {
    chunks.push(messages.slice(i, i + chunkSize));
  }
  return chunks;
}

export async function processMessageChunks(
  chunks: Message[][],
  outputDir: string
): Promise<string[]> {
  const generatedFiles: string[] = [];
  let totalProcessed = 0;
  const totalMessages = chunks.reduce((sum, chunk) => sum + chunk.length, 0);

  console.log(`🎬 All messages to process: ${totalMessages}`);

  for (const chunk of chunks) {
    const results = await Promise.all(
      chunk.map((msg, index) =>
        generateAudio(msg.text, msg.speaker, msg.id, index + 1, outputDir)
      )
    );

    // Add only successfully generated files and update progress
    results.forEach((filePath, index) => {
      totalProcessed++;
      const currentMessage = chunk[index];
      
      if (filePath) {
        generatedFiles.push(filePath);
        console.log(
          `✅ Processed message ${totalProcessed} of ${totalMessages}: ${currentMessage.text.substring(0, 50)}...`
        );
      } else {
        // More detailed log of error
        console.log(
          `❌ Error processing message ${totalProcessed} of ${totalMessages}`
        );
        console.log(`   Message ID: ${currentMessage.id}`);
        console.log(`   Speaker: ${currentMessage.speaker}`);
        console.log(`   Text: "${currentMessage.text.substring(0, 100)}${currentMessage.text.length > 100 ? '...' : ''}"`);
        console.log(`   Text length: ${currentMessage.text.length} characters`);
        console.log(`   Possible causes:`);
        console.log(`     - Voice model issues for ${currentMessage.speaker}`);
        console.log(`     - Processing timeout (>5 minutes)`);
        console.log(`     - Memory or GPU issues`);
        console.log(`     - File system issues in output directory`);
        console.log(`     - Incorrect text for TTS`);
      }
      // Output progress in format that server.ts parses
      console.log(`Processing ${totalProcessed} of ${totalMessages}`);
    });
  }

  console.log(
    `🎉 Processing completed. Successfully: ${generatedFiles.length} of ${totalMessages}`
  );
  return generatedFiles;
}
