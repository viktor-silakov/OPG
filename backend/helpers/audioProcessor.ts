import { unlinkSync } from "node:fs";
import { join } from "node:path";
import ffmpeg from "fluent-ffmpeg";

export function concatenateAudioFiles(
  files: string[],
  outputPath: string,
  outputDir: string
): Promise<void> {
  return new Promise((resolve, reject) => {
    const command = ffmpeg();

    files.forEach((file) => {
      command.input(file);
    });

    command
      .on("error", function (err: Error) {
        console.error("Error concatenating audio:", err.message);
        reject(err);
      })
      .on("end", function () {
        console.log("Concatenated audio created in:", outputPath);
        resolve();
      })
      .mergeToFile(outputPath, outputDir);
  });
}

export function cleanupIntermediateFiles(files: string[]): void {
  files.forEach((file) => {
    try {
      unlinkSync(file);
      // console.log(`Deleted intermediate file: ${file}`);
    } catch (error) {
      console.error(`Error deleting file ${file}:`, error);
    }
  });
  console.log("All intermediate files cleaned up.");
}

export async function processAudioFiles(
  files: string[],
  outputDir: string,
  filename: string
): Promise<void> {
  if (files.length === 0) {
    console.log("No audio files to process");
    return;
  }

  const finalOutputPath = join(outputDir, filename);
  
  try {
    await concatenateAudioFiles(files, finalOutputPath, outputDir);
    cleanupIntermediateFiles(files);
  } catch (error) {
    console.error("Error processing audio files:", error);
    throw error;
  }
} 