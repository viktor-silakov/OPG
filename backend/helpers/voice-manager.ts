import { 
  existsSync, 
  mkdirSync, 
  readdirSync, 
  statSync, 
  unlinkSync, 
  writeFileSync, 
  readFileSync 
} from "fs";
import { join, extname, basename } from "path";
import { spawn } from "child_process";

const VOICES_DIR = "../fs-python/voices";

function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

export function ensureVoicesDirectory() {
  if (!existsSync(VOICES_DIR)) {
    mkdirSync(VOICES_DIR, { recursive: true });
  }
}

export function getVoicesList() {
  console.log("Getting voices list...");

  ensureVoicesDirectory();

  const files = readdirSync(VOICES_DIR);
  const voiceMap = new Map();

  // Collect voice information
  for (const file of files) {
    if (file.startsWith(".")) continue;

    const filePath = join(VOICES_DIR, file);
    const stats = statSync(filePath);
    const ext = extname(file);
    const name = basename(file, ext);

    if (!voiceMap.has(name)) {
      voiceMap.set(name, {
        name,
        hasAudio: false,
        hasTokens: false,
        hasText: false,
        audioSize: undefined,
        textPreview: undefined,
        createdAt: stats.birthtime.toISOString(),
      });
    }

    const voice = voiceMap.get(name);

    if (ext === ".wav") {
      voice.hasAudio = true;
      voice.audioSize = formatFileSize(stats.size);
    } else if (ext === ".npy") {
      voice.hasTokens = true;
    } else if (ext === ".txt") {
      voice.hasText = true;
      try {
        const content = readFileSync(filePath, "utf-8");
        voice.textPreview = content.substring(0, 100).trim();
      } catch (error) {
        console.error(`Error reading text for ${name}:`, error);
      }
    }
  }

  const voices = Array.from(voiceMap.values());
  console.log(`Found voices: ${voices.length}`);

  return voices;
}

export async function createVoiceReference(
  name: string,
  audioPath: string,
  tokensPath: string
): Promise<{ success: boolean; error?: string }> {
  return new Promise((resolve) => {
    console.log("Creating voice reference...");
    const cliProcess = spawn(
      "poetry",
      [
        "run",
        "python",
        "cli_tts.py",
        "--create-reference",
        audioPath,
        tokensPath,
      ],
      {
        cwd: "../fs-python",
        stdio: ["pipe", "pipe", "pipe"],
      }
    );

    let output = "";
    let errorOutput = "";

    cliProcess.stdout.on("data", (data) => {
      output += data.toString();
    });

    cliProcess.stderr.on("data", (data) => {
      errorOutput += data.toString();
    });

    cliProcess.on("close", (code) => {
      if (code === 0 && existsSync(tokensPath)) {
        console.log(`Voice ${name} successfully created`);
        resolve({ success: true });
      } else {
        console.error(`Error creating reference for ${name}:`, errorOutput);
        resolve({ 
          success: false, 
          error: errorOutput || "Unknown error" 
        });
      }
    });
  });
}

export function deleteVoice(voiceName: string): { deletedFiles: number } {
  console.log(`Deleting voice: ${voiceName}`);

  const audioPath = join(VOICES_DIR, `${voiceName}.wav`);
  const textPath = join(VOICES_DIR, `${voiceName}.txt`);
  const tokensPath = join(VOICES_DIR, `${voiceName}.npy`);

  let deletedFiles = 0;

  // Delete all related files
  [audioPath, textPath, tokensPath].forEach((filePath) => {
    if (existsSync(filePath)) {
      try {
        unlinkSync(filePath);
        deletedFiles++;
        console.log(`Deleted file: ${filePath}`);
      } catch (error) {
        console.error(`Error deleting file ${filePath}:`, error);
      }
    }
  });

  return { deletedFiles };
}

export async function testVoice(
  voiceName: string,
  text: string
): Promise<{ success: boolean; audioPath?: string; error?: string }> {
  return new Promise((resolve) => {
    console.log(`Testing voice: ${voiceName}`);

    const tokensPath = join(VOICES_DIR, `${voiceName}.npy`);

    if (!existsSync(tokensPath)) {
      resolve({ 
        success: false, 
        error: "Voice tokens file not found" 
      });
      return;
    }

    // Create temporary file for test audio
    const tempAudioPath = join(
      "../fs-python/output",
      `test_${voiceName}_${Date.now()}.wav`
    );

    // Run speech synthesis
    const cliArgs = [
      "run",
      "python",
      "cli_tts.py",
      text,
      "--voice",
      voiceName,
      "-o",
      tempAudioPath,
    ];

    const cliProcess = spawn("poetry", cliArgs, {
      cwd: "../fs-python",
      stdio: ["pipe", "pipe", "pipe"],
    });

    let output = "";
    let errorOutput = "";

    cliProcess.stdout.on("data", (data) => {
      output += data.toString();
    });

    cliProcess.stderr.on("data", (data) => {
      errorOutput += data.toString();
    });

    cliProcess.on("close", (code) => {
      if (code === 0 && existsSync(tempAudioPath)) {
        console.log(`Test audio created: ${tempAudioPath}`);
        resolve({ 
          success: true, 
          audioPath: tempAudioPath 
        });
      } else {
        console.error(`Error testing voice ${voiceName}:`, errorOutput);
        resolve({ 
          success: false, 
          error: errorOutput || "Unknown error" 
        });
      }
    });
  });
}

export function saveVoiceFiles(
  name: string,
  audioPath: string,
  referenceText: string
): { 
  finalAudioPath: string;
  finalTextPath: string;
  finalTokensPath: string;
} {
  const finalAudioPath = join(VOICES_DIR, `${name}.wav`);
  const finalTextPath = join(VOICES_DIR, `${name}.txt`);
  const finalTokensPath = join(VOICES_DIR, `${name}.npy`);

  // Save text
  writeFileSync(finalTextPath, referenceText, "utf-8");

  return {
    finalAudioPath,
    finalTextPath,
    finalTokensPath,
  };
}

export function voiceExists(name: string): boolean {
  const audioPath = join(VOICES_DIR, `${name}.wav`);
  const tokensPath = join(VOICES_DIR, `${name}.npy`);
  return existsSync(audioPath) || existsSync(tokensPath);
}

export function getVoiceAudioPath(voiceName: string): string | null {
  const audioPath = join(VOICES_DIR, `${voiceName}.wav`);
  return existsSync(audioPath) ? audioPath : null;
}

export function getVoiceReferenceText(voiceName: string): string | null {
  const textPath = join(VOICES_DIR, `${voiceName}.txt`);
  
  if (!existsSync(textPath)) {
    return null;
  }

  try {
    return readFileSync(textPath, "utf-8");
  } catch (error) {
    console.error(`Error reading text file for ${voiceName}:`, error);
    return null;
  }
} 