import { existsSync, readFileSync } from 'fs';
import { join } from 'path';
import { fileURLToPath } from 'url';

const __dirname = fileURLToPath(new URL('.', import.meta.url));

export interface VoiceSettings {
  speed: number;
  volume: number;
  pitch: number;
  checkpointPath?: string;
}

export const DEFAULT_VOICE_SETTINGS: VoiceSettings = {
  speed: 1.0,
  volume: 0,
  pitch: 1.0,
  checkpointPath: undefined
};

// Voice configuration interface
export interface VoiceConfig {
  [voiceName: string]: VoiceSettings;
}

let cachedConfig: VoiceConfig | null = null;

function loadVoiceConfig(): VoiceConfig {
  if (cachedConfig) {
    return cachedConfig;
  }

  console.log(`Current working directory: ${process.cwd()}`);
  
  // Try multiple possible paths
  const possiblePaths = [
    join(process.cwd(), 'voice-config.json'),
  ];
  
  let configPath = '';
  let configFound = false;
  
  for (const path of possiblePaths) {
    console.log(`Checking voice config at: ${path}`);
    if (existsSync(path)) {
      configPath = path;
      configFound = true;
      console.log(`✅ Voice config file found at: ${path}`);
      break;
    } else {
      console.log(`❌ Voice config file not found at: ${path}`);
    }
  }
  
  if (!configFound) {
    console.log(`❌ Voice config file not found in any of the expected locations`);
    cachedConfig = {};
    return cachedConfig;
  }
  
  try {
    const configData = readFileSync(configPath, 'utf-8');
    const fullConfig = JSON.parse(configData);
    // Extract voices section from the config
    cachedConfig = fullConfig.voices || {};
    console.log(`Loaded ${Object.keys(cachedConfig || {}).length} voices from config`);
    console.log(`Available voices: ${Object.keys(cachedConfig || {}).join(', ')}`);
    return cachedConfig!;
  } catch (error) {
    console.warn(`Warning: Could not load voice config from ${configPath}:`, error);
  }

  // Return default settings if file not found
  cachedConfig = {};
  return cachedConfig;
}

export function getVoiceSettings(voiceName: string): VoiceSettings {
  const config = loadVoiceConfig();
  console.log(`Getting settings for voice: ${voiceName}`);
  
  if (config[voiceName]) {
    const settings = { ...DEFAULT_VOICE_SETTINGS, ...config[voiceName] };
    console.log(`Voice settings for ${voiceName}:`, settings);
    return settings;
  }

  console.log(`Voice ${voiceName} not found in config, using default settings`);
  // Get settings for specific voice or use default settings
  return DEFAULT_VOICE_SETTINGS;
}

export function getVoiceCommandArgs(voiceName: string): string[] {
  const settings = getVoiceSettings(voiceName);
  const args: string[] = [];

  // Add arguments only if they differ from default values
  if (settings.speed !== DEFAULT_VOICE_SETTINGS.speed) {
    args.push('--speed', settings.speed.toString());
  }

  if (settings.volume !== DEFAULT_VOICE_SETTINGS.volume) {
    args.push('--volume', settings.volume.toString());
  }

  if (settings.pitch !== DEFAULT_VOICE_SETTINGS.pitch) {
    args.push('--pitch', settings.pitch.toString());
  }

  return args;
}

export function clearVoiceConfigCache() {
  cachedConfig = null;
  console.log('Voice config cache cleared');
} 