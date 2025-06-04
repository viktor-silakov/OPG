import { existsSync, readFileSync } from 'fs';
import { join } from 'path';

export interface VoiceSettings {
  speed: number;
  volume: number;
  pitch: number;
  checkpointPath?: string;
}

export const DEFAULT_VOICE_SETTINGS: VoiceSettings = {
  speed: 1.0,
  volume: 0,
  pitch: 0,
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

  const configPath = join(process.cwd(), '../fs-python/voice-config.json');
  
  try {
    if (existsSync(configPath)) {
      const configData = readFileSync(configPath, 'utf-8');
      cachedConfig = JSON.parse(configData);
      return cachedConfig!;
    }
  } catch (error) {
    console.warn(`Warning: Could not load voice config from ${configPath}:`, error);
  }

  // Return default settings if file not found
  cachedConfig = {};
  return cachedConfig;
}

export function getVoiceSettings(voiceName: string): VoiceSettings {
  const config = loadVoiceConfig();
  
  if (config[voiceName]) {
    return { ...DEFAULT_VOICE_SETTINGS, ...config[voiceName] };
  }

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