import { readFileSync, mkdirSync } from "node:fs";
import { join } from "node:path";
import { ConversationData } from "./types.js";

export function loadConversationData(filePath: string): ConversationData {
  const jsonData = JSON.parse(readFileSync(filePath, "utf-8"));
  
  // Validate structure
  validateConversationData(jsonData);
  
  return {
    podcast_name: jsonData.podcast_name,
    filename: jsonData.filename,
    conversation: jsonData.conversation
  };
}

function validateConversationData(data: any): void {
  if (!data.podcast_name || typeof data.podcast_name !== 'string') {
    throw new Error('Missing or invalid podcast_name field');
  }
  
  if (!data.filename || typeof data.filename !== 'string') {
    throw new Error('Missing or invalid filename field');
  }
  
  if (!Array.isArray(data.conversation)) {
    throw new Error('conversation field must be an array');
  }
  
  data.conversation.forEach((msg: any, index: number) => {
    if (!msg.id || typeof msg.id !== 'number') {
      throw new Error(`Message ${index + 1}: missing or invalid id field`);
    }
    
    if (!msg.speaker || typeof msg.speaker !== 'string') {
      throw new Error(`Message ${index + 1}: missing or invalid speaker field`);
    }
    
    if (!msg.text || typeof msg.text !== 'string') {
      throw new Error(`Message ${index + 1}: missing or invalid text field`);
    }
  });
}

export function createOutputDirectory(filename: string): string {
  const outputDir = join("../output", filename);
  mkdirSync(outputDir, { recursive: true });
  return outputDir;
}

export function sortFilesByID(files: string[]): string[] {
  return files.sort((a, b) => {
    const idA = parseInt(a.match(/output_(\d+)\.wav$/)![1]);
    const idB = parseInt(b.match(/output_(\d+)\.wav$/)![1]);
    return idA - idB;
  });
} 