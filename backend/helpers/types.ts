export interface Message {
  id: number;
  speaker: string;
  text: string;
}

export interface ConversationData {
  podcast_name: string;
  conversation: Message[];
  filename: string;
}

// Alternative types for different validation levels
export interface BasicMessage {
  id?: number;
  speaker?: string;
  text?: string;
}

export interface BasicConversationData {
  podcast_name?: string;
  conversation?: BasicMessage[];
  filename?: string;
}

// Type for message validation
export interface MessageResult {
  success: boolean;
  error?: string;
  data?: Message;
}

// Type for processing result
export interface ProcessingResult {
  success: boolean;
  error?: string;
  files?: string[];
} 