const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:3000'

export interface ConversationData {
  podcast_name: string
  filename: string
  conversation: {
    id: number
    speaker: string
    text: string
  }[]
}

export interface GenerateScriptRequest {
  userPrompt: string
  systemPrompt?: string
}

export interface GenerateScriptResponse {
  success: boolean
  message: string
  filename: string
  conversation: ConversationData
  messageCount: number
}

export interface GeneratePodcastRequest {
  conversationData?: ConversationData
  conversationFile?: string
}

export interface GeneratePodcastResponse {
  success: boolean
  message: string
  podcast_name: string
  filename: string
  messageCount: number
  status: string
  jobId: string
}

export interface StatusResponse {
  status: string
  message: string
  endpoints: Record<string, string>
}

export interface ProgressUpdate {
  type: 'progress'
  jobId: string
  current: number
  total: number
  message?: string
}

export interface PodcastFile {
  filename: string
  url: string
  size: number
  created: string
  modified: string
}

export interface PodcastsResponse {
  podcasts: PodcastFile[]
}

export interface PromptInfo {
  name: string
  displayName: string
  description: string
  filename: string
  size: number
  modified: string
}

export interface PromptsResponse {
  prompts: PromptInfo[]
}

export interface PromptContentResponse {
  name: string
  content: string
}

// New interfaces for prompts management
export interface SavePromptRequest {
  name: string
  content: string
}

export interface SavePromptResponse {
  success: boolean
  message: string
  name: string
}

export interface DeletePromptResponse {
  success: boolean
  message: string
}

// Voice management interfaces
export interface Voice {
  name: string
  hasAudio: boolean
  hasTokens: boolean
  hasText: boolean
  audioSize?: string
  textPreview?: string
  createdAt?: string
}

export interface VoicesResponse {
  voices: Voice[]
}

export interface CreateVoiceRequest {
  name: string
  audioFile: File
  referenceText: string
}

export interface CreateVoiceResponse {
  success: boolean
  message: string
  voice: Voice
}

export interface DeleteVoiceResponse {
  success: boolean
  message: string
}

export interface TestVoiceRequest {
  voiceName: string
  text: string
}

export interface VoiceReferenceTextResponse {
  voiceName: string
  referenceText: string
}

// Env configuration interfaces
export interface EnvContentResponse {
  content: string
  exists: boolean
}

export interface UpdateEnvRequest {
  content: string
}

export interface UpdateEnvResponse {
  success: boolean
  message: string
}

class ApiClient {
  private baseUrl: string
  private wsUrl: string

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl
    this.wsUrl = baseUrl.replace('http', 'ws')
  }

  get apiBaseUrl(): string {
    return this.baseUrl
  }

  async generateScript(request: GenerateScriptRequest): Promise<GenerateScriptResponse> {
    console.log('🚀 Sending script generation request:', {
      userPromptLength: request.userPrompt?.length || 0,
      hasSystemPrompt: !!request.systemPrompt,
      systemPromptLength: request.systemPrompt?.length || 0
    });

    try {
      const response = await fetch(`${this.baseUrl}/generate-script`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
      })

      if (!response.ok) {
        let errorData;
        try {
          errorData = await response.json();
        } catch {
          const errorText = await response.text();
          console.error('❌ Script generation error (response text):', {
            status: response.status,
            statusText: response.statusText,
            errorText: errorText,
            headers: Object.fromEntries(response.headers.entries())
          });
          throw new Error(`HTTP ${response.status}: ${errorText}`);
        }

        console.error('❌ Script generation error:', {
          status: response.status,
          statusText: response.statusText,
          error: errorData.error,
          details: errorData.details,
          suggestion: errorData.suggestion,
          rawResponse: errorData.rawResponse,
          timestamp: errorData.timestamp,
          requestInfo: errorData.requestInfo
        });

        if (errorData.rawResponse) {
          console.group('🤖 Raw LLM response:');
          console.log(errorData.rawResponse);
          console.groupEnd();
        }

        if (errorData.details) {
          console.group('🔍 Error details:');
          console.log('Main error:', errorData.details.originalError || errorData.details.message);
          if (errorData.details.fallbackError) {
            console.log('Fallback error:', errorData.details.fallbackError);
          }
          if (errorData.details.textLength) {
            console.log('Text length:', errorData.details.textLength);
          }
          if (errorData.details.firstChars) {
            console.log('First chars:', errorData.details.firstChars);
          }
          if (errorData.details.lastChars) {
            console.log('Last chars:', errorData.details.lastChars);
          }
          if (errorData.details.stack) {
            console.log('Stack trace:', errorData.details.stack);
          }
          console.groupEnd();
        }

        throw new Error(errorData.error || `HTTP ${response.status}: ${response.statusText}`);
      }

      const result = await response.json();
      console.log('✅ Script generated successfully:', {
        filename: result.filename,
        messageCount: result.messageCount,
        podcastName: result.conversation?.podcast_name
      });

      return result;
    } catch (error) {
      console.error('💥 Critical error during script generation:', error);
      throw error;
    }
  }

  async generatePodcast(request: GeneratePodcastRequest): Promise<GeneratePodcastResponse> {
    console.log('🎙️ Sending podcast generation request:', {
      hasConversationData: !!request.conversationData,
      hasConversationFile: !!request.conversationFile,
      messageCount: request.conversationData?.conversation?.length || 0
    });

    try {
      const response = await fetch(`${this.baseUrl}/generate-podcast`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
      })

      if (!response.ok) {
        let errorData;
        try {
          errorData = await response.json();
        } catch {
          const errorText = await response.text();
          console.error('❌ Podcast generation error (response text):', {
            status: response.status,
            statusText: response.statusText,
            errorText: errorText,
            headers: Object.fromEntries(response.headers.entries())
          });
          throw new Error(`HTTP ${response.status}: ${errorText}`);
        }

        console.error('❌ Podcast generation error:', {
          status: response.status,
          statusText: response.statusText,
          error: errorData.error || errorData.message,
          details: errorData.details,
          timestamp: errorData.timestamp
        });

        throw new Error(errorData.error || errorData.message || `HTTP ${response.status}: ${response.statusText}`);
      }

      const result = await response.json();
      console.log('✅ Podcast generation started:', {
        jobId: result.jobId,
        podcastName: result.podcast_name,
        messageCount: result.messageCount,
        status: result.status
      });

      return result;
    } catch (error) {
      console.error('💥 Critical error during podcast generation:', error);
      throw error;
    }
  }

  async getStatus(): Promise<StatusResponse> {
    const response = await fetch(`${this.baseUrl}/status`)

    if (!response.ok) {
      const error = await response.text()
      throw new Error(`HTTP ${response.status}: ${error}`)
    }

    return response.json()
  }

  async getPodcasts(): Promise<PodcastsResponse> {
    const response = await fetch(`${this.baseUrl}/podcasts`)

    if (!response.ok) {
      const error = await response.text()
      throw new Error(`HTTP ${response.status}: ${error}`)
    }

    return response.json()
  }

  async getPrompts(): Promise<PromptsResponse> {
    const response = await fetch(`${this.baseUrl}/prompts`)

    if (!response.ok) {
      const error = await response.text()
      throw new Error(`HTTP ${response.status}: ${error}`)
    }

    return response.json()
  }

  async getPromptContent(name: string): Promise<PromptContentResponse> {
    const response = await fetch(`${this.baseUrl}/prompts/${name}`)

    if (!response.ok) {
      const error = await response.text()
      throw new Error(`HTTP ${response.status}: ${error}`)
    }

    return response.json()
  }

  // Voice management methods
  async getVoices(): Promise<VoicesResponse> {
    const response = await fetch(`${this.baseUrl}/api/voices`)

    if (!response.ok) {
      const error = await response.text()
      throw new Error(`HTTP ${response.status}: ${error}`)
    }

    return response.json()
  }

  async createVoice(request: CreateVoiceRequest): Promise<CreateVoiceResponse> {
    const formData = new FormData()
    formData.append('name', request.name)
    formData.append('audioFile', request.audioFile)
    formData.append('referenceText', request.referenceText)

    const response = await fetch(`${this.baseUrl}/api/voices/create`, {
      method: 'POST',
      body: formData,
    })

    if (!response.ok) {
      const error = await response.text()
      throw new Error(`HTTP ${response.status}: ${error}`)
    }

    return response.json()
  }

  async deleteVoice(voiceName: string): Promise<DeleteVoiceResponse> {
    const response = await fetch(`${this.baseUrl}/api/voices/${encodeURIComponent(voiceName)}`, {
      method: 'DELETE',
    })

    if (!response.ok) {
      const error = await response.text()
      throw new Error(`HTTP ${response.status}: ${error}`)
    }

    return response.json()
  }

  async testVoice(request: TestVoiceRequest): Promise<Blob> {
    const response = await fetch(`${this.baseUrl}/api/voices/test`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    })

    if (!response.ok) {
      const error = await response.text()
      throw new Error(`HTTP ${response.status}: ${error}`)
    }

    return response.blob()
  }

  async getVoiceReferenceText(voiceName: string): Promise<VoiceReferenceTextResponse> {
    const response = await fetch(`${this.baseUrl}/api/voices/${encodeURIComponent(voiceName)}/text`)

    if (!response.ok) {
      const error = await response.text()
      throw new Error(`HTTP ${response.status}: ${error}`)
    }

    return response.json()
  }

  // Prompts management methods
  async savePrompt(request: SavePromptRequest): Promise<SavePromptResponse> {
    const response = await fetch(`${this.baseUrl}/prompts/${encodeURIComponent(request.name)}`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ content: request.content }),
    })

    if (!response.ok) {
      const error = await response.text()
      throw new Error(`HTTP ${response.status}: ${error}`)
    }

    return response.json()
  }

  async deletePrompt(promptName: string): Promise<DeletePromptResponse> {
    const response = await fetch(`${this.baseUrl}/prompts/${encodeURIComponent(promptName)}`, {
      method: 'DELETE',
    })

    if (!response.ok) {
      const error = await response.text()
      throw new Error(`HTTP ${response.status}: ${error}`)
    }

    return response.json()
  }

  // WebSocket connection for progress updates
  connectProgress(jobId: string, onProgress: (update: ProgressUpdate) => void): WebSocket {
    const ws = new WebSocket(this.wsUrl)
    
    ws.onopen = () => {
      console.log('WebSocket connected')
      // Subscribe to specific job progress
      ws.send(JSON.stringify({
        type: 'subscribe',
        jobId
      }))
    }
    
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data) as ProgressUpdate
        if (data.type === 'progress' && data.jobId === jobId) {
          onProgress(data)
        }
      } catch (err) {
        console.error('Error parsing WebSocket message:', err)
      }
    }
    
    ws.onerror = (error) => {
      console.error('WebSocket error:', error)
    }
    
    ws.onclose = () => {
      console.log('WebSocket disconnected')
    }
    
    return ws
  }

  // Env configuration methods
  async getEnvContent(): Promise<EnvContentResponse> {
    const response = await fetch(`${this.baseUrl}/api/env`)

    if (!response.ok) {
      const error = await response.text()
      throw new Error(`HTTP ${response.status}: ${error}`)
    }

    return response.json()
  }

  async updateEnvContent(request: UpdateEnvRequest): Promise<UpdateEnvResponse> {
    const response = await fetch(`${this.baseUrl}/api/env`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    })

    if (!response.ok) {
      const error = await response.text()
      throw new Error(`HTTP ${response.status}: ${error}`)
    }

    return response.json()
  }
}

export const apiClient = new ApiClient() 