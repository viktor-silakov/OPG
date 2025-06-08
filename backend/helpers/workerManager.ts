import { spawn, ChildProcess } from "node:child_process";
import { EventEmitter } from "node:events";
import chalk from "chalk";

interface WorkerRequest {
  id: number;
  type: string;
  text?: string;
  output_path?: string;
  voice_settings?: any;
  model_version?: string;
  model_path?: string;
  use_semantic_cache?: boolean;
}

interface WorkerResponse {
  type: string;
  success?: boolean;
  worker_id: number;
  id?: number;
  output_path?: string;
  generation_time?: number;
  file_size_kb?: number;
  memory_usage?: {
    ram_mb: number;
    mps_mb: number;
  };
  error?: string;
}

class AudioWorker extends EventEmitter {
  public id: number;
  public process: ChildProcess;
  public isReady: boolean = false;
  public isBusy: boolean = false;
  public currentRequest: WorkerRequest | null = null;
  private device: string;
  private useSemanticCache: boolean;

  constructor(id: number, device: string = "mps", useSemanticCache: boolean = true) {
    super();
    this.id = id;
    this.device = device;
    this.useSemanticCache = useSemanticCache;
    
    // Spawn Python worker process
    this.process = spawn("poetry", ["run", "python", "audio_worker.py", id.toString(), device], {
      cwd: "../fs-python",
      stdio: ["pipe", "pipe", "pipe"],
      env: {
        ...process.env,
        PYTHONPATH: process.env.PYTHONPATH || "",
        OMP_NUM_THREADS: "1",
        MKL_NUM_THREADS: "1",
        OPENBLAS_NUM_THREADS: "1",
        PYTORCH_MPS_HIGH_WATERMARK_RATIO: "0.0",
        PYTORCH_MPS_ALLOCATOR_POLICY: "garbage_collection"
      }
    });

    this.setupEventHandlers();
    
    console.log(chalk.blue(`🤖 Worker ${id} started on device: ${device}`));
  }

  private setupEventHandlers() {
    this.process.stdout?.on("data", (data) => {
      const lines = data.toString().split('\n').filter((line: string) => line.trim());
      
      for (const line of lines) {
        try {
          const response: WorkerResponse = JSON.parse(line);
          this.handleResponse(response);
        } catch (e) {
          // Not JSON - probably just log output
          console.log(chalk.gray(`Worker ${this.id}: ${line}`));
        }
      }
    });

    this.process.stderr?.on("data", (data) => {
      const stderrText = data.toString();
      
      // Filter out progress bars and informational messages
      const linesToFilter = [
        // Huggingface progress bars
        /Fetching \d+ files:\s+\d+%/,
        /\d+%\|[█▏▎▍▌▋▊▉ ]*\|\s*\d+\/\d+/,
        /\d+\/\d+ \[\d+:\d+<\d+:\d+/,
        /\[\d+:\d+<\?\?\?\?/,
        /it\/s\]/,
        // Loading messages that aren't errors
        /\d+:\d+<\d+:\d+,/,
        /files:\s+\d+%/,
        // Git LFS messages
        /Git LFS/,
        /Downloading/,
        // Empty or whitespace-only lines
        /^\s*$/
      ];
      
      const lines = stderrText.split('\n');
      const filteredLines = lines.filter(line => {
        const trimmedLine = line.trim();
        if (!trimmedLine) return false;
        
        // Check if line matches any filter pattern
        return !linesToFilter.some(pattern => pattern.test(trimmedLine));
      });
      
      // Only output if there are actual error messages after filtering
      if (filteredLines.length > 0) {
        const filteredContent = filteredLines.join('\n').trim();
        if (filteredContent) {
          // Check if it looks like an actual error (contains error keywords)
          const errorKeywords = ['error', 'exception', 'traceback', 'failed', 'cannot', 'invalid', 'missing'];
          const isActualError = errorKeywords.some(keyword => 
            filteredContent.toLowerCase().includes(keyword)
          );
          
          if (isActualError) {
            console.error(chalk.red(`Worker ${this.id} stderr: ${filteredContent}`));
          } else {
            // Log as info instead of error for non-error messages
            console.log(chalk.yellow(`Worker ${this.id} info: ${filteredContent}`));
          }
        }
      }
    });

    this.process.on("close", (code) => {
      console.log(chalk.yellow(`Worker ${this.id} closed with code: ${code}`));
      this.emit("closed", this.id);
    });

    this.process.on("error", (error) => {
      console.error(chalk.red(`Worker ${this.id} error: ${error.message}`));
      this.emit("error", this.id, error);
    });
  }

  private handleResponse(response: WorkerResponse) {
    switch (response.type) {
      case "model_loaded":
        this.isReady = response.success || false;
        if (this.isReady) {
          console.log(chalk.green(`✅ Worker ${this.id}: Model loaded successfully`));
        } else {
          console.error(chalk.red(`❌ Worker ${this.id}: Failed to load model`));
        }
        this.emit("model_loaded", this.id, response.success);
        break;

      case "generation_result":
        this.isBusy = false;
        this.currentRequest = null;
        
        if (response.success) {
          console.log(chalk.green(
            `✅ Worker ${this.id}: Generated audio for request ${response.id} ` +
            `(${response.generation_time?.toFixed(2)}s, ${response.file_size_kb?.toFixed(1)}KB)`
          ));
        } else {
          console.error(chalk.red(
            `❌ Worker ${this.id}: Failed to generate audio for request ${response.id}: ${response.error}`
          ));
        }
        
        this.emit("generation_result", response);
        break;

      case "pong":
        // Health check response
        this.emit("pong", response);
        break;

      case "error":
        console.error(chalk.red(`❌ Worker ${this.id}: ${response.error}`));
        this.emit("worker_error", this.id, response.error);
        break;

      default:
        console.log(chalk.gray(`Worker ${this.id}: Unknown response type: ${response.type}`));
    }
  }

  sendRequest(request: WorkerRequest): boolean {
    if (!this.process.stdin) {
      console.error(chalk.red(`❌ Worker ${this.id}: stdin not available`));
      return false;
    }

    try {
      const requestJson = JSON.stringify(request) + '\n';
      this.process.stdin.write(requestJson);
      
      if (request.type === "generate") {
        this.isBusy = true;
        this.currentRequest = request;
      }
      
      return true;
    } catch (error) {
      console.error(chalk.red(`❌ Worker ${this.id}: Failed to send request: ${error}`));
      return false;
    }
  }

  loadModel(modelVersion: string, modelPath?: string): boolean {
    return this.sendRequest({
      id: 0,
      type: "load_model",
      model_version: modelVersion,
      model_path: modelPath,
      use_semantic_cache: this.useSemanticCache
    });
  }

  generateAudio(request: WorkerRequest): boolean {
    if (this.isBusy) {
      return false;
    }
    
    return this.sendRequest({
      ...request,
      type: "generate",
      use_semantic_cache: this.useSemanticCache
    });
  }

  ping(): boolean {
    return this.sendRequest({ id: 0, type: "ping" });
  }

  shutdown(): void {
    this.sendRequest({ id: 0, type: "shutdown" });
    
    // Force kill after 5 seconds if not closed gracefully
    setTimeout(() => {
      if (!this.process.killed) {
        this.process.kill("SIGKILL");
        console.log(chalk.yellow(`🔪 Worker ${this.id}: Force killed`));
      }
    }, 5000);
  }

  isAvailable(): boolean {
    return this.isReady && !this.isBusy;
  }
}

export class WorkerManager extends EventEmitter {
  private workers: Map<number, AudioWorker> = new Map();
  private requestQueue: WorkerRequest[] = [];
  private pendingRequests: Map<number, (result: WorkerResponse) => void> = new Map();
  private concurrentRequests: number;
  private device: string;
  private modelVersion: string;
  private modelPath?: string;
  private useSemanticCache: boolean;

  constructor(concurrentRequests: number = 2, device: string = "mps", modelVersion: string = "1.5", modelPath?: string, useSemanticCache: boolean = true) {
    super();
    this.concurrentRequests = concurrentRequests;
    this.device = device;
    this.modelVersion = modelVersion;
    this.modelPath = modelPath;
    this.useSemanticCache = useSemanticCache;
    
    console.log(chalk.blue(`🚀 WorkerManager: Initializing ${concurrentRequests} workers on ${device}`));
  }

  async initialize(): Promise<boolean> {
    console.log(chalk.blue(`🔄 WorkerManager: Starting ${this.concurrentRequests} workers...`));
    
    // Create workers
    for (let i = 0; i < this.concurrentRequests; i++) {
      const worker = new AudioWorker(i, this.device, this.useSemanticCache);
      this.workers.set(i, worker);
      
      // Set up worker event handlers
      worker.on("generation_result", (response: WorkerResponse) => {
        const callback = this.pendingRequests.get(response.id || 0);
        if (callback) {
          callback(response);
          this.pendingRequests.delete(response.id || 0);
        }
        
        // Process next request in queue
        this.processQueue();
      });

      worker.on("closed", (workerId: number) => {
        console.log(chalk.yellow(`🔄 WorkerManager: Worker ${workerId} closed, removing from pool`));
        this.workers.delete(workerId);
      });

      worker.on("error", (workerId: number, error: Error) => {
        console.error(chalk.red(`❌ WorkerManager: Worker ${workerId} error: ${error.message}`));
      });
    }

    // Wait for all workers to load models
    const modelLoadPromises = Array.from(this.workers.values()).map(worker => {
      return new Promise<boolean>((resolve) => {
        worker.once("model_loaded", (workerId: number, success: boolean) => {
          resolve(success);
        });
        
        // Start model loading
        worker.loadModel(this.modelVersion, this.modelPath);
      });
    });

    const results = await Promise.all(modelLoadPromises);
    const successCount = results.filter(r => r).length;
    
    if (successCount > 0) {
      console.log(chalk.green(`✅ WorkerManager: ${successCount}/${this.concurrentRequests} workers ready`));
      return true;
    } else {
      console.error(chalk.red(`❌ WorkerManager: No workers could load the model`));
      return false;
    }
  }

  async generateAudio(
    text: string,
    speaker: string,
    id: number,
    outputPath: string,
    voiceSettings: any
  ): Promise<WorkerResponse> {
    return new Promise((resolve, reject) => {
      const request: WorkerRequest = {
        id,
        type: "generate",
        text,
        output_path: outputPath,
        voice_settings: {
          voice: speaker,
          ...voiceSettings
        },
        use_semantic_cache: this.useSemanticCache
      };

      // Store callback for this request
      this.pendingRequests.set(id, resolve);

      // Try to find available worker immediately
      const availableWorker = this.findAvailableWorker();
      if (availableWorker) {
        availableWorker.generateAudio(request);
      } else {
        // Add to queue
        this.requestQueue.push(request);
        console.log(chalk.yellow(`📋 WorkerManager: Request ${id} queued (queue size: ${this.requestQueue.length})`));
      }

      // Set timeout for request
      setTimeout(() => {
        if (this.pendingRequests.has(id)) {
          this.pendingRequests.delete(id);
          reject(new Error(`Request ${id} timed out after 5 minutes`));
        }
      }, 5 * 60 * 1000);
    });
  }

  private findAvailableWorker(): AudioWorker | null {
    for (const worker of this.workers.values()) {
      if (worker.isAvailable()) {
        return worker;
      }
    }
    return null;
  }

  private processQueue(): void {
    if (this.requestQueue.length === 0) {
      return;
    }

    const availableWorker = this.findAvailableWorker();
    if (availableWorker) {
      const request = this.requestQueue.shift();
      if (request) {
        console.log(chalk.blue(`🔄 WorkerManager: Processing queued request ${request.id}`));
        availableWorker.generateAudio(request);
      }
    }
  }

  getStatus(): any {
    const workers = Array.from(this.workers.values()).map(worker => ({
      id: worker.id,
      ready: worker.isReady,
      busy: worker.isBusy,
      currentRequest: worker.currentRequest?.id || null
    }));

    return {
      totalWorkers: this.workers.size,
      readyWorkers: workers.filter(w => w.ready).length,
      busyWorkers: workers.filter(w => w.busy).length,
      queueSize: this.requestQueue.length,
      workers
    };
  }

  async healthCheck(): Promise<boolean> {
    const workers = Array.from(this.workers.values());
    const pings = workers.map(worker => {
      return new Promise<boolean>((resolve) => {
        const timeout = setTimeout(() => resolve(false), 3000);
        
        worker.once("pong", () => {
          clearTimeout(timeout);
          resolve(true);
        });
        
        worker.ping();
      });
    });

    const results = await Promise.all(pings);
    const healthyCount = results.filter(r => r).length;
    
    console.log(chalk.blue(`💊 WorkerManager: Health check - ${healthyCount}/${workers.length} workers healthy`));
    return healthyCount > 0;
  }

  shutdown(): void {
    console.log(chalk.yellow(`🛑 WorkerManager: Shutting down ${this.workers.size} workers...`));
    
    for (const worker of this.workers.values()) {
      worker.shutdown();
    }

    // Clear pending requests
    for (const [id, callback] of this.pendingRequests.entries()) {
      callback({
        type: "generation_result",
        success: false,
        error: "WorkerManager shutting down",
        worker_id: -1,
        id
      });
    }
    this.pendingRequests.clear();
    this.requestQueue.length = 0;

    setTimeout(() => {
      this.workers.clear();
      console.log(chalk.green(`✅ WorkerManager: Shutdown complete`));
    }, 6000);
  }
} 