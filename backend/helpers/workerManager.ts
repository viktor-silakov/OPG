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
  batch?: WorkerRequest[];
  batch_id?: number;
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
  batch_id?: number;
  processed_count?: number;
  successful_count?: number;
}

interface BatchConfig {
  enabled: boolean;
  maxSize: number;
  timeoutMs: number;
  maxTextLength: number;
}

interface QueuedRequest {
  request: WorkerRequest;
  resolve: (result: WorkerResponse) => void;
  reject: (error: any) => void;
  timestamp: number;
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
        PYTORCH_MPS_ALLOCATOR_POLICY: "garbage_collection",
        // Batching configuration
        ENABLE_BATCHING: process.env.ENABLE_BATCHING || "true",
        MAX_BATCH_SIZE: process.env.MAX_BATCH_SIZE || "4",
        BATCH_TIMEOUT_MS: process.env.BATCH_TIMEOUT_MS || "500"
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
      const filteredLines = lines.filter((line: string) => {
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

      case "batch_completed":
        this.isBusy = false;
        this.currentRequest = null;
        
        console.log(chalk.blue(
          `📦 Worker ${this.id}: Batch ${response.batch_id} completed - ` +
          `${response.successful_count}/${response.processed_count} successful`
        ));
        
        this.emit("batch_completed", response);
        break;

      case "batch_result":
        if (!response.success) {
          console.error(chalk.red(
            `❌ Worker ${this.id}: Batch ${response.batch_id} failed: ${response.error}`
          ));
        }
        this.emit("batch_result", response);
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

  generateBatch(requests: WorkerRequest[], batchId: number): boolean {
    if (this.isBusy) {
      return false;
    }
    
    // Mark worker as busy for batch processing
    this.isBusy = true;
    this.currentRequest = { id: batchId, type: "generate_batch" };
    
    return this.sendRequest({
      id: batchId,
      type: "generate_batch",
      batch: requests.map(req => ({
        ...req,
        use_semantic_cache: this.useSemanticCache
      })),
      batch_id: batchId,
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
  private requestQueue: QueuedRequest[] = [];
  private pendingRequests: Map<number, (result: WorkerResponse) => void> = new Map();
  private concurrentRequests: number;
  private device: string;
  private modelVersion: string;
  private modelPath?: string;
  private useSemanticCache: boolean;
  
  // Batching configuration
  private batchConfig: BatchConfig;
  private batchIdCounter: number = 1;
  private batchTimer: NodeJS.Timeout | null = null;

  constructor(concurrentRequests: number = 2, device: string = "mps", modelVersion: string = "1.5", modelPath?: string, useSemanticCache: boolean = true) {
    super();
    this.concurrentRequests = concurrentRequests;
    this.device = device;
    this.modelVersion = modelVersion;
    this.modelPath = modelPath;
    this.useSemanticCache = useSemanticCache;
    
    // Initialize batch configuration
    this.batchConfig = {
      enabled: process.env.ENABLE_BATCHING === "true",
      maxSize: parseInt(process.env.MAX_BATCH_SIZE || "4"),
      timeoutMs: parseInt(process.env.BATCH_TIMEOUT_MS || "500"),
      maxTextLength: parseInt(process.env.MAX_BATCH_TEXT_LENGTH || "2000")
    };
    
    console.log(chalk.blue(`🚀 WorkerManager: Initializing ${concurrentRequests} workers on ${device}`));
    if (this.batchConfig.enabled) {
      console.log(chalk.blue(
        `📦 Batching enabled: max_size=${this.batchConfig.maxSize}, ` +
        `timeout=${this.batchConfig.timeoutMs}ms, max_text=${this.batchConfig.maxTextLength}`
      ));
    }
  }

  async initialize(): Promise<boolean> {
    console.log(chalk.blue(`🔄 WorkerManager: Starting ${this.concurrentRequests} workers...`));
    
    // Create workers
    for (let i = 0; i < this.concurrentRequests; i++) {
      const worker = new AudioWorker(i, this.device, this.useSemanticCache);
      this.workers.set(i, worker);
      
      // Set up worker event handlers
      worker.on("generation_result", (response: WorkerResponse) => {
        console.log(chalk.green(`📬 Received result for request ${response.id} from worker ${response.worker_id}`));
        
        const callback = this.pendingRequests.get(response.id || 0);
        if (callback) {
          callback(response);
          this.pendingRequests.delete(response.id || 0);
          console.log(chalk.cyan(`✅ Resolved pending request ${response.id}`));
        } else {
          console.log(chalk.yellow(`⚠️ No pending callback found for request ${response.id}`));
        }
        
        // Process next request in queue
        console.log(chalk.cyan(`🔄 Triggering queue processing after result`));
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

      const queuedRequest: QueuedRequest = {
        request,
        resolve,
        reject,
        timestamp: Date.now()
      };

      // Store callback for this request
      this.pendingRequests.set(id, resolve);

      if (this.batchConfig.enabled) {
        // Add to queue for potential batching
        this.requestQueue.push(queuedRequest);
        this.processBatchQueue();
      } else {
        // Direct processing without batching
        const availableWorker = this.findAvailableWorker();
        if (availableWorker) {
          availableWorker.generateAudio(request);
        } else {
          this.requestQueue.push(queuedRequest);
          console.log(chalk.yellow(`📋 WorkerManager: Request ${id} queued (queue size: ${this.requestQueue.length})`));
        }
      }

      // Set timeout for request
      setTimeout(() => {
        if (this.pendingRequests.has(id)) {
          this.pendingRequests.delete(id);
          reject(new Error(`Request ${id} timed out after 15 minutes`));
        }
      }, 15 * 60 * 1000);
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
    console.log(chalk.cyan(`🔄 ProcessQueue called with ${this.requestQueue.length} requests`));
    
    if (this.requestQueue.length === 0) {
      console.log(chalk.cyan(`ℹ️ Queue is empty`));
      return;
    }

    const availableWorker = this.findAvailableWorker();
    if (availableWorker) {
      const queuedRequest = this.requestQueue.shift();
      if (queuedRequest) {
        console.log(chalk.blue(`🔄 WorkerManager: Processing queued request ${queuedRequest.request.id} with worker ${availableWorker.id}`));
        availableWorker.generateAudio(queuedRequest.request);
        
        // Continue processing if there are more requests
        if (this.requestQueue.length > 0) {
          console.log(chalk.cyan(`🔄 ${this.requestQueue.length} requests remaining, continuing`));
          setTimeout(() => this.processQueue(), 10);
        }
      } else {
        console.log(chalk.yellow(`⚠️ Queue had items but shift() returned undefined`));
      }
    } else {
      console.log(chalk.yellow(`⚠️ No available workers for queue processing`));
    }
  }

  private processBatchQueue(): void {
    if (!this.batchConfig.enabled || this.requestQueue.length === 0) {
      return;
    }

    console.log(chalk.cyan(`🔍 ProcessBatchQueue: ${this.requestQueue.length} requests in queue`));

    // Clear existing timer
    if (this.batchTimer) {
      clearTimeout(this.batchTimer);
      this.batchTimer = null;
    }

    // Try to form a batch immediately if we have enough requests
    if (this.requestQueue.length >= this.batchConfig.maxSize) {
      console.log(chalk.cyan(`📦 Immediate batch processing: ${this.requestQueue.length} >= ${this.batchConfig.maxSize}`));
      this.formAndProcessBatch();
      return;
    }

    // Set timer for batch timeout
    console.log(chalk.cyan(`⏰ Setting batch timer: ${this.batchConfig.timeoutMs}ms`));
    this.batchTimer = setTimeout(() => {
      console.log(chalk.cyan(`⏰ Batch timer fired, processing ${this.requestQueue.length} requests`));
      this.formAndProcessBatch();
    }, this.batchConfig.timeoutMs);
  }

  private formAndProcessBatch(): void {
    console.log(chalk.cyan(`🔄 FormAndProcessBatch called with ${this.requestQueue.length} requests`));
    
    if (this.requestQueue.length === 0) {
      console.log(chalk.cyan(`ℹ️ No requests to process, returning`));
      return;
    }

    const availableWorker = this.findAvailableWorker();
    if (!availableWorker) {
      console.log(chalk.yellow(`⚠️ No available workers, falling back to regular queue processing`));
      // Fallback: try regular queue processing instead of hanging
      this.processQueue();
      return;
    }

    console.log(chalk.cyan(`👷 Found available worker ${availableWorker.id}`));

    // Group compatible requests into batches
    const batches = this.groupCompatibleRequests();
    console.log(chalk.cyan(`📊 Grouped into ${batches.length} batches`));
    
    for (const batch of batches) {
      if (batch.length === 0) continue;
      
      console.log(chalk.cyan(`🔄 Processing batch with ${batch.length} requests`));
      
      const worker = this.findAvailableWorker();
      if (!worker) {
        console.log(chalk.yellow(`⚠️ No worker available for batch, re-queuing ${batch.length} requests`));
        // Re-queue the requests for later processing
        this.requestQueue.push(...batch);
        break;
      }

      if (batch.length === 1) {
        // Single request, process normally
        const queuedRequest = batch[0];
        console.log(chalk.blue(`🔄 WorkerManager: Processing single request ${queuedRequest.request.id}`));
        worker.generateAudio(queuedRequest.request);
      } else {
        // Batch processing
        const batchId = this.batchIdCounter++;
        const batchRequests = batch.map(qr => qr.request);
        
        console.log(chalk.blue(`📦 WorkerManager: Processing batch ${batchId} with ${batch.length} requests`));
        
        // Set up batch completion handler
        const handleBatchResult = (response: WorkerResponse) => {
          console.log(chalk.green(`✅ Batch ${batchId} completed: ${response.successful_count}/${response.processed_count} successful`));
        };
        
        worker.once("batch_completed", handleBatchResult);
        worker.generateBatch(batchRequests, batchId);
      }
    }

    // Continue processing if there are more requests and workers
    if (this.requestQueue.length > 0) {
      console.log(chalk.cyan(`🔄 Still ${this.requestQueue.length} requests remaining, continuing`));
      // Use a small delay to prevent infinite loops
      setTimeout(() => this.processBatchQueue(), 10);
    } else {
      console.log(chalk.green(`✅ All requests processed`));
    }
  }

  private groupCompatibleRequests(): QueuedRequest[][] {
    if (this.requestQueue.length === 0) {
      return [];
    }

    const batches: QueuedRequest[][] = [];
    const remaining = [...this.requestQueue];
    
    // Clear the queue as we're processing all items
    this.requestQueue.length = 0;

    while (remaining.length > 0) {
      const batch: QueuedRequest[] = [];
      const baseRequest = remaining.shift()!;
      batch.push(baseRequest);

      // Find compatible requests for this batch
      let i = 0;
      while (i < remaining.length && batch.length < this.batchConfig.maxSize) {
        const candidate = remaining[i];
        
        if (this.areRequestsCompatible(baseRequest.request, candidate.request)) {
          batch.push(candidate);
          remaining.splice(i, 1);
        } else {
          i++;
        }
      }

      batches.push(batch);
    }

    return batches;
  }

  private areRequestsCompatible(req1: WorkerRequest, req2: WorkerRequest): boolean {
    const voice1 = req1.voice_settings || {};
    const voice2 = req2.voice_settings || {};

    // Check if voice settings are compatible
    const compatibilityKeys = ["voice", "checkpointPath", "speed", "pitch", "emotion", "style"];
    for (const key of compatibilityKeys) {
      if (voice1[key] !== voice2[key]) {
        return false;
      }
    }

    // Check text length constraints
    const text1Length = req1.text?.length || 0;
    const text2Length = req2.text?.length || 0;
    if (text1Length + text2Length > this.batchConfig.maxTextLength) {
      return false;
    }

    return true;
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
    
    // Clear batch timer
    if (this.batchTimer) {
      clearTimeout(this.batchTimer);
      this.batchTimer = null;
    }
    
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
    
    // Clear queued requests
    for (const queuedRequest of this.requestQueue) {
      queuedRequest.reject(new Error("WorkerManager shutting down"));
    }
    this.requestQueue.length = 0;

    setTimeout(() => {
      this.workers.clear();
      console.log(chalk.green(`✅ WorkerManager: Shutdown complete`));
    }, 6000);
  }
} 