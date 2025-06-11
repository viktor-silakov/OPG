#!/usr/bin/env node
import { spawn, ChildProcess } from 'child_process';
import { readdir, stat, unlink } from 'fs/promises';
import { existsSync } from 'fs';
import { join } from 'path';

interface QuickBenchmarkConfig {
  name: string;
  env: Record<string, string>;
  description: string;
}

interface QuickBenchmarkResult {
  config: string;
  description: string;
  totalTime: number;
  avgTimePerMessage: number;
  successfulFiles: number;
  efficiency: number;
  throughput: number;
}

class QuickAudioBenchmark {
  private readonly outputDir = 'output';
  private readonly serverPort = 3000;
  private readonly serverUrl = `http://localhost:${this.serverPort}`;
  
  // Test configurations focused on small batch, fast timeout parameters
  private readonly configs: QuickBenchmarkConfig[] = [
    {
      name: 'no_batching',
      description: 'No batching baseline',
      env: {
        ENABLE_BATCHING: 'false',
        CONCURRENT_REQUESTS: '2',
        MAX_BATCH_SIZE: '4',
        BATCH_TIMEOUT_MS: '500'
      }
    },
    {
      name: 'small_batch_fast',
      description: 'Small batches with fast timeout (3 concurrent)',
      env: {
        ENABLE_BATCHING: 'true',
        CONCURRENT_REQUESTS: '3',
        MAX_BATCH_SIZE: '2',
        BATCH_TIMEOUT_MS: '100'
      }
    },
    {
      name: 'small_batch_low_concurrent',
      description: 'Small batches with lower concurrency (2 concurrent)',
      env: {
        ENABLE_BATCHING: 'true',
        CONCURRENT_REQUESTS: '2',
        MAX_BATCH_SIZE: '2',
        BATCH_TIMEOUT_MS: '100'
      }
    },
    {
      name: 'small_batch_high_concurrent',
      description: 'Small batches with higher concurrency (4 concurrent)',
      env: {
        ENABLE_BATCHING: 'true',
        CONCURRENT_REQUESTS: '4',
        MAX_BATCH_SIZE: '2',
        BATCH_TIMEOUT_MS: '100'
      }
    },
    {
      name: 'small_batch_medium_timeout',
      description: 'Small batches with medium timeout (200ms)',
      env: {
        ENABLE_BATCHING: 'true',
        CONCURRENT_REQUESTS: '3',
        MAX_BATCH_SIZE: '2',
        BATCH_TIMEOUT_MS: '200'
      }
    }
  ];

  private readonly testMessages = [
    { text: "Hello, this is a short test message.", speaker: "default", voiceSettings: { speed: 1.0, volume: 0, pitch: 1.0 } },
    { text: "This is a medium length message that contains more words.", speaker: "default", voiceSettings: { speed: 1.0, volume: 0, pitch: 1.0 } },
    { text: "This is a longer test message with much more content to process.", speaker: "default", voiceSettings: { speed: 1.0, volume: 0, pitch: 1.0 } }
  ];

  async run(): Promise<void> {
    console.log('🚀 Quick Audio Batching Benchmark\n');
    console.log(`📊 Testing ${this.configs.length} configurations with ${this.testMessages.length} messages each\n`);

    const results: QuickBenchmarkResult[] = [];

    for (let i = 0; i < this.configs.length; i++) {
      const config = this.configs[i];
      console.log(`\n🧪 [${i + 1}/${this.configs.length}] Testing: ${config.name}`);
      console.log(`   ${config.description}`);
      
      try {
        const result = await this.benchmarkConfiguration(config);
        results.push(result);
        console.log(`✅ Configuration ${config.name} completed`);
      } catch (error) {
        console.error(`❌ Configuration ${config.name} failed:`, error);
        results.push({
          config: config.name,
          description: config.description,
          totalTime: -1,
          avgTimePerMessage: -1,
          successfulFiles: 0,
          efficiency: 0,
          throughput: 0
        });
      }
      
      // Brief pause between tests
      if (i < this.configs.length - 1) {
        console.log('   ⏸️ Pausing 5 seconds between tests...');
        await this.sleep(5000);
      }
    }

    this.displayResults(results);
    this.recommendConfiguration(results);
  }

  private async benchmarkConfiguration(config: QuickBenchmarkConfig): Promise<QuickBenchmarkResult> {
    await this.cleanOutputDirectory();

    console.log('   🔄 Starting server...');
    const serverProcess = await this.startServer(config.env);
    
    try {
      await this.waitForServer();
      
      const startTime = Date.now();
      await this.sendTestRequests();
      await this.waitForAllFiles();
      const endTime = Date.now();
      
      const totalTime = (endTime - startTime) / 1000;
      const fileStats = await this.analyzeGeneratedFiles();

      return {
        config: config.name,
        description: config.description,
        totalTime,
        avgTimePerMessage: totalTime / this.testMessages.length,
        successfulFiles: fileStats.count,
        efficiency: (fileStats.count / this.testMessages.length) * 100,
        throughput: fileStats.count / totalTime
      };

    } finally {
      this.killServer(serverProcess);
    }
  }

  private async startServer(env: Record<string, string>): Promise<ChildProcess> {
    return new Promise((resolve, reject) => {
      const serverProcess = spawn('yarn', ['start'], {
        cwd: process.cwd(),
        env: { ...process.env, ...env },
        stdio: ['pipe', 'pipe', 'pipe']
      });

      let serverReady = false;

      serverProcess.stdout?.on('data', (data) => {
        const output = data.toString();
        if (output.includes('Server running') || output.includes('ready') || output.includes('listening')) {
          if (!serverReady) {
            serverReady = true;
            console.log('   ✅ Server ready');
            resolve(serverProcess);
          }
        }
      });

      serverProcess.on('error', (error) => {
        reject(new Error(`Failed to start server: ${error.message}`));
      });

      setTimeout(() => {
        if (!serverReady) {
          reject(new Error('Server startup timeout'));
        }
      }, 30000);
    });
  }

  private async waitForServer(): Promise<void> {
    for (let i = 0; i < 30; i++) {
      try {
        const response = await fetch(`${this.serverUrl}/status`);
        if (response.ok) return;
      } catch (error) {
        // Server not ready yet
      }
      await this.sleep(1000);
    }
    throw new Error('Server failed to become ready');
  }

  private async sendTestRequests(): Promise<void> {
    console.log('   📤 Sending requests...');

    const conversationData = {
      podcast_name: "Quick Benchmark",
      filename: "quick_benchmark.wav",
      conversation: this.testMessages.map((msg, index) => ({
        id: index + 1,
        speaker: msg.speaker,
        text: msg.text,
        voiceSettings: msg.voiceSettings
      }))
    };

    const response = await fetch(`${this.serverUrl}/generate-podcast`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ conversationData })
    });

    if (!response.ok) {
      throw new Error(`Request failed: ${response.status} ${response.statusText}`);
    }
  }

  private async waitForAllFiles(): Promise<void> {
    console.log('   ⏳ Waiting for generation...');
    
    const expectedFiles = this.testMessages.length;
    const maxWaitTime = 300000; // 5 minutes
    const startTime = Date.now();

    while (Date.now() - startTime < maxWaitTime) {
      const files = await this.getOutputFiles();
      
      if (files.length >= expectedFiles) {
        console.log(`   ✅ Generated ${files.length}/${expectedFiles} files`);
        return;
      }
      
      // Update progress every 10 seconds
      if ((Date.now() - startTime) % 10000 < 2000) {
        console.log(`   ⏳ Progress: ${files.length}/${expectedFiles} files`);
      }
      
      await this.sleep(2000);
    }

    console.log(`   ⚠️ Timeout - only ${(await this.getOutputFiles()).length}/${expectedFiles} files`);
  }

  private async getOutputFiles(): Promise<string[]> {
    try {
      if (!existsSync(this.outputDir)) return [];
      
      const allFiles: string[] = [];
      const items = await readdir(this.outputDir);
      
      for (const item of items) {
        const itemPath = join(this.outputDir, item);
        const stats = await stat(itemPath);
        
        if (stats.isDirectory()) {
          try {
            const subFiles = await readdir(itemPath);
            const wavFiles = subFiles.filter(file => file.endsWith('.wav'));
            allFiles.push(...wavFiles);
          } catch {
            // Ignore errors
          }
        } else if (item.endsWith('.wav')) {
          allFiles.push(item);
        }
      }
      
      return allFiles;
    } catch {
      return [];
    }
  }

  private async analyzeGeneratedFiles(): Promise<{count: number}> {
    const files = await this.getOutputFiles();
    return { count: files.length };
  }

  private async cleanOutputDirectory(): Promise<void> {
    try {
      if (!existsSync(this.outputDir)) return;
      
      const items = await readdir(this.outputDir);
      
      for (const item of items) {
        const itemPath = join(this.outputDir, item);
        const stats = await stat(itemPath);
        
        if (stats.isDirectory()) {
          try {
            const subFiles = await readdir(itemPath);
            for (const subFile of subFiles) {
              await unlink(join(itemPath, subFile));
            }
            await require('fs/promises').rmdir(itemPath);
          } catch {
            // Ignore errors
          }
        } else if (item.endsWith('.wav')) {
          await unlink(itemPath);
        }
      }
    } catch (error) {
      // Directory might not exist yet
    }
  }

  private killServer(serverProcess: ChildProcess): void {
    console.log('   🛑 Stopping server...');
    if (serverProcess && !serverProcess.killed) {
      serverProcess.kill('SIGTERM');
      setTimeout(() => {
        if (!serverProcess.killed) {
          serverProcess.kill('SIGKILL');
        }
      }, 5000);
    }
  }

  private displayResults(results: QuickBenchmarkResult[]): void {
    console.log('\n📊 QUICK BENCHMARK RESULTS\n');
    
    const tableData = results.map(result => ({
      'Configuration': result.config,
      'Description': result.description,
      'Total Time (s)': result.totalTime > 0 ? result.totalTime.toFixed(1) : 'FAILED',
      'Avg/Message (s)': result.avgTimePerMessage > 0 ? result.avgTimePerMessage.toFixed(1) : 'FAILED',
      'Success Rate (%)': result.efficiency.toFixed(0),
      'Throughput (msg/s)': result.throughput > 0 ? result.throughput.toFixed(2) : '0.00'
    }));

    console.table(tableData);
  }

  private recommendConfiguration(results: QuickBenchmarkResult[]): void {
    console.log('\n🏆 QUICK RECOMMENDATIONS\n');

    const validResults = results.filter(r => r.efficiency > 0);
    
    if (validResults.length === 0) {
      console.log('❌ No successful configurations found');
      return;
    }

    // Find best overall
    const best = validResults.reduce((prev, curr) => {
      const prevScore = prev.throughput * (prev.efficiency / 100);
      const currScore = curr.throughput * (curr.efficiency / 100);
      return currScore > prevScore ? curr : prev;
    });

    console.log('🥇 BEST CONFIGURATION:');
    console.log(`   ${best.config}: ${best.description}`);
    console.log(`   Time per message: ${best.avgTimePerMessage.toFixed(1)}s`);
    console.log(`   Success rate: ${best.efficiency.toFixed(0)}%`);
    console.log(`   Throughput: ${best.throughput.toFixed(2)} msg/s`);

    const bestConfig = this.configs.find(c => c.name === best.config);
    if (bestConfig) {
      console.log('\n🎯 RECOMMENDED SETTINGS:');
      Object.entries(bestConfig.env).forEach(([key, value]) => {
        console.log(`export ${key}=${value}`);
      });
    }
  }

  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

// Run benchmark
if (require.main === module) {
  const benchmark = new QuickAudioBenchmark();
  benchmark.run().catch(console.error);
}

export { QuickAudioBenchmark }; 