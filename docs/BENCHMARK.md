# Audio Batching Benchmark

Comprehensive benchmark for testing different audio generation batching configurations.

## Quick Start

```bash
# Run benchmark (25-30 minutes)
npx tsx benchmark/benchmark.ts

# Or using npm script
npm run benchmark
```

## What it tests

The benchmark tests 5 different configurations focused on small batch, fast timeout parameters:

1. **no_batching** - No batching baseline (2 concurrent, batching disabled)
2. **small_batch_fast** - Small batches with fast timeout (3 concurrent, batch size 2, 100ms timeout)
3. **small_batch_low_concurrent** - Small batches with lower concurrency (2 concurrent, batch size 2, 100ms timeout)
4. **small_batch_high_concurrent** - Small batches with higher concurrency (4 concurrent, batch size 2, 100ms timeout)
5. **small_batch_medium_timeout** - Small batches with medium timeout (3 concurrent, batch size 2, 200ms timeout)

## Metrics Measured

- **Total Time** - Complete generation time for all messages
- **Avg/Message** - Average time per message
- **Success Rate** - Percentage of successfully generated files
- **Throughput** - Messages processed per second

## Test Data

Each configuration is tested with 3 messages of varying lengths:
- Short message: "Hello, this is a short test message."
- Medium message: "This is a medium length message that contains more words."
- Long message: "This is a longer test message with much more content to process."

All messages use default speaker with standard voice settings (speed: 1.0, volume: 0, pitch: 1.0).

## Output

The benchmark provides:
- 📊 Detailed results table with all configurations
- 🏆 Best overall configuration recommendation based on throughput and success rate
- 🎯 Ready-to-use environment variables for optimal configuration

## Configuration Parameters Tested

- **ENABLE_BATCHING** - Enable/disable request batching (true/false)
- **MAX_BATCH_SIZE** - Maximum requests per batch (2-8)
- **BATCH_TIMEOUT_MS** - Timeout before processing incomplete batch (100-1000ms)
- **CONCURRENT_REQUESTS** - Number of concurrent requests (2-4)

## System Requirements

- Node.js 18+ (for native fetch support)
- tsx installed globally  
- Working Fish Speech environment
- Available port 3000 for server
- Server should respond to /status endpoint for health checks
- yarn command available for starting server

## Usage Tips

- Clean output directory between runs for accurate results
- Each configuration runs independently with server restart
- 5-second pause between configurations to ensure clean state
- Maximum wait time per configuration: 5 minutes
- Server startup timeout: 30 seconds

## Understanding Results

### Interpreting Metrics

- **Total Time**: Lower is better - indicates faster overall processing
- **Avg/Message**: Lower is better - shows efficiency per individual message
- **Success Rate**: Higher is better - percentage of successfully generated audio files
- **Throughput**: Higher is better - messages processed per second

### What Affects Performance

- **Batch Size**: Smaller batches (2-3) often perform better due to reduced coordination overhead
- **Timeout**: Fast timeouts (100-200ms) prevent waiting for incomplete batches
- **Concurrency**: Optimal concurrency depends on system resources and workload
- **System Load**: Results may vary based on CPU, memory, and disk I/O

### Recommended Approach

1. Run benchmark on your target environment
2. Identify the configuration with best throughput and success rate
3. Test the recommended settings with your actual workload
4. Fine-tune parameters based on production requirements

## Troubleshooting

### Common Issues

- **Server startup timeout**: Increase timeout in benchmark code or check server logs
- **Port conflicts**: Ensure port 3000 is available
- **Permission errors**: Check file system permissions for output directory
- **Memory issues**: Monitor system resources during benchmark

### Debug Mode

For debugging, you can modify the benchmark to run fewer configurations or add additional logging. 