#!/usr/bin/env npx tsx
/**
 * Example: Semantic Token Cache Demonstration
 * 
 * This script shows the difference between enabled and disabled semantic token caching.
 * Run with: npx tsx examples/cache-example.ts
 */

import { generateAudio, shutdownWorkerManager } from "../helpers/audioGenerator.js";
import { mkdirSync, existsSync } from "node:fs";
import chalk from "chalk";

async function demonstrateCache() {
  console.log(chalk.blue("🧪 Semantic Token Cache Demonstration"));
  console.log("=" * 50);
  
  // Test text
  const testText = "Hello, this is a demonstration of semantic token caching in Fish Speech.";
  const outputDir = "cache_demo_output";
  
  // Create output directory
  if (!existsSync(outputDir)) {
    mkdirSync(outputDir, { recursive: true });
  }
  
  console.log(chalk.blue("\n📋 Test Configuration:"));
  console.log(chalk.gray(`   Text: "${testText}"`));
  console.log(chalk.gray(`   SEMANTIC_TOKEN_CACHE: ${process.env.SEMANTIC_TOKEN_CACHE || "true (default)"}`));
  console.log(chalk.gray(`   CONCURRENT_REQUESTS: ${process.env.CONCURRENT_REQUESTS || "2 (default)"}`));
  
  try {
    console.log(chalk.blue("\n🚀 Starting first generation (cache miss)..."));
    const start1 = Date.now();
    
    const result1 = await generateAudio(
      testText,
      "default",
      1,
      1,
      outputDir
    );
    
    const time1 = (Date.now() - start1) / 1000;
    
    if (result1) {
      console.log(chalk.green(`✅ First generation: ${time1.toFixed(2)}s`));
    } else {
      console.log(chalk.red(`❌ First generation failed`));
      return;
    }
    
    // Wait a moment
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    console.log(chalk.blue("\n🔄 Starting second generation (potential cache hit)..."));
    const start2 = Date.now();
    
    const result2 = await generateAudio(
      testText,
      "default", 
      2,
      2,
      outputDir
    );
    
    const time2 = (Date.now() - start2) / 1000;
    
    if (result2) {
      console.log(chalk.green(`✅ Second generation: ${time2.toFixed(2)}s`));
    } else {
      console.log(chalk.red(`❌ Second generation failed`));
      return;
    }
    
    // Analysis
    console.log(chalk.blue("\n📊 Performance Analysis:"));
    const speedup = time1 / time2;
    const cacheEnabled = process.env.SEMANTIC_TOKEN_CACHE !== "false";
    
    if (cacheEnabled) {
      if (speedup > 1.2) {
        console.log(chalk.green(`🚀 Cache effective! Second generation was ${speedup.toFixed(1)}x faster`));
        console.log(chalk.gray(`   This indicates semantic tokens were cached and reused`));
      } else {
        console.log(chalk.yellow(`⚠️  Cache might not be effective (${speedup.toFixed(1)}x speedup)`));
        console.log(chalk.gray(`   Possible reasons: different text, cache miss, or warm-up effects`));
      }
    } else {
      console.log(chalk.blue(`🔄 Cache disabled - consistent performance expected`));
      console.log(chalk.gray(`   Speedup: ${speedup.toFixed(1)}x (should be close to 1.0)`));
    }
    
    console.log(chalk.blue("\n💡 Cache Benefits:"));
    if (cacheEnabled) {
      console.log(chalk.green("   ✅ Enabled - Faster subsequent generations with similar text"));
      console.log(chalk.gray("   📁 Cache location: fs-python/cache/semantic_tokens/"));
      console.log(chalk.gray("   🧹 Clear cache: rm -rf fs-python/cache/semantic_tokens/*"));
    } else {
      console.log(chalk.yellow("   🚫 Disabled - Consistent generation time, no disk cache"));
      console.log(chalk.gray("   🔧 Enable: export SEMANTIC_TOKEN_CACHE=true"));
    }
    
    console.log(chalk.blue("\n🎯 Recommendations:"));
    console.log(chalk.gray("   🏢 Production: Enable cache for better performance"));
    console.log(chalk.gray("   🧪 Development: Disable cache for consistent testing"));
    console.log(chalk.gray("   💾 Low storage: Disable cache to save disk space"));
    
  } catch (error) {
    console.error(chalk.red(`❌ Demo failed: ${error}`));
    console.error(error.stack);
  } finally {
    // Cleanup
    console.log(chalk.yellow("\n🧹 Shutting down workers..."));
    shutdownWorkerManager();
    
    setTimeout(() => {
      console.log(chalk.green("\n🎉 Demo completed!"));
      console.log(chalk.gray("   Generated files are in: " + outputDir));
      process.exit(0);
    }, 2000);
  }
}

// Run if executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  demonstrateCache().catch(console.error);
}

export { demonstrateCache }; 