import { exec, spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { join } from "node:path";
import chalk from "chalk";

export async function waitForServersToLoad(concurrentRequests: number): Promise<void> {
  console.log(chalk.blue("🐟 Checking fs-python environment readiness..."));
  
  return new Promise<void>((resolve, reject) => {
    // Check for fs-python directory (in project root)
    const fsPythonDir = "../fs-python";
    if (!existsSync(fsPythonDir)) {
      reject(new Error("fs-python directory not found"));
      return;
    }

    // Check for cli_tts.py
    const cliPath = join(fsPythonDir, "cli_tts.py");
    if (!existsSync(cliPath)) {
      reject(new Error("cli_tts.py file not found in fs-python"));
      return;
    }

    // Check poetry and dependencies
    console.log(chalk.grey("📦 Checking Poetry installation and dependencies..."));
    const checkProcess = spawn("poetry", ["check"], {
      cwd: fsPythonDir,
      stdio: ['pipe', 'pipe', 'pipe']
    });

    let stdout = '';
    let stderr = '';

    checkProcess.stdout?.on("data", (data) => {
      const output = data.toString();
      stdout += output;
      console.log(chalk.grey(`stdout: ${output.trim()}`));
    });

    checkProcess.stderr?.on("data", (data) => {
      const output = data.toString();
      stderr += output;
      console.log(chalk.grey(`stderr: ${output.trim()}`));
    });

    checkProcess.on("close", (code) => {
      if (code === 0) {
        console.log(chalk.green("✅ fs-python environment ready"));
        console.log(chalk.blue(`🎯 Ready for ${concurrentRequests} parallel requests`));
        resolve();
      } else {
        reject(new Error(`Poetry check exited with code ${code}. May need to run 'poetry install' in fs-python`));
      }
    });

    checkProcess.on("error", (err) => {
      reject(new Error(`Poetry check error: ${err.message}. Make sure Poetry is installed`));
    });
  });
} 