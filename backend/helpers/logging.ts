import { appendFileSync, existsSync, mkdirSync, readdirSync, statSync } from "fs";
import { join } from "path";

const LOGS_DIR = "../logs";

// Ensure logs directory exists
if (!existsSync(LOGS_DIR)) {
  mkdirSync(LOGS_DIR, { recursive: true });
}

export function logToFile(
  type: "success" | "error" | "api_request" | "api_response",
  data: any
) {
  const timestamp = new Date().toISOString();
  const date = timestamp.split("T")[0]; // YYYY-MM-DD
  const logFileName = `${date}-${type}.log`;
  const logFilePath = join(LOGS_DIR, logFileName);

  const logEntry = {
    timestamp,
    type,
    data,
  };

  const logLine =
    JSON.stringify(logEntry, null, 2) + "\n" + "=".repeat(80) + "\n";

  try {
    appendFileSync(logFilePath, logLine, "utf-8");
    console.log(`📝 Log saved: ${logFileName}`);
  } catch (error) {
    console.error("Error writing to log file:", error);
  }
}

export function formatFileSize(bytes: number): string {
  if (bytes === 0) return "0 B";
  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
}

export function getLogsList() {
  if (!existsSync(LOGS_DIR)) {
    return [];
  }

  const files = readdirSync(LOGS_DIR);
  const logs = [];

  for (const file of files) {
    if (file.endsWith(".log")) {
      const filePath = join(LOGS_DIR, file);
      const stats = statSync(filePath);

      // Parse filename to get date and type
      const [date, type] = file.replace(".log", "").split("-");

      logs.push({
        filename: file,
        date,
        type,
        size: stats.size,
        sizeFormatted: formatFileSize(stats.size),
        created: stats.birthtime,
        modified: stats.mtime,
      });
    }
  }

  // Sort by date (newest first)
  logs.sort(
    (a, b) => new Date(b.modified).getTime() - new Date(a.modified).getTime()
  );

  return logs;
} 