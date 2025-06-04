import { existsSync, readdirSync, statSync, readFileSync, writeFileSync, unlinkSync } from "fs";
import { join } from "path";

export function getPodcastsList() {
  console.log("Getting podcasts list...");

  if (!existsSync("../output")) {
    console.log("Output directory doesn't exist");
    return [];
  }

  console.log("Reading output directory contents...");
  const items = readdirSync("../output");
  console.log("Found items:", items);

  const podcasts = [];

  for (const item of items) {
    if (item === ".DS_Store") continue;

    const itemPath = join("../output", item);
    console.log(`Processing item: ${item}`);

    try {
      const stats = statSync(itemPath);

      if (stats.isDirectory()) {
        console.log(`${item} is a directory`);
        const dirContents = readdirSync(itemPath);
        console.log(`Directory ${item} contents:`, dirContents);

        // Look for wav file
        let wavFile = dirContents.find((file) => file === `${item}.wav`);
        if (!wavFile) {
          wavFile = dirContents.find((file) => file.endsWith(".wav"));
        }

        if (wavFile) {
          console.log(`Found wav file: ${wavFile}`);
          const wavPath = join(itemPath, wavFile);
          const wavStats = statSync(wavPath);

          podcasts.push({
            filename: item.endsWith(".wav") ? item : `${item}.wav`,
            url: `/output/${item}/${wavFile}`,
            size: wavStats.size,
            created: stats.birthtime,
            modified: wavStats.mtime,
          });
        }
      }
    } catch (err) {
      console.error(`Error processing ${item}:`, err);
    }
  }

  console.log(
    "Found podcasts:",
    podcasts.map((p) => p.filename)
  );
  
  podcasts.sort(
    (a, b) => new Date(b.created).getTime() - new Date(a.created).getTime()
  );

  return podcasts;
}

export function getPromptsList() {
  console.log("Getting system prompts list...");

  const promptsDir = "../prompts";
  if (!existsSync(promptsDir)) {
    console.log("Prompts directory doesn't exist");
    return [];
  }

  const files = readdirSync(promptsDir);
  const prompts = [];

  for (const file of files) {
    if (file.endsWith(".md")) {
      const filePath = join(promptsDir, file);
      const stats = statSync(filePath);
      const name = file.replace(".md", "");

      prompts.push({
        name,
        displayName: name,
        description: name,
        filename: file,
        size: stats.size,
        modified: stats.mtime,
      });
    }
  }

  // Sort alphabetically
  prompts.sort((a, b) => a.name.localeCompare(b.name));

  console.log(
    "Found prompts:",
    prompts.map((p) => p.name)
  );
  
  return prompts;
}

export function getPromptContent(name: string) {
  const promptPath = join("../prompts", `${name}.md`);

  if (!existsSync(promptPath)) {
    throw new Error("Prompt not found");
  }

  const content = readFileSync(promptPath, "utf-8");
  return {
    name,
    content,
  };
}

export function savePromptContent(name: string, content: string) {
  console.log(`Saving prompt: ${name}`);

  const promptsDir = "../prompts";
  if (!existsSync(promptsDir)) {
    throw new Error("Prompts directory doesn't exist");
  }

  const promptPath = join(promptsDir, `${name}.md`);
  
  try {
    writeFileSync(promptPath, content, "utf-8");
    console.log(`Prompt ${name} saved successfully`);
    
    return {
      success: true,
      message: `Prompt "${name}" saved successfully`,
      name,
    };
  } catch (error) {
    console.error(`Error saving prompt ${name}:`, error);
    throw new Error(`Failed to save prompt: ${error instanceof Error ? error.message : 'Unknown error'}`);
  }
}

export function deletePrompt(name: string) {
  console.log(`Deleting prompt: ${name}`);

  const promptPath = join("../prompts", `${name}.md`);

  if (!existsSync(promptPath)) {
    throw new Error("Prompt not found");
  }

  try {
    unlinkSync(promptPath);
    console.log(`Prompt ${name} deleted successfully`);
    
    return {
      success: true,
      message: `Prompt "${name}" deleted successfully`,
    };
  } catch (error) {
    console.error(`Error deleting prompt ${name}:`, error);
    throw new Error(`Failed to delete prompt: ${error instanceof Error ? error.message : 'Unknown error'}`);
  }
} 