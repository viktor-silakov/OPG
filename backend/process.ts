import { config } from "dotenv";
import { writeFileSync } from "fs";
import {
  loadConversationData,
  createOutputDirectory,
  waitForServersToLoad,
  createMessageChunks,
  processMessageChunks,
  sortFilesByID,
  processAudioFiles
} from "./helpers/index.js";

// Load environment variables from .env file
config();

const CONCURRENT_REQUESTS = parseInt(process.env.CONCURRENT_REQUESTS || "8");

async function main() {
  console.time("generationTime");
  
  try {
    await waitForServersToLoad(CONCURRENT_REQUESTS);
    console.log("🐟 fs-python is ready, starting audio generation, concurrent requests: ", CONCURRENT_REQUESTS);

    // Check if there is data in environment variables
    let conversationData;
    if (process.env.CONVERSATION_DATA) {
      console.log("📝 Using data from environment variables");
      try {
        conversationData = JSON.parse(process.env.CONVERSATION_DATA);
        // Write to a temporary file for processing
        writeFileSync("conversation.json", JSON.stringify(conversationData, null, 2));
        console.log("✅ Temporary conversation.json created from provided data");
      } catch (error) {
        console.error("❌ Error parsing data from environment variables:", error);
        throw error;
      }
    }

    conversationData = loadConversationData("conversation.json");
    const { podcast_name, filename, conversation } = conversationData;
    
    console.log(`🎙️ Generating podcast: "${podcast_name}"`);
    console.log(`📁 Output file: ${filename}`);
    
    const outputDir = createOutputDirectory(filename);

    const chunks = createMessageChunks(conversation, CONCURRENT_REQUESTS);
    const generatedFiles = await processMessageChunks(chunks, outputDir);

    if (generatedFiles.length > 0) {
      const sortedFiles = sortFilesByID(generatedFiles);
      await processAudioFiles(sortedFiles, outputDir, filename);
    } else {
      console.log("❌ Audio files were not generated successfully.");
    }
  } catch (error) {
    console.error("❌ Error in main process:", error);
  } finally {
    console.timeEnd("generationTime");
  }
}

main().catch(console.error);
