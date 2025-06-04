import { useState, useEffect } from "react";
import {
  Container,
  Stack,
  Alert,
  Button,
  Group,
  Text,
  Textarea,
  Paper,
  Badge,
  Code,
  Divider,
} from "@mantine/core";
import {
  IconSettings,
  IconAlertCircle,
  IconCheck,
  IconReload,
  IconDeviceFloppy,
  IconExclamationMark,
} from "@tabler/icons-react";
import { notifications } from "@mantine/notifications";
import { apiClient, type UpdateEnvRequest } from "../api/client";

interface EnvVariable {
  key: string;
  value: string;
  comment?: string;
}

export function EnvEditor() {
  const [content, setContent] = useState("");
  const [originalContent, setOriginalContent] = useState("");
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [envExists, setEnvExists] = useState(false);
  const [parsedVars, setParsedVars] = useState<EnvVariable[]>([]);
  const [hasChanges, setHasChanges] = useState(false);

  // Parse .env content to extract variables
  const parseEnvContent = (envContent: string): EnvVariable[] => {
    const lines = envContent.split('\n');
    const variables: EnvVariable[] = [];
    
    for (const line of lines) {
      const trimmed = line.trim();
      
      // Skip empty lines and comments
      if (!trimmed || trimmed.startsWith('#')) {
        continue;
      }
      
      // Parse KEY=VALUE format
      const match = trimmed.match(/^([A-Z_][A-Z0-9_]*)\s*=\s*(.*)$/);
      if (match) {
        const [, key, value] = match;
        variables.push({
          key,
          value: value.replace(/^["']|["']$/g, ''), // Remove quotes
        });
      }
    }
    
    return variables;
  };

  // Load .env content
  const loadEnvContent = async () => {
    setLoading(true);
    try {
      const response = await apiClient.getEnvContent();
      setContent(response.content);
      setOriginalContent(response.content);
      setEnvExists(response.exists);
      setParsedVars(parseEnvContent(response.content));
      setHasChanges(false);
    } catch (error) {
      console.error("Error loading .env file:", error);
      notifications.show({
        title: "Error",
        message: "Failed to load .env file",
        color: "red",
        icon: <IconAlertCircle />,
      });
    } finally {
      setLoading(false);
    }
  };

  // Save .env content
  const saveEnvContent = async () => {
    setSaving(true);
    try {
      const request: UpdateEnvRequest = {
        content,
      };

      await apiClient.updateEnvContent(request);
      
      notifications.show({
        title: "Success",
        message: ".env file saved successfully. Restart the server to apply changes.",
        color: "green",
        icon: <IconCheck />,
      });

      setOriginalContent(content);
      setHasChanges(false);
    } catch (error) {
      console.error("Error saving .env file:", error);
      notifications.show({
        title: "Error",
        message: error instanceof Error ? error.message : "Unknown error",
        color: "red",
        icon: <IconAlertCircle />,
      });
    } finally {
      setSaving(false);
    }
  };

  // Handle content change
  const handleContentChange = (value: string) => {
    setContent(value);
    setHasChanges(value !== originalContent);
    setParsedVars(parseEnvContent(value));
  };

  // Reset changes
  const resetChanges = () => {
    setContent(originalContent);
    setHasChanges(false);
    setParsedVars(parseEnvContent(originalContent));
  };

  // Get example .env content
  const getExampleContent = () => {
    return `# API Configuration
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.0-flash

# Server Configuration  
PORT=3000

# Voice Configuration
DEFAULT_VOICE=RU_Female_Anya

# Optional: Advanced Settings
GEMINI_TEMPERATURE=0.7
GEMINI_MAX_TOKENS=4000

# Optional: Fish Speech Settings
FISH_SPEECH_MODEL_PATH=../fs-python/checkpoints/fish-speech-1.4
FISH_SPEECH_DEVICE=mps

# Optional: Logging
LOG_LEVEL=info
`;
  };

  useEffect(() => {
    loadEnvContent();
  }, []);

  // Important environment variables
  const importantVars = ['GEMINI_API_KEY', 'GEMINI_MODEL', 'PORT'];
  const missingImportantVars = importantVars.filter(
    varName => !parsedVars.some(v => v.key === varName)
  );

  return (
    <Container size="lg">
      <Stack gap="lg">
        <div>
          <Group align="center" gap="md">
            <IconSettings size={24} />
            <Text size="xl" fw={600}>Environment Configuration</Text>
            {hasChanges && (
              <Badge color="orange" variant="light">
                Unsaved changes
              </Badge>
            )}
          </Group>
          <Text size="sm" c="dimmed">
            Configure API keys and system settings
          </Text>
        </div>

        <Alert
          icon={<IconExclamationMark />}
          title="Important"
          color="blue"
          variant="light"
        >
          <Stack gap="xs">
            <Text size="sm">
              Changes to the .env file require a server restart to take effect.
            </Text>
            <Text size="sm">
              A backup will be created automatically before saving changes.
            </Text>
          </Stack>
        </Alert>

        {missingImportantVars.length > 0 && (
          <Alert
            icon={<IconAlertCircle />}
            title="Missing Required Variables"
            color="red"
            variant="light"
          >
            <Text size="sm">
              The following important variables are missing: {missingImportantVars.join(', ')}
            </Text>
          </Alert>
        )}

        <Paper p="md" withBorder>
          <Stack gap="md">
            <Group justify="space-between">
              <Text fw={500}>Environment Variables</Text>
              <Group gap="xs">
                <Button
                  variant="light"
                  size="sm"
                  onClick={loadEnvContent}
                  loading={loading}
                  leftSection={<IconReload size={16} />}
                >
                  Reload
                </Button>
                {hasChanges && (
                  <Button
                    variant="light"
                    size="sm"
                    color="gray"
                    onClick={resetChanges}
                  >
                    Reset
                  </Button>
                )}
                <Button
                  size="sm"
                  onClick={saveEnvContent}
                  loading={saving}
                  disabled={!hasChanges}
                  leftSection={<IconDeviceFloppy size={16} />}
                >
                  Save Changes
                </Button>
              </Group>
            </Group>

            <Textarea
              placeholder={
                !envExists 
                  ? "No .env file found. Create one by adding environment variables below..."
                  : "Loading .env content..."
              }
              value={content}
              onChange={(event) => handleContentChange(event.target.value)}
              minRows={20}
              maxRows={30}
              autosize
              resize="vertical"
              styles={{
                input: {
                  fontFamily: 'Monaco, Menlo, "Ubuntu Mono", monospace',
                  fontSize: '14px',
                  lineHeight: '1.5',
                }
              }}
            />

            {parsedVars.length > 0 && (
              <div>
                <Divider label="Detected Variables" />
                <Group gap="xs" mt="sm">
                  {parsedVars.map((variable, index) => (
                    <Badge
                      key={index}
                      variant={importantVars.includes(variable.key) ? "filled" : "light"}
                      color={importantVars.includes(variable.key) ? "blue" : "gray"}
                      size="sm"
                    >
                      {variable.key}
                    </Badge>
                  ))}
                </Group>
              </div>
            )}
          </Stack>
        </Paper>

        <Paper p="md" withBorder>
          <Stack gap="md">
            <Text fw={500}>Example .env Configuration</Text>
            <Code block>{getExampleContent()}</Code>
            <Button
              variant="light"
              size="sm"
              onClick={() => {
                if (!content.trim()) {
                  handleContentChange(getExampleContent());
                } else {
                  handleContentChange(content + '\n\n' + getExampleContent());
                }
              }}
            >
              {!content.trim() ? 'Use Example' : 'Append Example'}
            </Button>
          </Stack>
        </Paper>

        <Alert
          icon={<IconAlertCircle />}
          title="Configuration Tips"
          variant="light"
        >
          <Stack gap="xs">
            <Text size="sm">
              • <strong>GEMINI_API_KEY</strong>: Get your API key from Google AI Studio
            </Text>
            <Text size="sm">
              • <strong>GEMINI_MODEL</strong>: Use "gemini-1.5-flash-8b" for free tier or "gemini-1.5-pro" for better quality
            </Text>
            <Text size="sm">
              • <strong>PORT</strong>: Server port (default: 3000)
            </Text>
            <Text size="sm">
              • Lines starting with # are comments and will be ignored
            </Text>
            <Text size="sm">
              • Variable names should be UPPERCASE with underscores
            </Text>
          </Stack>
        </Alert>
      </Stack>
    </Container>
  );
} 