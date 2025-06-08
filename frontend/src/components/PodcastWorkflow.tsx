import { useState, useEffect } from 'react';
import {
  Paper,
  Textarea,
  Button,
  Group,
  Text,
  Alert,
  JsonInput,
  Stack,
  Badge,
  Title,
  Loader,
  Progress,
  Stepper,
  Space,
  Divider,
  Select,
} from "@mantine/core";
import { useForm } from "@mantine/form";
import { notifications } from "@mantine/notifications";
import {
  IconCheck,
  IconX,
  IconScript,
  IconPlayerPlay,
  IconDownload,
  IconPlayerSkipForward,
} from "@tabler/icons-react";
import {
  apiClient,
  type GenerateScriptResponse,
  type GeneratePodcastResponse,
  type ProgressUpdate,
  type PromptInfo,
} from "../api/client";
import { NotificationManager } from "../utils/notifications";

interface PodcastGenerationData {
  promptName: string;
  userPrompt: string;
  systemPrompt: string;
}

interface PodcastWorkflowProps {
  generationData?: PodcastGenerationData | null;
}

export function PodcastWorkflow({ generationData }: PodcastWorkflowProps) {
  const [step, setStep] = useState(0);
  const [scriptLoading, setScriptLoading] = useState(false);
  const [podcastLoading, setPodcastLoading] = useState(false);
  const [scriptResult, setScriptResult] =
    useState<GenerateScriptResponse | null>(null);
  const [podcastResult, setPodcastResult] =
    useState<GeneratePodcastResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [defaultSystemPrompt, setDefaultSystemPrompt] = useState("");
  const [availablePrompts, setAvailablePrompts] = useState<PromptInfo[]>([]);
  const [promptsLoading, setPromptsLoading] = useState(false);
  const [selectedPrompt, setSelectedPrompt] = useState<string>("default");
  const [progressInfo, setProgressInfo] = useState({
    current: 0,
    total: 0,
    message: "",
  });
  const [currentJobId, setCurrentJobId] = useState<string | null>(null);
  const [ws, setWs] = useState<WebSocket | null>(null);

  // State for timer and replica count
  const [generationStartTime, setGenerationStartTime] = useState<number | null>(
    null
  );
  const [elapsedTime, setElapsedTime] = useState(0);
  const [replicasCount, setReplicasCount] = useState(0);
  const [showAudioPlayer, setShowAudioPlayer] = useState(false);

  const form = useForm({
    initialValues: {
      userPrompt: "",
      systemPrompt: "",
      conversationJson: "",
    },
    validate: {
      userPrompt: (value) => (!value.trim() ? "Specify the podcast topic" : null),
      conversationJson: (value) => {
        if (step >= 1 && !value.trim())
          return "Script is required to generate a podcast";
        if (step >= 1 && value.trim()) {
          try {
            JSON.parse(value);
            return null;
          } catch {
            return "Invalid JSON format";
          }
        }
        return null;
      },
    },
  });

  // Auto-fill form when generationData is provided
  useEffect(() => {
    if (generationData) {
      console.log('Received generation data:', generationData);
      
      // Set the form values
      form.setFieldValue("userPrompt", generationData.userPrompt);
      form.setFieldValue("systemPrompt", generationData.systemPrompt);
      
      // Set the selected prompt
      setSelectedPrompt(generationData.promptName);
      
      // Show notification
      notifications.show({
        title: "Prompt Applied",
        message: `Loaded content for "${generationData.promptName}" prompt`,
        color: "blue",
        icon: <IconCheck />,
      });
    }
  }, [generationData]);

  // Function to count replicas in JSON
  const countReplicas = (jsonString: string): number => {
    try {
      const data = JSON.parse(jsonString);
      if (data.conversation && Array.isArray(data.conversation)) {
        return data.conversation.length;
      }
      return 0;
    } catch {
      return 0;
    }
  };

  // Function to format time
  const formatElapsedTime = (seconds: number): string => {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}:${remainingSeconds.toString().padStart(2, "0")}`;
  };

  // Timer effect for generation
  useEffect(() => {
    let interval: number | undefined;

    if (podcastLoading && generationStartTime) {
      interval = window.setInterval(() => {
        const now = Date.now();
        const elapsed = Math.floor((now - generationStartTime) / 1000);
        setElapsedTime(elapsed);
      }, 1000);
    }

    return () => {
      if (interval) {
        clearInterval(interval);
      }
    };
  }, [podcastLoading, generationStartTime]);

  // Effect to count replicas when JSON changes
  useEffect(() => {
    const count = countReplicas(form.values.conversationJson);
    setReplicasCount(count);
  }, [form.values.conversationJson]);

  // Effect to show audio player after 5 seconds when reaching final step
  useEffect(() => {
    if (step === 2 && podcastResult) {
      setShowAudioPlayer(false); // Reset first
      const timer = setTimeout(() => {
        setShowAudioPlayer(true);
      }, 5000);

      return () => clearTimeout(timer);
    }
  }, [step, podcastResult]);

  // Load system prompts list
  useEffect(() => {
    const loadPrompts = async () => {
      setPromptsLoading(true);
      try {
        const response = await apiClient.getPrompts();
        setAvailablePrompts(response.prompts);

        // If there's a default prompt, select it (unless overridden by generationData)
        if (!generationData) {
          if (response.prompts.find((p) => p.name === "default")) {
            setSelectedPrompt("default");
          } else if (response.prompts.length > 0) {
            setSelectedPrompt(response.prompts[0].name);
          }
        }
      } catch (error) {
        console.error("Error loading prompts:", error);
        notifications.show({
          title: "Error",
          message: "Failed to load system prompts",
          color: "red",
          icon: <IconX size={16} />,
        });
      } finally {
        setPromptsLoading(false);
      }
    };

    loadPrompts();
  }, [generationData]);

  // Load content of selected prompt
  useEffect(() => {
    const loadPromptContent = async () => {
      if (!selectedPrompt) return;
      
      // Don't load if we already have generation data for this prompt
      if (generationData && generationData.promptName === selectedPrompt) {
        return;
      }

      try {
        const response = await apiClient.getPromptContent(selectedPrompt);
        setDefaultSystemPrompt(response.content);
        form.setFieldValue("systemPrompt", response.content);
      } catch (error) {
        console.error("Error loading prompt content:", error);
        // Fallback: try to load old prompt
        try {
          const response = await fetch("/system-prompt.md");
          if (response.ok) {
            const text = await response.text();
            setDefaultSystemPrompt(text);
            form.setFieldValue("systemPrompt", text);
          }
        } catch {
          console.log("Failed to load system-prompt.md");
        }
      }
    };

    loadPromptContent();
  }, [selectedPrompt, generationData]);

  // WebSocket cleanup on unmount
  useEffect(() => {
    return () => {
      if (ws) {
        ws.close();
      }
    };
  }, [ws]);

  // Remove old progress simulation useEffect and replace with:
  useEffect(() => {
    if (currentJobId && podcastLoading) {
      const websocket = apiClient.connectProgress(
        currentJobId,
        (update: ProgressUpdate) => {
          setProgressInfo({
            current: update.current,
            total: update.total,
            message: update.message || "",
          });

          // Check if completed - either message contains completion text or progress reached 100%
          const isCompleted =
            update.message?.includes("completed") ||
            update.message?.includes("completed") ||
            (update.current >= update.total && update.total > 0);

          if (isCompleted) {
            setPodcastLoading(false);
            setGenerationStartTime(null); // Reset timer on completion
            setStep(2);
            notifications.show({
              title: "Done!",
              message: "Podcast successfully generated",
              color: "green",
              icon: <IconCheck size={16} />,
            });

            // Push notification when completed
            NotificationManager.showPodcastComplete().catch((error) => {
              console.error("Error sending push notification:", error);
            });
          }
        }
      );

      setWs(websocket);

      return () => {
        websocket.close();
      };
    }
  }, [currentJobId, podcastLoading]);

  const handleGenerateScript = async (values: typeof form.values) => {
    setScriptLoading(true);
    setError(null);

    try {
      const request = {
        userPrompt: values.userPrompt,
        ...(values.systemPrompt && { systemPrompt: values.systemPrompt }),
      };

      const response = await apiClient.generateScript(request);
      setScriptResult(response);

      // Substitute generated script into JSON field
      form.setFieldValue(
        "conversationJson",
        JSON.stringify(response.conversation, null, 2)
      );

      setStep(1); // Move to next step

      notifications.show({
        title: "Success!",
        message: `Script generated: ${response.messageCount} replies`,
        color: "green",
        icon: <IconCheck size={16} />,
      });
    } catch (err) {
      const errorMessage =
        err instanceof Error ? err.message : "Unknown error";
      setError(errorMessage);

      notifications.show({
        title: "Error",
        message: errorMessage,
        color: "red",
        icon: <IconX size={16} />,
      });
    } finally {
      setScriptLoading(false);
    }
  };

  const handleGeneratePodcast = async (values: typeof form.values) => {
    setPodcastLoading(true);
    setError(null);
    setGenerationStartTime(Date.now()); // Start timer
    setElapsedTime(0);

    try {
      const conversationData = JSON.parse(values.conversationJson);
      console.log("Sending data to server:", conversationData);
      console.log("Number of replies:", conversationData.conversation?.length);

      setProgressInfo({
        current: 0,
        total: conversationData.conversation.length,
        message: "Starting generation...",
      });

      const request = { conversationData };
      const response = await apiClient.generatePodcast(request);
      setPodcastResult(response);
      setCurrentJobId(response.jobId);

      notifications.show({
        title: "Success!",
        message: `Podcast generation started: ${response.messageCount} replies`,
        color: "green",
        icon: <IconCheck size={16} />,
      });
    } catch (err) {
      const errorMessage =
        err instanceof Error ? err.message : "Unknown error";
      setError(errorMessage);
      setPodcastLoading(false);
      setGenerationStartTime(null); // Reset timer on error

      notifications.show({
        title: "Error",
        message: errorMessage,
        color: "red",
        icon: <IconX size={16} />,
      });
    }
  };

  const handleGeneratePodcastClick = () => {
    // Validate only the JSON field
    const jsonValue = form.values.conversationJson;
    if (!jsonValue.trim()) {
      notifications.show({
        title: "Error",
        message: "Script is required to generate a podcast",
        color: "red",
        icon: <IconX size={16} />,
      });
      return;
    }

    try {
      JSON.parse(jsonValue);
    } catch {
      notifications.show({
        title: "Error", 
        message: "Invalid JSON format",
        color: "red",
        icon: <IconX size={16} />,
      });
      return;
    }

    handleGeneratePodcast(form.values);
  };

  const downloadJson = () => {
    if (!scriptResult) return;

    const dataStr = JSON.stringify(scriptResult.conversation, null, 2);
    const dataBlob = new Blob([dataStr], { type: "application/json" });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement("a");
    link.href = url;
    link.download = scriptResult.filename;
    link.click();
    URL.revokeObjectURL(url);
  };

  const resetWorkflow = () => {
    if (ws) {
      ws.close();
    }
    setStep(0);
    setScriptResult(null);
    setPodcastResult(null);
    setError(null);
    setProgressInfo({ current: 0, total: 0, message: "" });
    setCurrentJobId(null);
    setWs(null);
    setGenerationStartTime(null);
    setElapsedTime(0);
    setReplicasCount(0);
    setShowAudioPlayer(false);
    form.reset();
    form.setFieldValue("systemPrompt", defaultSystemPrompt);
  };

  const handleSkipScript = () => {
    setStep(1); // Move to next step
    notifications.show({
      title: "Skipped",
      message: "Script generation skipped. You can manually enter JSON in the next step.",
      color: "blue",
      icon: <IconPlayerSkipForward size={16} />,
    });
  };

  return (
    <Stack gap="lg">
      <div>
        <Text size="sm" c="dimmed">
          Creating podcasts in three steps
        </Text>
        {generationData && (
          <Alert color="blue" mt="xs">
            <Text size="sm">
              <strong>Prompt applied:</strong> {generationData.promptName}
            </Text>
          </Alert>
        )}
      </div>

      <Stepper active={step} onStepClick={setStep} allowNextStepsSelect={true}>
        <Stepper.Step label="Generating script" description="Creating dialogue">
          <Space h="md" />

          {scriptLoading && (
            <Paper p="md" withBorder mb="md">
              <Group>
                <Loader size="sm" />
                <Text>Generating podcast script...</Text>
              </Group>
            </Paper>
          )}

          <Paper p="md" withBorder>
            <Stack gap="md">
              <Group justify="flex-end">
                <Button
                  variant="light"
                  onClick={handleSkipScript}
                  leftSection={<IconPlayerSkipForward size={16} />}
                >
                  Skip
                </Button>
                <Button
                  onClick={() => form.onSubmit(handleGenerateScript)()}
                  loading={scriptLoading}
                  leftSection={<IconScript size={16} />}
                >
                  Generate script
                </Button>  
              </Group>
              <Textarea
                label="Podcast topic"
                placeholder="Describe the topic of the podcast..."
                required
                minRows={5}
                maxRows={24}
                autosize
                resize="vertical"
                {...form.getInputProps("userPrompt")}
              />

              <Select
                label="Podcast style"
                description="Select the podcast style"
                placeholder={promptsLoading ? "Loading prompts..." : "Select prompt"}
                data={availablePrompts.map(prompt => ({
                  value: prompt.name,
                  label: prompt.displayName,
                  disabled: false
                }))}
                value={selectedPrompt}
                onChange={(value) => setSelectedPrompt(value || "default")}
                disabled={promptsLoading}
                searchable
                clearable={false}
                renderOption={({ option }) => {
                  const prompt = availablePrompts.find(p => p.name === option.value);
                  return (
                    <div>
                      <Text size="sm" fw={500}>{option.label}</Text>
                      {prompt && (
                        <Text size="xs" c="dimmed">{prompt.description}</Text>
                      )}
                    </div>
                  );
                }}
              />

              <Textarea
                label="System prompt"
                placeholder="Loaded automatically based on selected style..."
                minRows={6}
                maxRows={18}
                autosize
                resize="vertical"
                {...form.getInputProps("systemPrompt")}
              />
            </Stack>
          </Paper>

          {scriptResult && (
            <Paper p="md" withBorder mt="md">
              <Stack gap="md">
                <Group justify="space-between">
                  <Title order={4}>Script generated</Title>
                  <Button
                    variant="light"
                    size="sm"
                    leftSection={<IconDownload size={16} />}
                    onClick={downloadJson}
                  >
                    Download JSON
                  </Button>
                </Group>

                <Group gap="xs">
                  <Badge color="green" variant="light">
                    {scriptResult.messageCount} replies
                  </Badge>
                  <Badge color="blue" variant="light">
                    {scriptResult.filename}
                  </Badge>
                </Group>

                <div>
                  <Text size="sm" c="dimmed" mb="xs">
                    Podcast name:
                  </Text>
                  <Text fw={500}>{scriptResult.conversation.podcast_name}</Text>
                </div>

                <div>
                  <Text size="sm" c="dimmed" mb="xs">
                    Preview (first 3 replies):
                  </Text>
                  <Stack gap="xs">
                    {scriptResult.conversation.conversation
                      .slice(0, 3)
                      .map((message) => (
                        <Paper key={message.id} p="xs" bg="gray.0">
                          <Group gap="xs">
                            <Badge size="xs" variant="outline">
                              {message.speaker}
                            </Badge>
                            <Text size="sm">{message.text}</Text>
                          </Group>
                        </Paper>
                      ))}
                    {scriptResult.conversation.conversation.length > 3 && (
                      <Text size="sm" c="dimmed" ta="center">
                        ... and {scriptResult.conversation.conversation.length - 3} more
                      </Text>
                    )}
                  </Stack>
                </div>
              </Stack>
            </Paper>
          )}
        </Stepper.Step>

        <Stepper.Step label="Generating podcast" description="Creating audio">
          <Space h="md" />

          <Paper p="md" withBorder>
            <Stack gap="md">
              {podcastLoading && (
                <Paper p="md" withBorder mt="md">
                  <Stack gap="md">
                    <Group>
                      <Loader size="sm" />
                      <Text>Generating podcast audio...</Text>
                      {generationStartTime && (
                        <Badge color="blue" variant="light">
                          {formatElapsedTime(elapsedTime)}
                        </Badge>
                      )}
                    </Group>

                    {progressInfo.total > 0 && (
                      <Stack gap="xs">
                        <Group justify="space-between">
                          <Text size="sm">Generation progress:</Text>
                          <Text size="sm">
                            {progressInfo.current} out of {progressInfo.total} pieces
                          </Text>
                        </Group>
                        <Progress
                          value={
                            (progressInfo.current / progressInfo.total) * 100
                          }
                          size="lg"
                          animated
                        />
                        {progressInfo.message && (
                          <Text size="xs" c="dimmed">
                            {progressInfo.message}
                          </Text>
                        )}
                      </Stack>
                    )}
                  </Stack>
                </Paper>
              )}
              <Stack gap="md">
                <Group align="flex-end" gap="md">
                  <div style={{ flex: 1 }}>
                    <JsonInput
                      label="Script (JSON)"
                      placeholder="Script will be substituted automatically after generation"
                      required
                      minRows={10}
                      maxRows={16}
                      autosize
                      resize="vertical"
                      {...form.getInputProps("conversationJson")}
                    />
                  </div>
                  {replicasCount > 0 && (
                    <Badge
                      size="lg"
                      color="grape"
                      variant="gradient"
                      style={{ alignSelf: "flex-start", marginTop: "28px" }}
                    >
                      {replicasCount} replies
                    </Badge>
                  )}
                </Group>
              </Stack>

              <Group justify="flex-end">
                <Button
                  onClick={handleGeneratePodcastClick}
                  loading={podcastLoading}
                  leftSection={<IconPlayerPlay size={16} />}
                >
                  Generate podcast
                </Button>
              </Group>
            </Stack>
          </Paper>
        </Stepper.Step>

        <Stepper.Step label="Done" description="Podcast created">
          <Space h="md" />

          {podcastResult && (
            <Paper p="md" withBorder>
              <Stack gap="md">
                <Group justify="space-between">
                  <Title order={4}>🎉 Podcast ready!</Title>
                  <Button onClick={resetWorkflow} variant="light">
                    Create new
                  </Button>
                </Group>

                <Group gap="xs">
                  <Badge color="blue" variant="light">
                    {podcastResult.status}
                  </Badge>
                  <Badge color="green" variant="light">
                    {podcastResult.messageCount} replies
                  </Badge>
                </Group>

                <div>
                  <Text size="sm" c="dimmed" mb="xs">
                    Podcast name:
                  </Text>
                  <Text fw={500}>{podcastResult.podcast_name}</Text>
                </div>

                <div>
                  <Text size="sm" c="dimmed" mb="xs">
                    Output file:
                  </Text>
                  <Text fw={500}>{podcastResult.filename}</Text>
                </div>

                {/* Audio player and buttons */}
                <Divider label="Playing and downloading" />

                <Stack gap="md">
                  {showAudioPlayer ? (
                    <div>
                      <Text size="sm" c="dimmed" mb="xs">
                        Audio player:
                      </Text>
                      <audio
                        controls
                        style={{ width: "100%" }}
                        src={`http://localhost:3000/output/${podcastResult.filename}/${podcastResult.filename}`}
                        preload="metadata"
                        onError={(e) => {
                          console.error("Audio loading error:", e);
                          notifications.show({
                            title: "Error loading audio",
                            message:
                              "Failed to load audio file. Please try downloading the file.",
                            color: "orange",
                          });
                        }}
                      >
                        Your browser does not support the audio element.
                      </audio>
                    </div>
                  ) : (
                    <div>
                      <Text size="sm" c="dimmed" mb="xs">
                        Audio player:
                      </Text>
                      <Paper p="md" withBorder>
                        <Group>
                          <Loader size="sm" />
                          <Text size="sm" c="dimmed">
                            Loading audio player...
                          </Text>
                        </Group>
                      </Paper>
                    </div>
                  )}

                  <Group>
                    <Button
                      onClick={() => {
                        const downloadUrl = `http://localhost:3000/download/${podcastResult.filename}/${podcastResult.filename}`;
                        const link = document.createElement("a");
                        link.href = downloadUrl;
                        link.target = "_blank";
                        document.body.appendChild(link);
                        link.click();
                        document.body.removeChild(link);
                      }}
                      leftSection={<IconDownload size={16} />}
                      variant="filled"
                    >
                      Download podcast
                    </Button>

                    <Button
                      onClick={() =>
                        window.open(
                          `http://localhost:3000/output/${podcastResult.filename}/${podcastResult.filename}`,
                          "_blank"
                        )
                      }
                      leftSection={<IconPlayerPlay size={16} />}
                      variant="outline"
                    >
                      Open in new tab
                    </Button>
                  </Group>
                </Stack>

                <Alert color="green" title="Done!">
                  Podcast successfully generated and saved in the output folder on
                  the server. You can play it here or download it to your device.
                </Alert>
              </Stack>
            </Paper>
          )}
        </Stepper.Step>
      </Stepper>

      {error && (
        <Alert color="red" title="Error" icon={<IconX size={16} />} mt="md">
          {error}
        </Alert>
      )}
    </Stack>
  );
}
