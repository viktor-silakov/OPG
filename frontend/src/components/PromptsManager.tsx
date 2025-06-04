import { useState, useEffect } from "react";
import {
  Container,
  Card,
  Text,
  Button,
  Group,
  Stack,
  Alert,
  Progress,
  Badge,
  ActionIcon,
  Modal,
  TextInput,
  Textarea,
  Grid,
  Box,
  Chip,
} from "@mantine/core";
import {
  IconFileText,
  IconEdit,
  IconTrash,
  IconPlus,
  IconPlayerPlay,
  IconAlertCircle,
  IconCheck,
  IconFilter,
  IconCopy,
} from "@tabler/icons-react";
import { notifications } from "@mantine/notifications";
import { 
  apiClient, 
  type PromptInfo, 
  type SavePromptRequest,
} from "../api/client";

interface PromptEditData {
  name: string;
  content: string;
  isNewPrompt: boolean;
}

interface PodcastGenerationData {
  promptName: string;
  userPrompt: string;
  systemPrompt: string;
}

interface PromptsManagerProps {
  onNavigateToGeneration: (data: PodcastGenerationData) => void;
}

export function PromptsManager({ onNavigateToGeneration }: PromptsManagerProps) {
  const [prompts, setPrompts] = useState<PromptInfo[]>([]);
  const [filteredPrompts, setFilteredPrompts] = useState<PromptInfo[]>([]);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [isEditModalOpen, setIsEditModalOpen] = useState(false);
  const [isDeleteConfirmOpen, setIsDeleteConfirmOpen] = useState(false);
  const [isGenerateModalOpen, setIsGenerateModalOpen] = useState(false);
  const [promptToDelete, setPromptToDelete] = useState<string | null>(null);
  const [selectedLanguages, setSelectedLanguages] = useState<string[]>([]);
  const [selectedStyles, setSelectedStyles] = useState<string[]>([]);
  const [availableLanguages, setAvailableLanguages] = useState<string[]>([]);
  const [availableStyles, setAvailableStyles] = useState<string[]>([]);
  const [selectedPromptForGeneration, setSelectedPromptForGeneration] = useState<string | null>(null);
  const [generationUserPrompt, setGenerationUserPrompt] = useState("");

  const [editData, setEditData] = useState<PromptEditData>({
    name: "",
    content: "",
    isNewPrompt: false,
  });

  // Extract language and style from prompt name (e.g., "RU_default" -> language: "RU", style: "default")
  const parsePromptName = (promptName: string): { language: string; style: string } => {
    const parts = promptName.split('_');
    if (parts.length >= 2) {
      return {
        language: parts[0],
        style: parts.slice(1).join('_')
      };
    }
    return {
      language: 'OTHER',
      style: promptName
    };
  };

  // Get language display name
  const getLanguageDisplayName = (prefix: string): string => {
    const languageMap: { [key: string]: string } = {
      'EN': '🇺🇸 English',
      'RU': '🇷🇺 Russian',
      'ES': '🇪🇸 Spanish',
      'FR': '🇫🇷 French',
      'DE': '🇩🇪 German',
      'IT': '🇮🇹 Italian',
      'PT': '🇵🇹 Portuguese',
      'JA': '🇯🇵 Japanese',
      'KO': '🇰🇷 Korean',
      'ZH': '🇨🇳 Chinese',
      'OTHER': '🌐 Other'
    };
    return languageMap[prefix] || `${prefix} Unknown`;
  };

  // Get style display name
  const getStyleDisplayName = (style: string): string => {
    const styleMap: { [key: string]: string } = {
      'default': '💬 Default',
      'business': '💼 Business',
      'comedy': '😄 Comedy',
      'scientific': '🔬 Scientific',
      'storytelling': '📚 Storytelling',
      'debate': '⚡ Debate',
      'goblin': '👹 Goblin',
      'default_live': '🎙️ Live',
      'default_old': '📜 Old Default',
      'test': '🧪 Test'
    };
    return styleMap[style] || `✨ ${style}`;
  };

  // Update available filters and apply filtering
  const updateFiltersAndApply = (promptsList: PromptInfo[]) => {
    const languages = [...new Set(promptsList.map(prompt => parsePromptName(prompt.name).language))];
    const styles = [...new Set(promptsList.map(prompt => parsePromptName(prompt.name).style))];
    
    setAvailableLanguages(languages);
    setAvailableStyles(styles);

    // Apply filters
    let filtered = promptsList;
    
    if (selectedLanguages.length > 0) {
      filtered = filtered.filter(prompt => 
        selectedLanguages.includes(parsePromptName(prompt.name).language)
      );
    }
    
    if (selectedStyles.length > 0) {
      filtered = filtered.filter(prompt => 
        selectedStyles.includes(parsePromptName(prompt.name).style)
      );
    }
    
    setFilteredPrompts(filtered);
  };

  // Handle language filter change
  const handleLanguageFilterChange = (languages: string[]) => {
    setSelectedLanguages(languages);
    updateFiltersAndApply(prompts);
  };

  // Handle style filter change
  const handleStyleFilterChange = (styles: string[]) => {
    setSelectedStyles(styles);
    updateFiltersAndApply(prompts);
  };

  // Load prompts list
  const loadPrompts = async () => {
    setLoading(true);
    try {
      const response = await apiClient.getPrompts();
      setPrompts(response.prompts);
      updateFiltersAndApply(response.prompts);
    } catch (error) {
      console.error("Error loading prompts:", error);
      notifications.show({
        title: "Error",
        message: "Failed to load prompts list",
        color: "red",
        icon: <IconAlertCircle />,
      });
    } finally {
      setLoading(false);
    }
  };

  // Open edit modal for existing prompt
  const openEditPrompt = async (promptName: string) => {
    try {
      const response = await apiClient.getPromptContent(promptName);
      setEditData({
        name: promptName,
        content: response.content,
        isNewPrompt: false,
      });
      setIsEditModalOpen(true);
    } catch (error) {
      console.error("Error loading prompt content:", error);
      notifications.show({
        title: "Error",
        message: "Failed to load prompt content",
        color: "red",
        icon: <IconAlertCircle />,
      });
    }
  };

  // Open edit modal for new prompt
  const openNewPrompt = () => {
    setEditData({
      name: "",
      content: "",
      isNewPrompt: true,
    });
    setIsEditModalOpen(true);
  };

  // Save prompt
  const savePrompt = async () => {
    if (!editData.name || !editData.content) {
      notifications.show({
        title: "Error",
        message: "Fill in all fields",
        color: "red",
        icon: <IconAlertCircle />,
      });
      return;
    }

    setSaving(true);

    try {
      const request: SavePromptRequest = {
        name: editData.name,
        content: editData.content,
      };

      await apiClient.savePrompt(request);
      
      notifications.show({
        title: "Success",
        message: `Prompt "${editData.name}" saved successfully`,
        color: "green",
        icon: <IconCheck />,
      });

      // Reset form and close modal
      setEditData({
        name: "",
        content: "",
        isNewPrompt: false,
      });
      setIsEditModalOpen(false);
      
      // Refresh prompts list
      loadPrompts();
    } catch (error) {
      console.error("Error saving prompt:", error);
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

  // Open delete confirmation dialog
  const openDeleteConfirm = (promptName: string) => {
    setPromptToDelete(promptName);
    setIsDeleteConfirmOpen(true);
  };

  // Confirm prompt deletion
  const confirmDeletePrompt = async () => {
    if (!promptToDelete) return;

    try {
      await apiClient.deletePrompt(promptToDelete);
      
      notifications.show({
        title: "Success",
        message: `Prompt "${promptToDelete}" deleted`,
        color: "green",
        icon: <IconCheck />,
      });
      loadPrompts();
    } catch (error) {
      console.error("Error deleting prompt:", error);
      notifications.show({
        title: "Error",
        message: "Failed to delete prompt",
        color: "red",
        icon: <IconAlertCircle />,
      });
    } finally {
      setIsDeleteConfirmOpen(false);
      setPromptToDelete(null);
    }
  };

  // Open generation modal
  const openGenerationModal = (promptName: string) => {
    setSelectedPromptForGeneration(promptName);
    setGenerationUserPrompt("");
    setIsGenerateModalOpen(true);
  };

  // Navigate to podcast generation with prompt data
  const handleNavigateToPodcastGeneration = async () => {
    if (!selectedPromptForGeneration || !generationUserPrompt) {
      notifications.show({
        title: "Error",
        message: "Fill in the content for podcast generation",
        color: "red",
        icon: <IconAlertCircle />,
      });
      return;
    }

    try {
      // Get the prompt content
      const promptResponse = await apiClient.getPromptContent(selectedPromptForGeneration);
      
      // Navigate to podcast generation page with data
      onNavigateToGeneration({
        promptName: selectedPromptForGeneration,
        userPrompt: generationUserPrompt,
        systemPrompt: promptResponse.content
      });

      notifications.show({
        title: "Redirected",
        message: `Redirected to podcast generation with prompt "${selectedPromptForGeneration}"`,
        color: "blue",
        icon: <IconPlayerPlay />,
      });

      // Reset and close modal
      setSelectedPromptForGeneration(null);
      setGenerationUserPrompt("");
      setIsGenerateModalOpen(false);
      
    } catch (error) {
      console.error("Error loading prompt for generation:", error);
      notifications.show({
        title: "Error",
        message: error instanceof Error ? error.message : "Unknown error",
        color: "red",
        icon: <IconAlertCircle />,
      });
    }
  };

  // Copy prompt content to clipboard
  const copyPromptContent = async (promptName: string) => {
    try {
      const response = await apiClient.getPromptContent(promptName);
      await navigator.clipboard.writeText(response.content);
      notifications.show({
        title: "Copied",
        message: `Prompt "${promptName}" copied to clipboard`,
        color: "blue",
        icon: <IconCopy />,
      });
    } catch (error) {
      console.error("Error copying prompt:", error);
      notifications.show({
        title: "Error",
        message: "Failed to copy prompt",
        color: "red",
        icon: <IconAlertCircle />,
      });
    }
  };

  useEffect(() => {
    loadPrompts();
  }, []);

  // Update filtered prompts when filters change
  useEffect(() => {
    updateFiltersAndApply(prompts);
  }, [selectedLanguages, selectedStyles, prompts]);

  return (
    <Container size="lg">
      <Stack gap="md">
        <Group justify="space-between" align="center">
          <Button
            leftSection={<IconPlus size={16} />}
            onClick={openNewPrompt}
            variant="filled"
            color="blue"
          >
            Add prompt
          </Button>
        </Group>

        <Alert
          icon={<IconAlertCircle />}
          title="Info"
          variant="light"
          color="blue"
        >
          Here you can manage podcast prompts. Create, edit, and generate podcasts using different styles and languages.
        </Alert>

        {/* Filters */}
        {(availableLanguages.length > 1 || availableStyles.length > 1) && (
          <Card withBorder>
            <Stack gap="sm">
              <Group align="center">
                <IconFilter size={16} />
                <Text fw={500}>Filters:</Text>
              </Group>
              
              {availableLanguages.length > 1 && (
                <div>
                  <Text size="sm" fw={500} mb="xs">Languages:</Text>
                  <Chip.Group 
                    multiple 
                    value={selectedLanguages} 
                    onChange={handleLanguageFilterChange}
                  >
                    <Group gap="xs">
                      {availableLanguages.map((lang) => (
                        <Chip key={lang} value={lang} variant="light">
                          {getLanguageDisplayName(lang)}
                        </Chip>
                      ))}
                    </Group>
                  </Chip.Group>
                </div>
              )}

              {availableStyles.length > 1 && (
                <div>
                  <Text size="sm" fw={500} mb="xs">Styles:</Text>
                  <Chip.Group 
                    multiple 
                    value={selectedStyles} 
                    onChange={handleStyleFilterChange}
                  >
                    <Group gap="xs">
                      {availableStyles.map((style) => (
                        <Chip key={style} value={style} variant="light">
                          {getStyleDisplayName(style)}
                        </Chip>
                      ))}
                    </Group>
                  </Chip.Group>
                </div>
              )}

              {(selectedLanguages.length > 0 || selectedStyles.length > 0) && (
                <Text size="sm" c="dimmed">
                  Showing {filteredPrompts.length} of {prompts.length} prompts
                </Text>
              )}
            </Stack>
          </Card>
        )}

        {loading ? (
          <Progress value={100} animated />
        ) : (
          <Grid>
            {filteredPrompts.map((prompt) => {
              const { language, style } = parsePromptName(prompt.name);
              return (
                <Grid.Col span={{ base: 12, md: 6, lg: 4 }} key={prompt.name}>
                  <Card shadow="sm" padding="lg" radius="md" withBorder>
                    <Stack gap="xs">
                      <Group justify="space-between" align="flex-start">
                        <div>
                          <Group gap="xs" align="center">
                            <Text fw={500} size="lg">
                              {prompt.name}
                            </Text>
                          </Group>
                          <Group gap="xs" mt="xs">
                            <Badge 
                              variant="dot" 
                              color="blue" 
                              size="xs"
                            >
                              {getLanguageDisplayName(language)}
                            </Badge>
                            <Badge 
                              variant="dot" 
                              color="green" 
                              size="xs"
                            >
                              {getStyleDisplayName(style)}
                            </Badge>
                          </Group>
                        </div>
                      </Group>

                      <Text size="xs" c="dimmed">
                        Size: {Math.round(prompt.size / 1024)} KB • Modified: {new Date(prompt.modified).toLocaleDateString()}
                      </Text>

                      <Group justify="space-between" mt="md">
                        <Group gap="xs">
                          <ActionIcon
                            variant="light"
                            color="blue"
                            onClick={() => openEditPrompt(prompt.name)}
                            title="Edit prompt"
                          >
                            <IconEdit size={16} />
                          </ActionIcon>

                          <ActionIcon
                            variant="light"
                            color="green"
                            onClick={() => openGenerationModal(prompt.name)}
                            title="Generate podcast"
                          >
                            <IconPlayerPlay size={16} />
                          </ActionIcon>

                          <ActionIcon
                            variant="light"
                            color="gray"
                            onClick={() => copyPromptContent(prompt.name)}
                            title="Copy to clipboard"
                          >
                            <IconCopy size={16} />
                          </ActionIcon>
                        </Group>

                        <ActionIcon
                          variant="light"
                          color="red"
                          onClick={() => openDeleteConfirm(prompt.name)}
                          title="Delete prompt"
                        >
                          <IconTrash size={16} />
                        </ActionIcon>
                      </Group>
                    </Stack>
                  </Card>
                </Grid.Col>
              );
            })}
          </Grid>
        )}

        {filteredPrompts.length === 0 && !loading && prompts.length > 0 && (
          <Box ta="center" py="xl">
            <IconFilter size={48} stroke={1.5} color="gray" />
            <Text size="lg" fw={500} mt="md">
              No prompts for selected filters
            </Text>
            <Text size="sm" c="dimmed">
              Change the filter or add new prompts
            </Text>
          </Box>
        )}

        {prompts.length === 0 && !loading && (
          <Box ta="center" py="xl">
            <IconFileText size={48} stroke={1.5} color="gray" />
            <Text size="lg" fw={500} mt="md">
              No prompts found
            </Text>
            <Text size="sm" c="dimmed">
              Add a new prompt to get started
            </Text>
          </Box>
        )}
      </Stack>

      {/* Edit/Create Prompt Modal */}
      <Modal
        opened={isEditModalOpen}
        onClose={() => setIsEditModalOpen(false)}
        title={editData.isNewPrompt ? "Create new prompt" : `Edit prompt: ${editData.name}`}
        size="xl"
        styles={{
          body: {
            paddingBottom: "80px", // Add space for sticky buttons
          }
        }}
      >
        <Stack gap="md">
          {editData.isNewPrompt && (
            <TextInput
              label="Prompt name"
              placeholder="e.g., EN_comedy or RU_business"
              value={editData.name}
              onChange={(event) =>
                setEditData((prev) => ({
                  ...prev,
                  name: event.target.value,
                }))
              }
              description="Use language prefix (EN_, RU_) and style name"
              required
            />
          )}

          <Textarea
            label="Prompt content"
            placeholder="Enter the prompt content in Markdown format..."
            value={editData.content}
            onChange={(event) =>
              setEditData((prev) => ({
                ...prev,
                content: event.target.value,
              }))
            }
            minRows={15}
            maxRows={25}
            autosize
            required
          />

          {saving && (
            <Progress value={100} size="lg" animated />
          )}

          {/* Sticky buttons container */}
          <div
            style={{
              position: "sticky",
              bottom: 0,
              backgroundColor: "white",
              borderTop: "1px solid var(--mantine-color-gray-3)",
              padding: "16px 0",
              marginTop: "16px",
              zIndex: 100,
            }}
          >
            <Group justify="flex-end">
              <Button
                variant="light"
                onClick={() => setIsEditModalOpen(false)}
                disabled={saving}
              >
                Cancel
              </Button>
              <Button
                onClick={savePrompt}
                loading={saving}
                leftSection={<IconCheck size={16} />}
              >
                {editData.isNewPrompt ? "Create" : "Save"}
              </Button>
            </Group>
          </div>
        </Stack>
      </Modal>

      {/* Delete Confirmation Modal */}
      <Modal
        opened={isDeleteConfirmOpen}
        onClose={() => setIsDeleteConfirmOpen(false)}
        title="Delete confirmation"
        size="md"
        styles={{
          body: {
            paddingBottom: "80px", // Add space for sticky buttons
          }
        }}
      >
        <Stack gap="md">
          <Alert
            icon={<IconAlertCircle />}
            title="Warning!"
            color="red"
            variant="light"
          >
            Are you sure you want to delete the prompt "{promptToDelete}"?
            <br />
            <strong>This action cannot be undone.</strong>
          </Alert>
          
          {/* Sticky buttons container */}
          <div
            style={{
              position: "sticky",
              bottom: 0,
              backgroundColor: "white",
              borderTop: "1px solid var(--mantine-color-gray-3)",
              padding: "16px 0",
              marginTop: "16px",
              zIndex: 100,
            }}
          >
            <Group justify="flex-end">
              <Button
                variant="light"
                onClick={() => setIsDeleteConfirmOpen(false)}
              >
                Cancel
              </Button>
              <Button
                color="red"
                onClick={confirmDeletePrompt}
                leftSection={<IconTrash size={16} />}
              >
                Delete
              </Button>
            </Group>
          </div>
        </Stack>
      </Modal>

      {/* Generate Podcast Modal */}
      <Modal
        opened={isGenerateModalOpen}
        onClose={() => setIsGenerateModalOpen(false)}
        title={`Generate podcast using: ${selectedPromptForGeneration}`}
        size="lg"
        styles={{
          body: {
            paddingBottom: "80px", // Add space for sticky buttons
          }
        }}
      >
        <Stack gap="md">
          <Alert
            icon={<IconPlayerPlay />}
            title="Podcast Generation"
            variant="light"
            color="blue"
          >
            Enter the content or topic you want to create a podcast about. You will be redirected to the main podcast generation page.
          </Alert>

          <Textarea
            label="Content for podcast"
            placeholder="Enter text, article, or topic to generate podcast about..."
            value={generationUserPrompt}
            onChange={(event) => setGenerationUserPrompt(event.target.value)}
            minRows={8}
            maxRows={15}
            autosize
            required
          />

          {/* Sticky buttons container */}
          <div
            style={{
              position: "sticky",
              bottom: 0,
              backgroundColor: "white",
              borderTop: "1px solid var(--mantine-color-gray-3)",
              padding: "16px 0",
              marginTop: "16px",
              zIndex: 100,
            }}
          >
            <Group justify="flex-end">
              <Button
                variant="light"
                onClick={() => setIsGenerateModalOpen(false)}
              >
                Cancel
              </Button>
              <Button
                onClick={handleNavigateToPodcastGeneration}
                leftSection={<IconPlayerPlay size={16} />}
              >
                Go to Podcast Generation
              </Button>
            </Group>
          </div>
        </Stack>
      </Modal>
    </Container>
  );
} 