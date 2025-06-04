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
  FileInput,
  Grid,
  Box,
  Chip,
} from "@mantine/core";
import {
  IconUpload,
  IconMicrophone,
  IconTrash,
  IconDownload,
  IconAlertCircle,
  IconCheck,
  IconPlayerPlay,
  IconPlayerStop,
  IconFilter,
} from "@tabler/icons-react";
import { notifications } from "@mantine/notifications";
import { apiClient, type Voice, type CreateVoiceRequest } from "../api/client";

interface VoiceUploadData {
  name: string;
  audioFile: File | null;
  referenceText: string;
}

export function VoiceManager() {
  const [voices, setVoices] = useState<Voice[]>([]);
  const [filteredVoices, setFilteredVoices] = useState<Voice[]>([]);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [playingAudio, setPlayingAudio] = useState<string | null>(null);
  const [generatingAudio, setGeneratingAudio] = useState<string | null>(null);
  const [isDeleteConfirmOpen, setIsDeleteConfirmOpen] = useState(false);
  const [voiceToDelete, setVoiceToDelete] = useState<string | null>(null);
  const [selectedLanguages, setSelectedLanguages] = useState<string[]>([]);
  const [availableLanguages, setAvailableLanguages] = useState<string[]>([]);

  const [uploadData, setUploadData] = useState<VoiceUploadData>({
    name: "",
    audioFile: null,
    referenceText: "",
  });

  // Extract language prefix from voice name
  const getLanguagePrefix = (voiceName: string): string => {
    const match = voiceName.match(/^([A-Z]{2})_/);
    return match ? match[1] : 'OTHER';
  };

  // Get language display name
  const getLanguageDisplayName = (prefix: string): string => {
    const languageMap: { [key: string]: string } = {
      // Main world languages by prevalence
      'EN': '🇺🇸 English',
      'ZH': '🇨🇳 中文',
      'HI': '🇮🇳 हिन्दी',
      'ES': '🇪🇸 Español', 
      'FR': '🇫🇷 Français',
      'AR': '🇸🇦 العربية',
      'BN': '🇧🇩 বাংলা',
      'PT': '🇵🇹 Português',
      'RU': '🇷🇺 Russian',
      'ID': '🇮🇩 Bahasa Indonesia',
      'UR': '🇵🇰 اردو',
      'DE': '🇩🇪 Deutsch',
      'JA': '🇯🇵 日本語',
      'SW': '🇰🇪 Kiswahili',
      'TR': '🇹🇷 Türkçe',
      'VI': '🇻🇳 Tiếng Việt',
      'IT': '🇮🇹 Italiano',
      'KO': '🇰🇷 한국어',
      'TH': '🇹🇭 ไทย',
      'NL': '🇳🇱 Nederlands',
      'PL': '🇵🇱 Polski',
      'SV': '🇸🇪 Svenska',
      'NO': '🇳🇴 Norsk',
      'DA': '🇩🇰 Dansk',
      'FI': '🇫🇮 Suomi',
      'CS': '🇨🇿 Čeština',
      'HU': '🇭🇺 Magyar',
      'RO': '🇷🇴 Română',
      'EL': '🇬🇷 Ελληνικά',
      'HE': '🇮🇱 עברית',
      
      // Regional languages of the post-Soviet space
      'BY': '🇧🇾 Belarusian', 
      'UA': '🇺🇦 Ukrainian',
      'KZ': '🇰🇿 Kazakh',
      'UZ': '🇺🇿 Uzbek',
      'KY': '🇰🇬 Kyrgyz',
      'TJ': '🇹🇯 Tajik',
      'AZ': '🇦🇿 Azerbaijani',
      'AM': '🇦🇲 Armenian',
      'GE': '🇬🇪 Georgian',
      'LV': '🇱🇻 Latvian',
      'LT': '🇱🇹 Lithuanian',
      'EE': '🇪🇪 Estonian',
      'MD': '🇲🇩 Romanian',
      
      // Other
      'OTHER': '🌐 Other'
    };
    return languageMap[prefix] || `${prefix} Unknown`;
  };

  // Update available languages and filter voices
  const updateLanguagesAndFilter = (voicesList: Voice[]) => {
    const languages = [...new Set(voicesList.map(voice => getLanguagePrefix(voice.name)))];
    setAvailableLanguages(languages);

    // Apply language filter
    if (selectedLanguages.length === 0) {
      setFilteredVoices(voicesList);
    } else {
      const filtered = voicesList.filter(voice => 
        selectedLanguages.includes(getLanguagePrefix(voice.name))
      );
      setFilteredVoices(filtered);
    }
  };

  // Handle language filter change
  const handleLanguageFilterChange = (languages: string[]) => {
    setSelectedLanguages(languages);
    updateLanguagesAndFilter(voices);
  };

  // Load voices list
  const loadVoices = async () => {
    setLoading(true);
    try {
      const response = await apiClient.getVoices();
      setVoices(response.voices);
      updateLanguagesAndFilter(response.voices);
    } catch (error) {
      console.error("Error loading voices:", error);
      notifications.show({
        title: "Error",
        message: "Failed to load voices list",
        color: "red",
        icon: <IconAlertCircle />,
      });
    } finally {
      setLoading(false);
    }
  };

  // Create new voice
  const createVoice = async () => {
    if (!uploadData.name || !uploadData.audioFile || !uploadData.referenceText) {
      notifications.show({
        title: "Error",
        message: "Fill in all fields",
        color: "red",
        icon: <IconAlertCircle />,
      });
      return;
    }

    setUploading(true);
    setUploadProgress(0);

    try {
      const request: CreateVoiceRequest = {
        name: uploadData.name,
        audioFile: uploadData.audioFile,
        referenceText: uploadData.referenceText,
      };

      await apiClient.createVoice(request);
      
      notifications.show({
        title: "Success",
        message: `Voice "${uploadData.name}" created successfully`,
        color: "green",
        icon: <IconCheck />,
      });

      // Reset form and close modal
      setUploadData({
        name: "",
        audioFile: null,
        referenceText: "",
      });
      setIsModalOpen(false);
      
      // Refresh voices list
      loadVoices();
    } catch (error) {
      console.error("Error creating voice:", error);
      notifications.show({
        title: "Error",
        message: error instanceof Error ? error.message : "Unknown error",
        color: "red",
        icon: <IconAlertCircle />,
      });
    } finally {
      setUploading(false);
      setUploadProgress(0);
    }
  };

  // Open delete confirmation dialog
  const openDeleteConfirm = (voiceName: string) => {
    setVoiceToDelete(voiceName);
    setIsDeleteConfirmOpen(true);
  };

  // Confirm voice deletion
  const confirmDeleteVoice = async () => {
    if (!voiceToDelete) return;

    try {
      await apiClient.deleteVoice(voiceToDelete);
      
      notifications.show({
        title: "Success",
        message: `Voice "${voiceToDelete}" deleted`,
        color: "green",
        icon: <IconCheck />,
      });
      loadVoices();
    } catch (error) {
      console.error("Error deleting voice:", error);
      notifications.show({
        title: "Error",
        message: "Failed to delete voice",
        color: "red",
        icon: <IconAlertCircle />,
      });
    } finally {
      setIsDeleteConfirmOpen(false);
      setVoiceToDelete(null);
    }
  };

  // Play reference text for voice
  const playReferenceText = async (voiceName: string) => {
    setGeneratingAudio(voiceName);
    try {
      // Get full reference text from API
      let referenceText = "This is a test phrase for voice quality check.";
      
      try {
        const textResponse = await apiClient.getVoiceReferenceText(voiceName);
        referenceText = textResponse.referenceText;
      } catch (textError) {
        console.warn("Failed to get reference text, using fallback:", textError);
        // Use preview text as fallback
        const voice = voices.find(v => v.name === voiceName);
        if (voice?.textPreview) {
          referenceText = voice.textPreview;
        }
      }

      const audioBlob = await apiClient.testVoice({
        voiceName,
        text: referenceText,
      });

      const url = URL.createObjectURL(audioBlob);
      const audio = new Audio(url);
      
      setGeneratingAudio(null);
      setPlayingAudio(voiceName);
      audio.play();
      
      audio.onended = () => {
        setPlayingAudio(null);
        URL.revokeObjectURL(url);
      };
    } catch (error) {
      console.error("Voice reference text playback error:", error);
      setGeneratingAudio(null);
      notifications.show({
        title: "Error",
        message: "Failed to play reference text",
        color: "red",
        icon: <IconAlertCircle />,
      });
    }
  };

  // Download voice
  const downloadVoice = async (voiceName: string) => {
    try {
      const downloadUrl = `${import.meta.env.VITE_API_URL || 'http://localhost:8080'}/api/voices/download/${encodeURIComponent(voiceName)}`;
      
      const link = document.createElement('a');
      link.href = downloadUrl;
      link.download = `${voiceName}.wav`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      
      notifications.show({
        title: "Download",
        message: `Started downloading voice "${voiceName}"`,
        color: "blue",
        icon: <IconDownload />,
      });
    } catch (error) {
      console.error("Error downloading voice:", error);
      notifications.show({
        title: "Error",
        message: "Failed to download voice",
        color: "red",
        icon: <IconAlertCircle />,
      });
    }
  };

  useEffect(() => {
    loadVoices();
  }, []);

  // Update filtered voices when selected languages change
  useEffect(() => {
    updateLanguagesAndFilter(voices);
  }, [selectedLanguages, voices]);

  return (
    <Container size="lg">
      <Stack gap="md">
        <Group justify="space-between" align="center">
          {/* <Title order={2}>Voice Management</Title> */}
          <Button
            leftSection={<IconUpload size={16} />}
            onClick={() => setIsModalOpen(true)}
            variant="filled"
            color="blue"
          >
            Add voice
          </Button>
        </Group>

        <Alert
          icon={<IconAlertCircle />}
          title="Info"
          variant="light"
          color="blue"
        >
          Here you can manage voice models. Upload a WAV audio file and the corresponding text to create a new voice.
        </Alert>

        {/* Language Filter */}
        {availableLanguages.length > 1 && (
          <Card withBorder>
            <Stack gap="sm">
              <Group align="center">
                <IconFilter size={16} />
                <Text fw={500}>Language filter:</Text>
              </Group>
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
              {selectedLanguages.length > 0 && (
                <Text size="sm" c="dimmed">
                  Showing {filteredVoices.length} of {voices.length} voices
                </Text>
              )}
            </Stack>
          </Card>
        )}

        {loading ? (
          <Progress value={100} animated />
        ) : (
          <Grid>
            {filteredVoices.map((voice) => (
              <Grid.Col span={{ base: 12, md: 6, lg: 4 }} key={voice.name}>
                <Card shadow="sm" padding="lg" radius="md" withBorder>
                  <Stack gap="xs">
                    <Group justify="space-between" align="flex-start">
                      <div>
                        <Group gap="xs" align="center">
                          <Text fw={500} size="lg">
                            {voice.name}
                          </Text>
                          <Badge 
                            variant="dot" 
                            color="blue" 
                            size="xs"
                          >
                            {getLanguageDisplayName(getLanguagePrefix(voice.name))}
                          </Badge>
                        </Group>
                        {voice.textPreview && (
                          <Text size="sm" c="dimmed" lineClamp={2}>
                            {voice.textPreview}
                          </Text>
                        )}
                      </div>
                      <Group gap="xs">
                        <Badge
                          color={voice.hasTokens ? "green" : "red"}
                          variant="light"
                          size="xs"
                        >
                          {voice.hasTokens ? "✓ Tokens" : "✗ Tokens"}
                        </Badge>
                        <Badge
                          color={voice.hasText ? "green" : "red"}
                          variant="light"
                          size="xs"
                        >
                          {voice.hasText ? "✓ Text" : "✗ Text"}
                        </Badge>
                      </Group>
                    </Group>

                    {voice.audioSize && (
                      <Text size="xs" c="dimmed">
                        Audio size: {voice.audioSize}
                      </Text>
                    )}

                    <Group justify="space-between" mt="md">
                      <Group gap="xs">
                        <ActionIcon
                          variant="light"
                          color="blue"
                          onClick={() => playReferenceText(voice.name)}
                          disabled={!voice.hasTokens || !voice.hasText}
                          loading={generatingAudio === voice.name}
                          title="Play reference text"
                        >
                          {playingAudio === voice.name ? (
                            <IconPlayerStop size={16} />
                          ) : (
                            <IconPlayerPlay size={16} />
                          )}
                        </ActionIcon>

                        <ActionIcon
                          variant="light"
                          color="green"
                          onClick={() => downloadVoice(voice.name)}
                          disabled={!voice.hasAudio}
                          title="Download audio file"
                        >
                          <IconDownload size={16} />
                        </ActionIcon>
                      </Group>

                      <ActionIcon
                        variant="light"
                        color="red"
                        onClick={() => openDeleteConfirm(voice.name)}
                        title="Delete voice"
                      >
                        <IconTrash size={16} />
                      </ActionIcon>
                    </Group>
                  </Stack>
                </Card>
              </Grid.Col>
            ))}
          </Grid>
        )}

        {filteredVoices.length === 0 && !loading && voices.length > 0 && (
          <Box ta="center" py="xl">
            <IconFilter size={48} stroke={1.5} color="gray" />
            <Text size="lg" fw={500} mt="md">
              No voices for selected languages
            </Text>
            <Text size="sm" c="dimmed">
              Change the filter or add new voices
            </Text>
          </Box>
        )}

        {voices.length === 0 && !loading && (
          <Box ta="center" py="xl">
            <IconMicrophone size={48} stroke={1.5} color="gray" />
            <Text size="lg" fw={500} mt="md">
              No voices found
            </Text>
            <Text size="sm" c="dimmed">
              Add a new voice to get started
            </Text>
          </Box>
        )}
      </Stack>

      {/* Modal window for adding a voice */}
      <Modal
        opened={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        title="Add new voice"
        size="md"
      >
        <Stack gap="md">
          <TextInput
            label="Voice name"
            placeholder="For example: RU_Male_MyVoice"
            value={uploadData.name}
            onChange={(event) =>
              setUploadData((prev) => ({
                ...prev,
                name: event.target.value,
              }))
            }
            description="It is recommended to use a language prefix (RU_, BY_, UA_, GE_)"
            required
          />

          <FileInput
            label="Audio file (WAV)"
            placeholder="Select WAV file"
            accept=".wav,audio/wav"
            value={uploadData.audioFile}
            onChange={(file) =>
              setUploadData((prev) => ({ ...prev, audioFile: file }))
            }
            leftSection={<IconMicrophone size={16} />}
            required
          />

          <Textarea
            label="Reference text"
            placeholder="Enter the text that matches the audio file..."
            value={uploadData.referenceText}
            onChange={(event) =>
              setUploadData((prev) => ({
                ...prev,
                referenceText: event.target.value,
              }))
            }
            minRows={4}
            description="This text will be used for voice testing"
            required
          />

          {uploading && (
            <Progress
              value={uploadProgress}
              size="lg"
              animated
            />
          )}

          <Group justify="flex-end" mt="md">
            <Button
              variant="light"
              onClick={() => setIsModalOpen(false)}
              disabled={uploading}
            >
              Cancel
            </Button>
            <Button
              onClick={createVoice}
              loading={uploading}
              leftSection={<IconUpload size={16} />}
            >
              Create voice
            </Button>
          </Group>
        </Stack>
      </Modal>

      {/* Modal window for confirming voice deletion */}
      <Modal
        opened={isDeleteConfirmOpen}
        onClose={() => setIsDeleteConfirmOpen(false)}
        title="Delete confirmation"
        size="md"
      >
        <Stack gap="md">
          <Alert
            icon={<IconAlertCircle />}
            title="Warning!"
            color="red"
            variant="light"
          >
            Are you sure you want to delete the voice "{voiceToDelete}"?
            <br />
            <strong>This action cannot be undone.</strong>
          </Alert>
          
          <Group justify="flex-end" mt="md">
            <Button
              variant="light"
              onClick={() => setIsDeleteConfirmOpen(false)}
            >
              Cancel
            </Button>
            <Button
              color="red"
              onClick={confirmDeleteVoice}
              leftSection={<IconTrash size={16} />}
            >
              Delete
            </Button>
          </Group>
        </Stack>
      </Modal>
    </Container>
  );
} 