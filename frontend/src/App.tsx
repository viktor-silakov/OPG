import { useState, useEffect } from "react";
import {
  AppShell,
  Burger,
  Group,
  Container,
  Tabs,
  Title,
  MantineProvider,
  Anchor,
} from "@mantine/core";
import { useDisclosure } from "@mantine/hooks";
import {
  IconMicrophone,
  IconPlayerPlay,
  IconServer,
  IconLibraryPlus,
  IconVolume,
  IconFileText,
  IconSettings,
} from "@tabler/icons-react";
import { PodcastWorkflow } from "./components/PodcastWorkflow";
import { StatusChecker } from "./components/StatusChecker";
import { PodcastLibrary } from "./components/PodcastLibrary";
import { VoiceManager } from "./components/VoiceManager";
import { PromptsManager } from "./components/PromptsManager";
import { EnvEditor } from "./components/EnvEditor";
import { Notifications } from "@mantine/notifications";
import { NotificationManager } from "./utils/notifications";
import "@mantine/core/styles.css";
import "@mantine/notifications/styles.css";

interface PodcastGenerationData {
  promptName: string;
  userPrompt: string;
  systemPrompt: string;
}

function App() {
  const [opened, { toggle }] = useDisclosure();
  const [activeTab, setActiveTab] = useState<string | null>("workflow");
  const [podcastGenerationData, setPodcastGenerationData] = useState<PodcastGenerationData | null>(null);

  // Function to handle navigation to podcast generation with data
  const handleNavigateToPodcastGeneration = (data: PodcastGenerationData) => {
    setPodcastGenerationData(data);
    setActiveTab("workflow");
  };

  // Clear generation data when tab changes
  const handleTabChange = (value: string | null) => {
    if (value !== "workflow") {
      setPodcastGenerationData(null);
    }
    setActiveTab(value);
  };

  // Request permission for push notifications on load
  useEffect(() => {
    const initNotifications = async () => {
      console.log('🚀 Initializing notifications...');
      NotificationManager.diagnose();
      if (!NotificationManager.isSupported()) {
        console.log('❌ Browser does not support Notification API');
        return;
      }
      console.log('✅ Notification API is supported');
      console.log('📋 Current permission state:', NotificationManager.getPermission());
      try {
        if (NotificationManager.getPermission() === 'default') {
          console.log('❓ Requesting notification permission...');
          const permission = await NotificationManager.requestPermission();
          if (permission === 'granted') {
            console.log('✅ Notifications allowed, showing welcome message');
            await NotificationManager.showWelcome();
          }
        } else if (NotificationManager.getPermission() === 'granted') {
          console.log('✅ Permission already granted');
        } else {
          console.log('❌ Notifications denied by user');
        }
      } catch (error) {
        console.error('❌ Error initializing notifications:', error);
      }
    };
    initNotifications();
  }, []);

  return (
    <MantineProvider>
      <Notifications />
      <AppShell
        header={{ height: 60 }}
        navbar={{
          width: 300,
          breakpoint: "sm",
          collapsed: { mobile: !opened },
        }}
        padding="md"
      >
        <AppShell.Header>
          <Group h="100%" px="md">
            <Burger
              opened={opened}
              onClick={toggle}
              hiddenFrom="sm"
              size="sm"
            />
            <Group>
              <IconMicrophone size={24} />
              <Anchor underline="never" href="/" c="inherit">
                <Title order={3}>Open NotebookLM RU - Podcast Generator</Title>
              </Anchor>
            </Group>
          </Group>
        </AppShell.Header>

        <AppShell.Navbar
          p="md"
          pr="0"
         
        >
          <Tabs
            value={activeTab}
            onChange={handleTabChange}
            orientation="vertical"
            styles={{
              list: {
                // alignItems: "stretch", // Stretches the tab to the full width of the container
              },
              tab: {
                // justifyContent: "flex-start", // Icon and text on the left
                // textAlign: "left", // Text on the left
                // width: "100%", // Important: stretches the tab itself
              },
              tabLabel: {
                // Explicitly align the text within the label to the left.
                textAlign: "left",
              },
            }}
          >
            <Tabs.List style={{ width: "100%" }}>
              <Tabs.Tab
                value="workflow"
                // style={{ width: "100%" }}
                leftSection={<IconPlayerPlay size={16} />}
              >
                Podcast Creation
              </Tabs.Tab>
              <Tabs.Tab
                value="library"
                leftSection={<IconLibraryPlus size={16} />}
              >
                Podcast Library
              </Tabs.Tab>
              <Tabs.Tab
                value="voices"
                leftSection={<IconVolume size={16} />}
              >
                Voice Management
              </Tabs.Tab>
              <Tabs.Tab
                value="prompts"
                leftSection={<IconFileText size={16} />}
              >
                Prompts Management
              </Tabs.Tab>
              <Tabs.Tab value="status" leftSection={<IconServer size={16} />}>
                Server Status
              </Tabs.Tab>
              <Tabs.Tab value="env" leftSection={<IconSettings size={16} />}>
                Configuration
              </Tabs.Tab>
            </Tabs.List>
          </Tabs>
        </AppShell.Navbar>

        <AppShell.Main>
          <Container size="lg">
            <Title order={2} mb="lg">
              {activeTab === "workflow" && "Podcast Creation"}
              {activeTab === "library" && "Podcast Library"}
              {activeTab === "voices" && "Voice Management"}
              {activeTab === "prompts" && "Prompts Management"}
              {activeTab === "status" && "Server Status"}
              {activeTab === "env" && "Configuration"}
            </Title>

            <Tabs value={activeTab} onChange={handleTabChange}>
              <Tabs.Panel value="workflow">
                <PodcastWorkflow generationData={podcastGenerationData} />
              </Tabs.Panel>

              <Tabs.Panel value="library">
                <PodcastLibrary />
              </Tabs.Panel>

              <Tabs.Panel value="voices">
                <VoiceManager />
              </Tabs.Panel>

              <Tabs.Panel value="prompts">
                <PromptsManager onNavigateToGeneration={handleNavigateToPodcastGeneration} />
              </Tabs.Panel>

              <Tabs.Panel value="status">
                <StatusChecker />
              </Tabs.Panel>

              <Tabs.Panel value="env">
                <EnvEditor />
              </Tabs.Panel>
            </Tabs>
          </Container>
        </AppShell.Main>
      </AppShell>
    </MantineProvider>
  );
}

export default App;
