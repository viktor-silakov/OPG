import { useState } from 'react'
import {
  Paper,
  TextInput,
  Button,
  Group,
  Text,
  Alert,
  JsonInput,
  Stack,
  Badge,
  Title,
  Loader,
  Radio,
  FileInput,
} from '@mantine/core'
import { useForm } from '@mantine/form'
import { notifications } from '@mantine/notifications'
import { IconCheck, IconX, IconPlayerPlay, IconUpload } from '@tabler/icons-react'
import { apiClient, type GeneratePodcastResponse } from '../api/client'

type GenerationMode = 'file' | 'json' | 'default'

export function PodcastGenerator() {
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<GeneratePodcastResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [mode, setMode] = useState<GenerationMode>('default')

  const form = useForm({
    initialValues: {
      conversationFile: '',
      conversationJson: '',
    },
    validate: {
      conversationFile: (value) => 
        mode === 'file' && !value.trim() ? 'Specify the script file path' : null,
      conversationJson: (value) => {
        if (mode !== 'json') return null
        if (!value.trim()) return 'Enter JSON data'
        try {
          JSON.parse(value)
          return null
        } catch {
          return 'Invalid JSON format'
        }
      },
    },
  })

  const handleSubmit = async (values: typeof form.values) => {
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      let request: any = {}

      if (mode === 'file') {
        request.conversationFile = values.conversationFile
      } else if (mode === 'json') {
        request.conversationData = JSON.parse(values.conversationJson)
      }
      // For mode === 'default', an empty object is sent

      const response = await apiClient.generatePodcast(request)
      setResult(response)
      
      notifications.show({
        title: 'Success!',
        message: `Podcast generation started: ${response.messageCount} lines`,
        color: 'green',
        icon: <IconCheck size={16} />,
      })
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error'
      setError(errorMessage)
      
      notifications.show({
        title: 'Error',
        message: errorMessage,
        color: 'red',
        icon: <IconX size={16} />,
      })
    } finally {
      setLoading(false)
    }
  }

  const handleFileUpload = (file: File | null) => {
    if (!file) return

    const reader = new FileReader()
    reader.onload = (e) => {
      try {
        const content = e.target?.result as string
        const jsonData = JSON.parse(content)
        form.setFieldValue('conversationJson', JSON.stringify(jsonData, null, 2))
        setMode('json')
      } catch (err) {
        notifications.show({
          title: 'Error',
          message: 'Unable to read JSON file',
          color: 'red',
          icon: <IconX size={16} />,
        })
      }
    }
    reader.readAsText(file)
  }

  return (
    <Stack gap="lg">
      <Paper p="md" withBorder>
        <Stack gap="md">
          <Title order={5}>Select script source:</Title>
          
          <Radio.Group
            value={mode}
            onChange={(value) => setMode(value as GenerationMode)}
          >
            <Stack gap="xs">
              <Radio 
                value="default" 
                label="Use conversation.json from server" 
              />
              <Radio 
                value="file" 
                label="Specify file path on server" 
              />
              <Radio 
                value="json" 
                label="Upload JSON data" 
              />
            </Stack>
          </Radio.Group>

          <form onSubmit={form.onSubmit(handleSubmit)}>
            <Stack gap="md">
              {mode === 'file' && (
                <TextInput
                  label="Script file path"
                  placeholder="conversation-2024-01-15T10-30-45-123Z.json"
                  required
                  {...form.getInputProps('conversationFile')}
                />
              )}

              {mode === 'json' && (
                <>
                  <FileInput
                    label="Upload JSON file"
                    placeholder="Select script file"
                    accept=".json"
                    leftSection={<IconUpload size={16} />}
                    onChange={handleFileUpload}
                  />
                  
                  <JsonInput
                    label="Script JSON data"
                    placeholder='{"podcast_name": "...", "filename": "...", "conversation": [...]}'
                    required
                    minRows={6}
                    maxRows={12}
                    {...form.getInputProps('conversationJson')}
                  />
                </>
              )}

              {mode === 'default' && (
                <Alert color="blue" title="Info">
                  The conversation.json file from the server will be used. 
                  Make sure the script is already generated.
                </Alert>
              )}

              <Group justify="flex-end">
                <Button 
                  type="submit" 
                  loading={loading}
                  leftSection={<IconPlayerPlay size={16} />}
                >
                  Generate podcast
                </Button>
              </Group>
            </Stack>
          </form>
        </Stack>
      </Paper>

      {error && (
        <Alert color="red" title="Error" icon={<IconX size={16} />}>
          {error}
        </Alert>
      )}

      {loading && (
        <Paper p="md" withBorder>
          <Group>
            <Loader size="sm" />
            <Text>Loading podcast list...</Text>
          </Group>
        </Paper>
      )}

      {result && (
        <Paper p="md" withBorder>
          <Stack gap="md">
            <Title order={4}>Podcast generation started</Title>
            
            <Group gap="xs">
              <Badge color="blue" variant="light">
                {result.status}
              </Badge>
              <Badge color="green" variant="light">
                {result.messageCount} lines
              </Badge>
            </Group>

            <div>
              <Text size="sm" c="dimmed" mb="xs">
                Podcast name:
              </Text>
              <Text fw={500}>{result.podcast_name}</Text>
            </div>

            <div>
              <Text size="sm" c="dimmed" mb="xs">
                Output file:
              </Text>
              <Text fw={500}>{result.filename}</Text>
            </div>

            <Alert color="yellow" title="Attention">
              Podcast generation is performed in the background. 
              The process may take several minutes depending on the number of lines.
              The finished file will be saved in the output folder on the server.
            </Alert>
          </Stack>
        </Paper>
      )}
    </Stack>
  )
} 