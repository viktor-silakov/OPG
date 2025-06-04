import { useState } from 'react'
import {
  Paper,
  Button,
  Group,
  Text,
  Alert,
  Stack,
  Badge,
  Title,
  Divider,
  Loader,
  Textarea,
} from '@mantine/core'
import { useForm } from '@mantine/form'
import { notifications } from '@mantine/notifications'
import { IconCheck, IconX, IconPlayerPlay } from '@tabler/icons-react'
import { apiClient, type GenerateScriptResponse } from '../api/client'

export function ScriptGenerator() {
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<GenerateScriptResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  const form = useForm({
    initialValues: {
      userPrompt: '',
      systemPrompt: '',
    },
    validate: {
      userPrompt: (value) => (!value.trim() ? 'Specify the podcast topic' : null),
    },
  })

  const handleSubmit = async (values: typeof form.values) => {
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const request = {
        userPrompt: values.userPrompt,
        ...(values.systemPrompt && { systemPrompt: values.systemPrompt }),
      }

      const response = await apiClient.generateScript(request)
      setResult(response)
      
      notifications.show({
        title: 'Success!',
        message: `Script generated: ${response.messageCount} lines`,
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

  const downloadJson = () => {
    if (!result) return

    const dataStr = JSON.stringify(result.conversation, null, 2)
    const dataBlob = new Blob([dataStr], { type: 'application/json' })
    const url = URL.createObjectURL(dataBlob)
    const link = document.createElement('a')
    link.href = url
    link.download = result.filename
    link.click()
    URL.revokeObjectURL(url)
  }

  return (
    <Stack gap="lg">
      <Paper p="md" withBorder>
        <form onSubmit={form.onSubmit(handleSubmit)}>
          <Stack gap="md">
            <Textarea
              label="Podcast topic"
              placeholder="Describe the topic for the podcast..."
              required
              minRows={3}
              maxRows={6}
              {...form.getInputProps('userPrompt')}
            />

            <Textarea
              label="System prompt (optional)"
              placeholder="Custom system prompt (if not specified, the default will be used)"
              minRows={8}
              maxRows={16}
              autosize
              resize="vertical"
              {...form.getInputProps('systemPrompt')}
            />

            <Group justify="flex-end">
              <Button 
                type="submit" 
                loading={loading}
                leftSection={<IconPlayerPlay size={16} />}
              >
                Generate script
              </Button>
            </Group>
          </Stack>
        </form>
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
            <Text>Generating podcast script...</Text>
          </Group>
        </Paper>
      )}

      {result && (
        <Paper p="md" withBorder>
          <Stack gap="md">
            <Group justify="space-between">
              <Title order={4}>Generation result</Title>
              <Button
                variant="light"
                size="sm"
                leftSection={<IconPlayerPlay size={16} />}
                onClick={downloadJson}
              >
                Download JSON
              </Button>
            </Group>
            
            <Group gap="xs">
              <Badge color="green" variant="light">
                {result.messageCount} lines
              </Badge>
              <Badge color="blue" variant="light">
                {result.filename}
              </Badge>
            </Group>

            <div>
              <Text size="sm" c="dimmed" mb="xs">
                Podcast name:
              </Text>
              <Text fw={500}>{result.conversation.podcast_name}</Text>
            </div>

            <Divider />

            <div>
              <Text size="sm" c="dimmed" mb="xs">
                Preview (first 3 lines):
              </Text>
              <Stack gap="xs">
                {result.conversation.conversation.slice(0, 3).map((message) => (
                  <Paper key={message.id} p="xs" bg="gray.0">
                    <Group gap="xs">
                      <Badge size="xs" variant="outline">
                        {message.speaker}
                      </Badge>
                      <Text size="sm">{message.text}</Text>
                    </Group>
                  </Paper>
                ))}
                {result.conversation.conversation.length > 3 && (
                  <Text size="sm" c="dimmed" ta="center">
                    ...and {result.conversation.conversation.length - 3} more lines
                  </Text>
                )}
              </Stack>
            </div>
          </Stack>
        </Paper>
      )}
    </Stack>
  )
} 