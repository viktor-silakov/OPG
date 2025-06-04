import { useState, useEffect } from 'react'
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
  Code,
  List,
} from '@mantine/core'
import { notifications } from '@mantine/notifications'
import { IconCheck, IconX, IconRefresh } from '@tabler/icons-react'
import { apiClient, type StatusResponse } from '../api/client'

export function StatusChecker() {
  const [loading, setLoading] = useState(false)
  const [status, setStatus] = useState<StatusResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [lastChecked, setLastChecked] = useState<Date | null>(null)

  const checkStatus = async () => {
    setLoading(true)
    setError(null)

    try {
      const response = await apiClient.getStatus()
      setStatus(response)
      setLastChecked(new Date())
      
      // Remove automatic notification, keep only for manual check
      // notifications.show({
      //   title: 'Server available',
      //   message: response.message,
      //   color: 'green',
      //   icon: <IconCheck size={16} />,
      // })
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error'
      setError(errorMessage)
      setStatus(null)
      
      // Remove automatic error notification
      // notifications.show({
      //   title: 'Connection error',
      //   message: errorMessage,
      //   color: 'red',
      //   icon: <IconX size={16} />,
      // })
    } finally {
      setLoading(false)
    }
  }

  // Add function for manual check with notification
  const checkStatusWithNotification = async () => {
    setLoading(true)
    setError(null)

    try {
      const response = await apiClient.getStatus()
      setStatus(response)
      setLastChecked(new Date())
      
      notifications.show({
        title: 'Server available',
        message: response.message,
        color: 'green',
        icon: <IconCheck size={16} />,
      })
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error'
      setError(errorMessage)
      setStatus(null)
      
      notifications.show({
        title: 'Connection error',
        message: errorMessage,
        color: 'red',
        icon: <IconX size={16} />,
      })
    } finally {
      setLoading(false)
    }
  }

  // Automatic check on component mount
  useEffect(() => {
    checkStatus()
  }, [])

  const getStatusColor = () => {
    if (error) return 'red'
    if (status?.status === 'ok') return 'green'
    return 'gray'
  }

  const getStatusText = () => {
    if (error) return 'Unavailable'
    if (status?.status === 'ok') return 'Running'
    return 'Unknown'
  }

  return (
    <Stack gap="lg">
      <Paper p="md" withBorder>
        <Stack gap="md">
          <Group justify="space-between">
            <Title order={4}>Server status</Title>
            <Button
              variant="light"
              size="sm"
              loading={loading}
              leftSection={<IconRefresh size={16} />}
              onClick={checkStatusWithNotification}
            >
              Update
            </Button>
          </Group>

          <Group gap="md">
            <Badge color={getStatusColor()} variant="light" size="lg">
              {getStatusText()}
            </Badge>
            {lastChecked && (
              <Text size="sm" c="dimmed">
                Last checked: {lastChecked.toLocaleTimeString()}
              </Text>
            )}
          </Group>

          {loading && (
            <Group>
              <Loader size="sm" />
              <Text>Checking server status...</Text>
            </Group>
          )}
        </Stack>
      </Paper>

      {error && (
        <Alert color="red" title="Connection error" icon={<IconX size={16} />}>
          <Stack gap="xs">
            <Text>{error}</Text>
            <Text size="sm" c="dimmed">
              Make sure the server is running on the correct port and available at: {apiClient.apiBaseUrl || 'http://localhost:3000'}
            </Text>
          </Stack>
        </Alert>
      )}

      {status && (
        <Paper p="md" withBorder>
          <Stack gap="md">
            <Title order={4}>Server info</Title>
            
            <div>
              <Text size="sm" c="dimmed" mb="xs">
                Message:
              </Text>
              <Text>{status.message}</Text>
            </div>

            <Divider />

            <div>
              <Text size="sm" c="dimmed" mb="xs">
                Available endpoints:
              </Text>
              <List spacing="xs">
                {Object.entries(status.endpoints).map(([endpoint, description]) => (
                  <List.Item key={endpoint}>
                    <Group gap="xs">
                      <Code>{endpoint}</Code>
                      <Text size="sm">{description}</Text>
                    </Group>
                  </List.Item>
                ))}
              </List>
            </div>

            <Alert color="blue" title="API URL">
              Server available at: <Code>{apiClient.apiBaseUrl || 'http://localhost:3000'}</Code>
            </Alert>
          </Stack>
        </Paper>
      )}

      <Paper p="md" withBorder>
        <Stack gap="md">
          <Title order={4}>Help info</Title>
          
          <div>
            <Text size="sm" c="dimmed" mb="xs">
              To use the app you need:
            </Text>
            <List spacing="xs">
              <List.Item>Running backend server on port 3000</List.Item>
              <List.Item>Configured environment variables (GEMINI_API_KEY, etc.)</List.Item>
              <List.Item>Available services for audio generation</List.Item>
            </List>
          </div>

          <Divider />

          <div>
            <Text size="sm" c="dimmed" mb="xs">
              Server startup commands:
            </Text>
            <Stack gap="xs">
              <Code block>npm run server</Code>
              <Text size="xs" c="dimmed">or</Text>
              <Code block>npm run dev</Code>
            </Stack>
          </div>
        </Stack>
      </Paper>
    </Stack>
  )
} 