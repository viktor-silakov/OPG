import { useState, useEffect } from 'react'
import {
  Paper,
  Button,
  Group,
  Text,
  Stack,
  Badge,
  Loader,
  Alert,
  Card,
  Divider,
} from '@mantine/core'
import { notifications } from '@mantine/notifications'
import { IconDownload, IconPlayerPlay, IconRefresh, IconFileMusic } from '@tabler/icons-react'
import { apiClient, type PodcastFile } from '../api/client'

export function PodcastLibrary() {
  const [podcasts, setPodcasts] = useState<PodcastFile[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const loadPodcasts = async () => {
    setLoading(true)
    setError(null)

    try {
      const response = await apiClient.getPodcasts()
      setPodcasts(response.podcasts)
      
      if (response.podcasts.length === 0) {
        notifications.show({
          title: 'Info',
          message: 'No podcasts created yet',
          color: 'blue',
        })
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error'
      setError(errorMessage)
      
      notifications.show({
        title: 'Error',
        message: 'Failed to load podcast list',
        color: 'red',
      })
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadPodcasts()
  }, [])

  const formatFileSize = (bytes: number) => {
    const mb = bytes / (1024 * 1024)
    return `${mb.toFixed(1)} MB`
  }

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString('en-US')
  }

  return (
    <Stack gap="lg">
      <Group justify="space-between">
        <Button
          onClick={loadPodcasts}
          loading={loading}
          leftSection={<IconRefresh size={16} />}
          variant="light"
        >
          Refresh list
        </Button>
      </Group>

      {loading && (
        <Paper p="md" withBorder>
          <Group>
            <Loader size="sm" />
            <Text>Loading podcast list...</Text>
          </Group>
        </Paper>
      )}

      {error && (
        <Alert color="red" title="Error">
          {error}
        </Alert>
      )}

      {!loading && podcasts.length === 0 && !error && (
        <Paper p="xl" withBorder ta="center">
          <Stack gap="md" align="center">
            <IconFileMusic size={48} color="gray" />
            <Text size="lg" c="dimmed">
              No podcasts created yet
            </Text>
            <Text size="sm" c="dimmed">
              Go to the "Podcast Creation" tab to generate your first podcast
            </Text>
          </Stack>
        </Paper>
      )}

      {podcasts.length > 0 && (
        <Stack gap="md">
          {podcasts.map((podcast) => (
            <Card key={podcast.filename} withBorder>
              <Stack gap="md">
                <Group justify="space-between">
                  <div>
                    <Text fw={500} size="lg">
                      {podcast.filename.replace('.wav', '')}
                    </Text>
                    <Text size="sm" c="dimmed">
                      Created: {formatDate(podcast.created)}
                    </Text>
                  </div>
                  <Badge color="blue" variant="light">
                    {formatFileSize(podcast.size)}
                  </Badge>
                </Group>

                <Divider />

                <div>
                  <Text size="sm" c="dimmed" mb="xs">
                    Audio player:
                  </Text>
                  <audio 
                    controls 
                    style={{ width: '100%' }}
                    src={`http://localhost:3000${podcast.url}`}
                    preload="metadata"
                    onError={(e) => {
                      console.error('Audio loading error:', e);
                      notifications.show({
                        title: 'Audio load error',
                        message: 'Failed to load audio file. Try downloading the file.',
                        color: 'orange',
                      });
                    }}
                  >
                    Your browser does not support the audio element.
                  </audio>
                </div>

                <Group>
                  <Button
                    onClick={() => {
                      const downloadUrl = `http://localhost:3000/download/${podcast.filename}/${podcast.filename}`;
                      const link = document.createElement('a');
                      link.href = downloadUrl;
                      link.target = '_blank';
                      document.body.appendChild(link);
                      link.click();
                      document.body.removeChild(link);
                    }}
                    leftSection={<IconDownload size={16} />}
                    variant="filled"
                    size="sm"
                  >
                    Download
                  </Button>
                  
                  <Button
                    onClick={() => window.open(`http://localhost:3000${podcast.url}`, '_blank')}
                    leftSection={<IconPlayerPlay size={16} />}
                    variant="outline"
                    size="sm"
                  >
                    Open in new tab
                  </Button>
                </Group>
              </Stack>
            </Card>
          ))}
        </Stack>
      )}
    </Stack>
  )
} 