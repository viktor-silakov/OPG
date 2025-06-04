import { WebSocket, WebSocketServer } from "ws";

// Extended WebSocket interface
export interface ExtendedWebSocket extends WebSocket {
  jobId?: string;
}

// Store WebSocket clients for progress updates
export const progressClients = new Map<string, ExtendedWebSocket>();

export function handleWebSocketConnection(wss: WebSocketServer) {
  wss.on("connection", (ws, req) => {
    const extendedWs = ws as ExtendedWebSocket;
    const clientId = Date.now().toString();
    progressClients.set(clientId, extendedWs);

    console.log(`WebSocket client connected: ${clientId}`);

    extendedWs.on("message", (message: any) => {
      try {
        const data = JSON.parse(message.toString());
        if (data.type === "subscribe" && data.jobId) {
          extendedWs.jobId = data.jobId;
        }
      } catch (err) {
        console.error("WebSocket message error:", err);
      }
    });

    extendedWs.on("close", () => {
      progressClients.delete(clientId);
      console.log(`WebSocket client disconnected: ${clientId}`);
    });
  });
}

export function sendProgressUpdate(
  jobId: string,
  current: number,
  total: number,
  message?: string
) {
  for (const [clientId, ws] of progressClients.entries()) {
    if (ws.readyState === 1 && ws.jobId === jobId) {
      // OPEN state
      try {
        ws.send(
          JSON.stringify({
            type: "progress",
            jobId,
            current,
            total,
            message,
          })
        );
      } catch (err) {
        console.error(`Error sending progress to client ${clientId}:`, err);
        progressClients.delete(clientId);
      }
    }
  }
} 