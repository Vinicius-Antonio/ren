import { useEffect, useRef, useState } from "react";
import type {
  ActiveWidgetCard,
  GatewayOutboundMessage,
  RenStatus,
} from "../types/gateway";

const GATEWAY_WS_URL = "ws://127.0.0.1:8765";
const WIDGET_TTL_MS = 9000;

export interface RenGatewayState {
  connected: boolean;
  status: RenStatus;
  rms: number;
  frequencies: number[];
  widgets: ActiveWidgetCard[];
  dismissWidget: (id: string) => void;
}

export function useRenGateway(url: string = GATEWAY_WS_URL): RenGatewayState {
  const [connected, setConnected] = useState<boolean>(false);
  const [status, setStatus] = useState<RenStatus>("STANDBY");
  const [rms, setRms] = useState<number>(0);
  const [frequencies, setFrequencies] = useState<number[]>(() =>
    new Array(16).fill(0)
  );
  const [widgets, setWidgets] = useState<ActiveWidgetCard[]>([]);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    let cancelled = false;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

    const connect = () => {
      if (cancelled) return;
      try {
        const ws = new WebSocket(url);
        wsRef.current = ws;

        ws.onopen = () => {
          if (!cancelled) setConnected(true);
        };

        ws.onmessage = (event: MessageEvent<string>) => {
          try {
            const msg = JSON.parse(event.data) as GatewayOutboundMessage;
            if (msg.type === "STATE_CHANGED") {
              setStatus(msg.status);
              if (msg.status === "STANDBY") {
                setRms(0);
                setFrequencies(new Array(16).fill(0));
              }
            } else if (msg.type === "AUDIO_LEVEL") {
              setRms(msg.rms);
              if (Array.isArray(msg.frequencies) && msg.frequencies.length > 0) {
                setFrequencies(msg.frequencies);
              }
            } else if (msg.type === "UI_WIDGET") {
              const item: ActiveWidgetCard = {
                id: `${Date.now()}_${Math.random().toString(36).slice(2, 7)}`,
                widget: msg.widget,
                payload: msg.payload,
                timestamp: Date.now(),
              };
              setWidgets((prev) => [item, ...prev].slice(0, 3));
            }
          } catch (err) {
            console.error("Falha ao processar mensagem do gateway:", err);
          }
        };

        ws.onerror = () => {
          if (!cancelled) setConnected(false);
        };

        ws.onclose = () => {
          if (cancelled) return;
          setConnected(false);
          setStatus("STANDBY");
          setRms(0);
          setFrequencies(new Array(16).fill(0));
          reconnectTimer = setTimeout(connect, 1500);
        };
      } catch (err) {
        if (!cancelled) {
          setConnected(false);
          reconnectTimer = setTimeout(connect, 1500);
        }
      }
    };

    connect();

    return () => {
      cancelled = true;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      if (wsRef.current) {
        try {
          wsRef.current.close();
        } catch {
          // ignore
        }
      }
    };
  }, [url]);

  useEffect(() => {
    const timer = setInterval(() => {
      const cutoff = Date.now() - WIDGET_TTL_MS;
      setWidgets((prev) => prev.filter((w) => w.timestamp >= cutoff));
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  const dismissWidget = (id: string) => {
    setWidgets((prev) => prev.filter((w) => w.id !== id));
  };

  return { connected, status, rms, frequencies, widgets, dismissWidget };
}
