export type RenStatus = "STANDBY" | "LISTENING" | "PROCESSING" | "SPEAKING";
export type HermesStatus = RenStatus;

export interface StateChangedMessage {
  type: "STATE_CHANGED";
  status: RenStatus;
}

export interface AudioLevelMessage {
  type: "AUDIO_LEVEL";
  rms: number;
  raw_rms?: number;
  frequencies: number[];
}

export interface UiWidgetMessage {
  type: "UI_WIDGET";
  widget: string;
  payload: Record<string, unknown>;
}

export type GatewayOutboundMessage =
  | StateChangedMessage
  | AudioLevelMessage
  | UiWidgetMessage;

export interface ActiveWidgetCard {
  id: string;
  widget: string;
  payload: Record<string, unknown>;
  timestamp: number;
}
