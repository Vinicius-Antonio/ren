import React from "react";
import { HolographicOrb } from "./components/HolographicOrb";
import { WidgetDock } from "./components/WidgetDock";
import { useRenGateway } from "./hooks/useRenGateway";
import type { RenStatus } from "./types/gateway";

const STATUS_BADGE: Record<RenStatus, { label: string; dot: string }> = {
  STANDBY: { label: "STANDBY", dot: "bg-cyan-400" },
  LISTENING: { label: "OUVINDO", dot: "bg-emerald-400 animate-pulse" },
  PROCESSING: { label: "PROCESSANDO", dot: "bg-sky-400 animate-ping" },
  SPEAKING: { label: "FALANDO", dot: "bg-blue-400 animate-pulse" },
};

export const App: React.FC = () => {
  const { connected, status, rms, frequencies, widgets, dismissWidget } =
    useRenGateway();
  const badge = STATUS_BADGE[status] ?? STATUS_BADGE.STANDBY;

  return (
    <div
      data-tauri-drag-region
      className="w-screen h-screen flex flex-col items-center justify-start pt-3 gap-2 bg-transparent text-white"
    >
      <div className="relative flex flex-col items-center">
        <HolographicOrb status={status} rms={rms} frequencies={frequencies} />

        <div className="mt-[-12px] flex items-center gap-2 px-3 py-1 rounded-full bg-slate-950/75 border border-cyan-500/30 backdrop-blur-md text-[11px] tracking-widest font-medium">
          <span className={`w-2 h-2 rounded-full ${badge.dot}`} />
          <span>REN • {badge.label}</span>
          {!connected && (
            <span className="text-amber-400 text-[10px]">(OFFLINE)</span>
          )}
        </div>
      </div>

      <WidgetDock widgets={widgets} onDismiss={dismissWidget} />
    </div>
  );
};

export default App;
