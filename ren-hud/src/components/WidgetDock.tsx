import React from "react";
import type { ActiveWidgetCard } from "../types/gateway";

interface WidgetDockProps {
  widgets: ActiveWidgetCard[];
  onDismiss: (id: string) => void;
}

const WIDGET_LABELS: Record<string, string> = {
  executive_briefing: "Briefing Executivo",
  music_player: "Player de Música",
  media_control: "Controle de Mídia",
  system_control: "Sistema Windows",
  app_launcher: "Aplicativo",
  google_search: "Google Search Grounding",
  vision: "Visão Contínua",
  memory: "Memória Episódica",
  timer: "Timer Ativo",
  email_inbox: "Caixa de Entrada",
  voice_note: "Nota de Voz",
  background_task: "Slow Loop Task",
  pc_status: "Status do Sistema",
};

export const WidgetDock: React.FC<WidgetDockProps> = ({ widgets, onDismiss }) => {
  if (widgets.length === 0) return null;

  return (
    <div className="w-full max-w-[360px] flex flex-col gap-2 px-3">
      {widgets.map((card) => {
        const title = WIDGET_LABELS[card.widget] ?? card.widget;
        const summary =
          typeof card.payload.summary === "string"
            ? card.payload.summary
            : typeof card.payload.message === "string"
              ? card.payload.message
              : typeof card.payload.fact === "string"
                ? card.payload.fact
                : typeof card.payload.status === "string"
                  ? card.payload.status
                  : JSON.stringify(card.payload);

        return (
          <div
            key={card.id}
            className="rounded-xl bg-slate-950/80 border border-cyan-500/30 backdrop-blur-md px-3.5 py-2.5 text-xs text-slate-100 shadow-lg transition-all"
          >
            <div className="flex items-center justify-between mb-1">
              <span className="font-semibold uppercase tracking-wider text-[10px] text-cyan-400">
                {title}
              </span>
              <button
                onClick={() => onDismiss(card.id)}
                className="text-slate-400 hover:text-white text-[11px]"
              >
                ✕
              </button>
            </div>
            <p className="text-slate-200 leading-relaxed line-clamp-3">
              {summary}
            </p>
          </div>
        );
      })}
    </div>
  );
};

