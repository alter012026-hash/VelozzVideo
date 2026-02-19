import React, { useState, useEffect, useMemo, useRef } from 'react';
import { createRoot } from 'react-dom/client';

// --- Types ---
interface Scene {
  id: string;
  text: string;
  visualPrompt: string;
  localImage?: string;
  narrationVolume?: number;
  trimStartMs?: number;
  trimEndMs?: number;
  audioOffsetMs?: number;
  localSfx?: string;
  sfxVolume?: number;
  narrationAudio?: string; // Data URL MP3 (preview)
  status: 'pending' | 'processing' | 'completed';
  transition?: string;
  filter?: string;
  flowTo?: string | null;
}

interface GeneralSettings {
  theme: string;
  localThemeRef?: string; // Imagem local de referência de estilo
  backgroundMusic: string;
  localBackgroundMusic?: string; // Áudio local de fundo
  voiceName: string;
  defaultTransition?: string;
  defaultFilter?: string;
}

interface Script {
  title: string;
  scenes: Scene[];
}

type WorkflowStep = 'topic' | 'scripting' | 'assets' | 'rendering' | 'done';
type WorkspacePage = 'studio' | 'timeline';
type TimelineEditMode = 'pre' | 'post';

interface GridTile {
  id: string;
  row: number;
  col: number;
  dataUrl: string;
}

interface TimelineDragState {
  sceneId: string;
  startSec: number;
  durationSec: number;
  originClientX: number;
  minSec: number;
  maxSec: number;
  currentSec: number;
  pointerId: number;
}

interface OpenSections {
  ollama: boolean;
  theme: boolean;
  effects: boolean;
  audio: boolean;
  voice: boolean;
  backend: boolean;
  grid: boolean;
  advanced: boolean;
}

interface SessionPayload {
  topic: string;
  format: '16:9' | '9:16';
  videoLength: '1m' | '5m';
  ffmpegFilters?: string;
  stabilize?: boolean;
  aiEnhance?: boolean;
  engine?: string;
  settings: GeneralSettings;
  script: Script | null;
  scenePositions: Record<string, { x: number; y: number }>;
  gridRows: number;
  gridCols: number;
  gridSceneTarget: string | null;
  edgeVoiceId: string;
  currentStep?: WorkflowStep;
  videoUrl?: string | null;
  timelineEditMode?: TimelineEditMode;
  ollamaHost?: string;
  ollamaModel?: string;
  captionScale?: number;
  captionBg?: number;
  captionColor?: string;
  captionHighlight?: string;
  captionY?: number;
  musicVolume?: number;
  narrationVolume?: number;
  colorStrength?: number;
  transitionDuration?: number;
  imageScale?: number;
  previewText?: string;
  previewTransition?: string;
  previewFilter?: string;
  previewAnimation?: string;
  previewColorStrength?: number;
  previewFilters?: string;
  previewUrl?: string | null;
  cleanupAgeDays?: number;
  canvasZoom?: number;
  openSections?: Partial<OpenSections>;
  gridFile?: string | null;
  collapsedScenes?: Record<string, boolean>;
}

const SESSION_KEY = 'vv_session';
const SESSION_PREV_KEY = 'vv_session_prev';
const SESSION_DB_NAME = 'vv_session_db';
const SESSION_DB_STORE = 'sessions';
type SessionStoreKey = typeof SESSION_KEY | typeof SESSION_PREV_KEY;

const hasIndexedDb = () => typeof window !== 'undefined' && typeof window.indexedDB !== 'undefined';

const openSessionDb = (): Promise<IDBDatabase> =>
  new Promise((resolve, reject) => {
    if (!hasIndexedDb()) {
      reject(new Error('IndexedDB indisponivel'));
      return;
    }
    const req = window.indexedDB.open(SESSION_DB_NAME, 1);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(SESSION_DB_STORE)) {
        db.createObjectStore(SESSION_DB_STORE);
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error || new Error('Falha ao abrir IndexedDB'));
  });

const idbGetText = async (key: SessionStoreKey): Promise<string | null> => {
  const db = await openSessionDb();
  try {
    return await new Promise((resolve, reject) => {
      const tx = db.transaction(SESSION_DB_STORE, 'readonly');
      const store = tx.objectStore(SESSION_DB_STORE);
      const req = store.get(key);
      req.onsuccess = () => resolve(typeof req.result === 'string' ? req.result : null);
      req.onerror = () => reject(req.error || new Error('Falha ao ler IndexedDB'));
    });
  } finally {
    db.close();
  }
};

const idbSetText = async (key: SessionStoreKey, value: string): Promise<void> => {
  const db = await openSessionDb();
  try {
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(SESSION_DB_STORE, 'readwrite');
      const store = tx.objectStore(SESSION_DB_STORE);
      const req = store.put(value, key);
      req.onsuccess = () => resolve();
      req.onerror = () => reject(req.error || new Error('Falha ao gravar IndexedDB'));
    });
  } finally {
    db.close();
  }
};

const idbRemoveKey = async (key: SessionStoreKey): Promise<void> => {
  const db = await openSessionDb();
  try {
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(SESSION_DB_STORE, 'readwrite');
      const store = tx.objectStore(SESSION_DB_STORE);
      const req = store.delete(key);
      req.onsuccess = () => resolve();
      req.onerror = () => reject(req.error || new Error('Falha ao remover chave no IndexedDB'));
    });
  } finally {
    db.close();
  }
};

const App: React.FC = () => {
  const [topic, setTopic] = useState('');
  const [videoLength, setVideoLength] = useState<'1m' | '5m'>('1m');
  const [currentStep, setCurrentStep] = useState<WorkflowStep>('topic');
  const [workspacePage, setWorkspacePage] = useState<WorkspacePage>(() => {
    if (typeof window === 'undefined') return 'studio';
    return window.location.hash === '#timeline' ? 'timeline' : 'studio';
  });
  const [timelineEditMode, setTimelineEditMode] = useState<TimelineEditMode>('pre');
  const [timelineZoom, setTimelineZoom] = useState<number>(1);
  const [timelinePlayheadSec, setTimelinePlayheadSec] = useState<number>(0);
  const [isTimelinePlaying, setIsTimelinePlaying] = useState<boolean>(false);
  const [timelineSelectedSceneId, setTimelineSelectedSceneId] = useState<string | null>(null);
  const [showTimelineInspector, setShowTimelineInspector] = useState<boolean>(false);
  const [showTimelineRenderPanel, setShowTimelineRenderPanel] = useState<boolean>(false);
  const [timelineTrackHeight, setTimelineTrackHeight] = useState<number>(86);
  const [timelineDrag, setTimelineDrag] = useState<TimelineDragState | null>(null);
  const timelineDragSecRef = useRef<number>(0);
  const stepOrder: WorkflowStep[] = ['topic', 'scripting', 'assets', 'rendering', 'done'];
  const transitionOptions = [
    { id: 'fade', label: 'Fade in/out (padrao)' },
    { id: 'crossfade', label: 'Crossfade' },
    { id: 'slide_left', label: 'Slide Left' },
    { id: 'slide_right', label: 'Slide Right' },
    { id: 'slide_up', label: 'Slide Up' },
    { id: 'slide_down', label: 'Slide Down' },
    { id: 'none', label: 'Sem transicao' },
  ];
  const animationOptions = [
    { id: 'kenburns', label: 'Ken Burns' },
    { id: 'zoom_in', label: 'Zoom In' },
    { id: 'zoom_out', label: 'Zoom Out' },
    { id: 'zoom_in_fast', label: 'Zoom In Rápido' },
    { id: 'zoom_out_fast', label: 'Zoom Out Rápido' },
    { id: 'pan_left', label: 'Pan Left' },
    { id: 'pan_right', label: 'Pan Right' },
    { id: 'pan_up', label: 'Pan Up' },
    { id: 'pan_down', label: 'Pan Down' },
    { id: 'rotate_left', label: 'Rotate Left' },
    { id: 'rotate_right', label: 'Rotate Right' },
    { id: 'sway', label: 'Sway' },
    { id: 'pulse', label: 'Pulse' },
    { id: 'warp_in', label: 'Warp In' },
    { id: 'warp_out', label: 'Warp Out' },
  ];
  const filterOptions = [
    { id: 'none', label: 'Nenhum' },
    { id: 'cinematic', label: 'Cinematic quente' },
    { id: 'cool', label: 'Frio / teal' },
    { id: 'warm', label: 'Quente' },
    { id: 'bw', label: 'Preto e branco' },
    { id: 'vibrant', label: 'Vibrante' },
    { id: 'vhs', label: 'VHS grao leve' },
    { id: 'matte', label: 'Matte suave' },
  ];
  const [isGeneratingScript, setIsGeneratingScript] = useState(false);
  const [script, setScript] = useState<Script | null>(null);
  const [isGeneratingVideo, setIsGeneratingVideo] = useState(false);
  const [isTestingRender, setIsTestingRender] = useState(false);
  const [isGeneratingNarration, setIsGeneratingNarration] = useState<string | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [statusMessage, setStatusMessage] = useState('');
  const [apiBase, setApiBase] = useState<string>('');
  const [renderTaskId, setRenderTaskId] = useState<string | null>(null);
  const [renderProgress, setRenderProgress] = useState<number>(0);
  const [renderStage, setRenderStage] = useState<string>('');
  const [renderRealPct, setRenderRealPct] = useState<number>(0);
  const [renderRealInfo, setRenderRealInfo] = useState<string>('');
  const [renderStaleSeconds, setRenderStaleSeconds] = useState<number>(0);
  const [renderElapsedSeconds, setRenderElapsedSeconds] = useState<number>(0);
  const renderStartedAtRef = useRef<number | null>(null);
  const [format, setFormat] = useState<'16:9' | '9:16'>('16:9');
  const [importError, setImportError] = useState<string | null>(null);
  const [ollamaHost, setOllamaHost] = useState('http://127.0.0.1:11434');
  const [ollamaModel, setOllamaModel] = useState('llama3');
  const [ollamaModels, setOllamaModels] = useState<string[]>([]);
  const [isLoadingModels, setIsLoadingModels] = useState(false);
  const [ollamaError, setOllamaError] = useState<string | null>(null);
  const [logFeed, setLogFeed] = useState<{ id: string; msg: string; level: 'info' | 'warn' | 'error'; time: string }[]>([]);
  const [isPreviewingVoice, setIsPreviewingVoice] = useState(false);
  const [previewText, setPreviewText] = useState('Esta e uma previa de voz.');
  const [edgeVoiceId, setEdgeVoiceId] = useState('pt-BR-ThalitaMultilingualNeural');
  const [edgeVoices, setEdgeVoices] = useState<{ id: string; label: string; gender: string; locale: string }[]>([]);
  const [livePreview, setLivePreview] = useState<{ sceneId: string; label: string } | null>(null);
  const [cleanupAgeDays, setCleanupAgeDays] = useState<number>(7);
  const [cleanupResult, setCleanupResult] = useState<string | null>(null);
  const [isCleaningCache, setIsCleaningCache] = useState(false);
  const [captionScale, setCaptionScale] = useState<number>(1);
  const [captionBg, setCaptionBg] = useState<number>(0.55);
  const [captionColor, setCaptionColor] = useState<string>('#ffffff');
  const [captionHighlight, setCaptionHighlight] = useState<string>('#ffd166');
  const [captionY, setCaptionY] = useState<number>(82); // porcentagem
  const [musicVolume, setMusicVolume] = useState<number>(0.25);
  const [narrationVolume, setNarrationVolume] = useState<number>(1);
  const [colorStrength, setColorStrength] = useState<number>(0.35);
  const [transitionDuration, setTransitionDuration] = useState<number>(0.6);
  const [imageScale, setImageScale] = useState<number>(1);
  const [previewTransition, setPreviewTransition] = useState<string>('crossfade');
  const [previewFilter, setPreviewFilter] = useState<string>('cinematic');
  const [previewAnimation, setPreviewAnimation] = useState<string>('kenburns');
  const [previewColorStrength, setPreviewColorStrength] = useState<number>(0.5);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isPreviewRendering, setIsPreviewRendering] = useState<boolean>(false);
  const [previewFilters, setPreviewFilters] = useState<string>('');
  const backendWarnRef = useRef<number>(0);
  const [ffmpegFilters, setFfmpegFilters] = useState<string>('');
  const [stabilize, setStabilize] = useState<boolean>(false);
  const [aiEnhance, setAiEnhance] = useState<boolean>(false);
  const [engine, setEngine] = useState<string>('moviepy');
  const staticEdgeVoices = [
    { id: 'pt-BR-ThalitaMultilingualNeural', label: 'ThalitaMultilingualNeural', gender: 'Female', locale: 'pt-BR' },
    { id: 'pt-BR-FranciscaNeural', label: 'FranciscaNeural', gender: 'Female', locale: 'pt-BR' },
    { id: 'pt-BR-AntonioNeural', label: 'AntonioNeural', gender: 'Male', locale: 'pt-BR' },
    { id: 'pt-PT-DuarteNeural', label: 'DuarteNeural', gender: 'Male', locale: 'pt-PT' },
    { id: 'pt-PT-RaquelNeural', label: 'RaquelNeural', gender: 'Female', locale: 'pt-PT' },
  ];
  const edgeVoiceOptions = edgeVoices.length ? edgeVoices : staticEdgeVoices;
  const selectedEdgeVoice = edgeVoiceOptions.find(v => v.id === edgeVoiceId) || edgeVoiceOptions[0];
  const [openSections, setOpenSections] = useState<OpenSections>({
    ollama: true,
    theme: true,
    effects: true,
    audio: true,
    voice: true,
    backend: true,
    grid: true,
    advanced: true,
  });

  const toggleSection = (key: keyof typeof openSections) =>
    setOpenSections(prev => ({ ...prev, [key]: !prev[key] }));

  useEffect(() => {
    const onHashChange = () => {
      setWorkspacePage(window.location.hash === '#timeline' ? 'timeline' : 'studio');
    };
    window.addEventListener('hashchange', onHashChange);
    return () => window.removeEventListener('hashchange', onHashChange);
  }, []);

  useEffect(() => {
    const currentHash = window.location.hash;
    const desiredHash = workspacePage === 'timeline' ? '#timeline' : '#studio';
    if (currentHash !== desiredHash) {
      window.history.replaceState(null, '', `${window.location.pathname}${window.location.search}${desiredHash}`);
    }
  }, [workspacePage]);

  useEffect(() => {
    if (!videoUrl) {
      setTimelineEditMode('pre');
    }
  }, [videoUrl]);

  const maxScenes = videoLength === '1m' ? 6 : 20; // estimativa: ~10s/cena ou ~15s/cena
  const CARD_W = 288;
  const CARD_H = 340;
  const CARD_H_COLLAPSED = 188;
  const GRID_COLS = 4;
  const GRID_GAP = 24;
  const CANVAS_PAD = 24;
  const MIN_CANVAS_W = 1100;
  const MIN_CANVAS_H = 520;

  const [canvasZoom, setCanvasZoom] = useState<number>(() => {
    const v = Number(localStorage.getItem('vv_canvas_zoom') || '');
    return Number.isFinite(v) && v >= 0.5 && v <= 1.6 ? v : 0.9;
  });
  const [scenePositions, setScenePositions] = useState<Record<string, { x: number; y: number }>>({});
  const [collapsedScenes, setCollapsedScenes] = useState<Record<string, boolean>>({});
  const [draggingSceneId, setDraggingSceneId] = useState<string | null>(null);
  const dragRef = useRef<{ id: string; startX: number; startY: number; origX: number; origY: number } | null>(null);
  const canvasScrollRef = useRef<HTMLDivElement | null>(null);
  const [gridRows, setGridRows] = useState(2);
  const [gridCols, setGridCols] = useState(3);
  const [gridFile, setGridFile] = useState<string | null>(null);
  const [gridTiles, setGridTiles] = useState<GridTile[]>([]);
  const [gridProcessing, setGridProcessing] = useState(false);
  const [gridError, setGridError] = useState<string | null>(null);
  const [gridSceneTarget, setGridSceneTarget] = useState<string | null>(null);
  const [isSessionReady, setIsSessionReady] = useState(false);
  const sessionBootRef = useRef(false);

  // resolve API base (auto fallback, scans common ports)
  useEffect(() => {
    let cancelled = false;
    const resolveApi = async () => {
      const requiredRevision = '2026-02-17-audiofix-1';
      const saved = localStorage.getItem('vv_api_base');
      const envBase = (import.meta as any).env?.API_HOST || (import.meta as any).env?.VITE_API_HOST;
      const portRange = Array.from({ length: 11 }, (_, i) => 8000 + i);
      const portCandidates = portRange.flatMap(p => [`http://127.0.0.1:${p}`, `http://localhost:${p}`]);

      // Prioriza checar direto nos ports comuns antes de tentar o proxy do dev server.
      const rawCandidates = [
        envBase,
        saved,
        ...portCandidates,
        window.location.origin,
      ].filter(Boolean) as string[];
      const seen = new Set<string>();
      const candidates = rawCandidates.filter((candidate) => {
        if (seen.has(candidate)) return false;
        seen.add(candidate);
        return true;
      });
      let fallbackOkBase: string | null = null;

      for (const base of candidates) {
        try {
          const ctrl = new AbortController();
          const timer = setTimeout(() => ctrl.abort(), 1200);
          const res = await fetch(`${base}/api/ping`, { signal: ctrl.signal });
          clearTimeout(timer);
          if (res.ok) {
            const data = await res.json().catch(() => ({}));
            if (data?.revision === requiredRevision) {
              if (!cancelled) {
                setApiBase(base);
                localStorage.setItem('vv_api_base', base);
              }
              return;
            }
            if (!fallbackOkBase) fallbackOkBase = base;
          }
        } catch {
          continue;
        }
      }
      if (fallbackOkBase) {
        if (!cancelled) {
          setApiBase(fallbackOkBase);
          localStorage.setItem('vv_api_base', fallbackOkBase);
        }
        return;
      }
      setApiBase('');
    };
    resolveApi();
    return () => {
      cancelled = true;
    };
  }, []);

  const apiFetch = (path: string, options?: RequestInit) => {
    if (!apiBase) throw new Error('API indisponivel');
    return fetch(`${apiBase}${path}`, options);
  };

  const ensureApiOrWarn = () => {
    if (apiBase) return true;
    const now = Date.now();
    // evita spam: so avisa a cada 2 minutos
    if (now - backendWarnRef.current > 120000) {
      backendWarnRef.current = now;
      const hint = 'Backend nao encontrado. Inicie: cd video_factory; .\\.venv\\Scripts\\Activate.ps1; uvicorn api:app --reload --port 8000 (ou use start_all.bat)';
      setStatusMessage('API indisponivel para requisicoes.');
      pushLog(hint, 'warn');
    }
    return false;
  };

  const pushLog = (msg: string, level: 'info' | 'warn' | 'error' = 'info') => {
    const time = new Date().toLocaleTimeString();
    setLogFeed(prev => [{ id: Math.random().toString(36).slice(2), msg, level, time }, ...prev].slice(0, 12));
  };

  const clampZoom = (z: number) => Math.min(1.6, Math.max(0.5, z));
  const zoomIn = () => setCanvasZoom(z => clampZoom(Number((z + 0.1).toFixed(2))));
  const zoomOut = () => setCanvasZoom(z => clampZoom(Number((z - 0.1).toFixed(2))));

  const resetCanvasView = () => {
    setCanvasZoom(1);
    requestAnimationFrame(() => canvasScrollRef.current?.scrollTo({ left: 0, top: 0, behavior: 'smooth' }));
  };

  const toggleSceneCollapse = (sceneId: string) => {
    setCollapsedScenes(prev => ({ ...prev, [sceneId]: !prev[sceneId] }));
  };

  const autoLayoutScenes = () => {
    if (!script) return;
    setScenePositions(() => {
      const next: Record<string, { x: number; y: number }> = {};
      const rowHeights: number[] = [];

      script.scenes.forEach((s, idx) => {
        const row = Math.floor(idx / GRID_COLS);
        const h = collapsedScenes[s.id] ? CARD_H_COLLAPSED : CARD_H;
        rowHeights[row] = Math.max(rowHeights[row] || 0, h);
      });

      const rowOffsets: number[] = [];
      let accY = CANVAS_PAD;
      rowHeights.forEach((h, row) => {
        rowOffsets[row] = accY;
        accY += h + GRID_GAP;
      });

      script.scenes.forEach((s, idx) => {
        const col = idx % GRID_COLS;
        const row = Math.floor(idx / GRID_COLS);
        next[s.id] = {
          x: CANVAS_PAD + col * (CARD_W + GRID_GAP),
          y: rowOffsets[row] ?? CANVAS_PAD,
        };
      });
      return next;
    });
    pushLog('Cards organizados (auto layout)', 'info');
  };

  // --- General Settings ---
  const [settings, setSettings] = useState<GeneralSettings>({
    theme: 'Cinematographic',
    backgroundMusic: 'Audio Local',
    voiceName: 'Kore',
    defaultTransition: 'fade',
    defaultFilter: 'cinematic',
  });

  const sceneIdKey = script ? script.scenes.map(s => s.id).join('|') : '';

  const applySessionPayload = (payload: SessionPayload | null): boolean => {
    if (!payload) return false;
    setTopic(payload.topic ?? '');
    setFormat(payload.format ?? '16:9');
    setVideoLength(payload.videoLength ?? '1m');
    setSettings(prev => payload.settings ? { ...prev, ...payload.settings } : prev);
    if (payload.currentStep && stepOrder.includes(payload.currentStep)) setCurrentStep(payload.currentStep);
    if (payload.ffmpegFilters !== undefined) setFfmpegFilters(payload.ffmpegFilters);
    if (payload.stabilize !== undefined) setStabilize(payload.stabilize);
    if (payload.aiEnhance !== undefined) setAiEnhance(payload.aiEnhance);
    if (payload.engine) setEngine(payload.engine);
    if (payload.ollamaHost) setOllamaHost(payload.ollamaHost);
    if (payload.ollamaModel) setOllamaModel(payload.ollamaModel);
    if (payload.videoUrl !== undefined) setVideoUrl(toPlayableVideoUrl(payload.videoUrl || null));
    if (payload.timelineEditMode) setTimelineEditMode(payload.timelineEditMode);
    else if (payload.videoUrl) setTimelineEditMode('post');
    if (payload.captionScale !== undefined) setCaptionScale(Math.max(0.6, Math.min(1.6, Number(payload.captionScale) || 1)));
    if (payload.captionBg !== undefined) setCaptionBg(Math.max(0, Math.min(1, Number(payload.captionBg) || 0.55)));
    if (payload.captionColor) setCaptionColor(payload.captionColor);
    if (payload.captionHighlight) setCaptionHighlight(payload.captionHighlight);
    if (payload.captionY !== undefined) setCaptionY(Math.max(50, Math.min(95, Number(payload.captionY) || 82)));
    if (payload.musicVolume !== undefined) setMusicVolume(Math.max(0, Number(payload.musicVolume) || 0));
    if (payload.narrationVolume !== undefined) setNarrationVolume(Math.max(0, Number(payload.narrationVolume) || 0));
    if (payload.colorStrength !== undefined) setColorStrength(Math.max(0, Number(payload.colorStrength) || 0));
    if (payload.transitionDuration !== undefined) setTransitionDuration(Math.max(0.05, Number(payload.transitionDuration) || 0.6));
    if (payload.imageScale !== undefined) setImageScale(Math.max(0.8, Math.min(1.2, Number(payload.imageScale) || 1)));
    if (payload.previewText !== undefined) setPreviewText(payload.previewText || '');
    if (payload.previewTransition) setPreviewTransition(payload.previewTransition);
    if (payload.previewFilter) setPreviewFilter(payload.previewFilter);
    if (payload.previewAnimation) setPreviewAnimation(payload.previewAnimation);
    if (payload.previewColorStrength !== undefined) setPreviewColorStrength(Math.max(0, Number(payload.previewColorStrength) || 0));
    if (payload.previewFilters !== undefined) setPreviewFilters(payload.previewFilters || '');
    if (payload.previewUrl !== undefined) setPreviewUrl(payload.previewUrl || null);
    if (payload.cleanupAgeDays !== undefined) setCleanupAgeDays(Math.max(1, Math.floor(Number(payload.cleanupAgeDays) || 7)));
    if (payload.canvasZoom !== undefined) setCanvasZoom(clampZoom(Number(payload.canvasZoom) || 0.9));
    if (payload.openSections) setOpenSections(prev => ({ ...prev, ...payload.openSections }));
    setScenePositions(payload.scenePositions || {});
    setCollapsedScenes(payload.collapsedScenes || {});
    setGridRows(payload.gridRows || 2);
    setGridCols(payload.gridCols || 3);
    setGridSceneTarget(payload.gridSceneTarget || null);
    if (payload.gridFile !== undefined) setGridFile(payload.gridFile || null);
    if (payload.edgeVoiceId) setEdgeVoiceId(payload.edgeVoiceId);
    if (payload.script) {
      const scenesWithStatus = payload.script.scenes.map((s, idx) => ({
        ...s,
        id: String((s as any).id ?? idx + 1),
        status: s.status ?? 'completed',
        narrationVolume: Math.max(0, Math.min(2, Number((s as any).narrationVolume) || 1)),
        trimStartMs: Math.max(0, Math.min(5000, Math.floor(Number((s as any).trimStartMs) || 0))),
        trimEndMs: Math.max(0, Math.min(5000, Math.floor(Number((s as any).trimEndMs) || 0))),
        audioOffsetMs: Math.max(-3000, Math.min(3000, Math.floor(Number((s as any).audioOffsetMs) || 0))),
        transition: s.transition || payload.settings?.defaultTransition || settings.defaultTransition || 'fade',
        filter: s.filter || payload.settings?.defaultFilter || settings.defaultFilter || 'none',
        localSfx: typeof (s as any).localSfx === 'string' ? (s as any).localSfx : undefined,
        sfxVolume: Math.max(0, Math.min(2, Number((s as any).sfxVolume) || 0.35)),
        flowTo: typeof s.flowTo === 'string' ? String(s.flowTo) : null,
      }));
      setScript({ ...payload.script, scenes: scenesWithStatus });
      if (!payload.currentStep || !stepOrder.includes(payload.currentStep)) {
        setCurrentStep('assets');
      }
    } else {
      setScript(null);
      if (!payload.currentStep || !stepOrder.includes(payload.currentStep)) {
        setCurrentStep('topic');
      }
    }
    return true;
  };

  const loadSessionSerialized = async (key: SessionStoreKey): Promise<string | null> => {
    if (hasIndexedDb()) {
      try {
        const fromDb = await idbGetText(key);
        if (fromDb) {
          try {
            localStorage.setItem(key, fromDb);
          } catch {
            // ignore localStorage quota
          }
          return fromDb;
        }
      } catch {
        // ignore IndexedDB read errors and fallback to localStorage
      }
    }
    return localStorage.getItem(key);
  };

  useEffect(() => {
    if (sessionBootRef.current) return;
    sessionBootRef.current = true;
    let cancelled = false;
    const boot = async () => {
      const stored = await loadSessionSerialized(SESSION_KEY);
      if (!stored) {
        if (!cancelled) setIsSessionReady(true);
        return;
      }
      try {
        const payload = JSON.parse(stored) as SessionPayload;
        if (!cancelled && applySessionPayload(payload)) {
          pushLog('Sessao restaurada automaticamente', 'info');
        }
      } catch (err: any) {
        if (!cancelled) pushLog(`Falha ao restaurar sessao automatica: ${err?.message || err}`, 'warn');
      } finally {
        if (!cancelled) setIsSessionReady(true);
      }
    };
    void boot();
    return () => {
      cancelled = true;
    };
  }, []);

  const persistSessionPayload = (payload: SessionPayload) => {
    const serialized = JSON.stringify(payload);
    try {
      const current = localStorage.getItem(SESSION_KEY);
      if (current && current !== serialized) {
        localStorage.setItem(SESSION_PREV_KEY, current);
      }
      if (current !== serialized) {
        localStorage.setItem(SESSION_KEY, serialized);
      }
    } catch {
      // ignore localStorage quota
    }
    if (!hasIndexedDb()) return;
    void (async () => {
      try {
        const current = await idbGetText(SESSION_KEY);
        if (current === serialized) return;
        if (current) {
          await idbSetText(SESSION_PREV_KEY, current);
        }
        await idbSetText(SESSION_KEY, serialized);
      } catch {
        // ignore IndexedDB errors
      }
    })();
  };

  const canvasBounds = useMemo(() => {
    if (!script || script.scenes.length === 0) return { w: MIN_CANVAS_W, h: MIN_CANVAS_H };
    let maxX = 0;
    let maxY = 0;
    for (const s of script.scenes) {
      const pos = scenePositions[s.id] || { x: 0, y: 0 };
      const cardH = collapsedScenes[s.id] ? CARD_H_COLLAPSED : CARD_H;
      if (pos.x > maxX) maxX = pos.x;
      if (pos.y + cardH > maxY) maxY = pos.y + cardH;
    }
    const w = Math.max(MIN_CANVAS_W, maxX + CARD_W + CANVAS_PAD);
    const h = Math.max(MIN_CANVAS_H, maxY + CANVAS_PAD);
    return { w, h };
  }, [script, scenePositions, collapsedScenes]);

  useEffect(() => {
    localStorage.setItem('vv_canvas_zoom', String(canvasZoom));
  }, [canvasZoom]);

  useEffect(() => {
    if (!livePreview) return;
    const t = setTimeout(() => setLivePreview(null), 1200);
    return () => clearTimeout(t);
  }, [livePreview]);

  useEffect(() => {
    if (!script) {
      setScenePositions({});
      setCollapsedScenes({});
      return;
    }

    setScenePositions(prev => {
      const next: Record<string, { x: number; y: number }> = {};

      // keep positions for existing scenes
      for (const s of script.scenes) {
        if (prev[s.id]) next[s.id] = prev[s.id];
      }

      // assign default positions for new scenes
      script.scenes.forEach((s, idx) => {
        if (next[s.id]) return;
        const col = idx % GRID_COLS;
        const row = Math.floor(idx / GRID_COLS);
        next[s.id] = {
          x: CANVAS_PAD + col * (CARD_W + GRID_GAP),
          y: CANVAS_PAD + row * (CARD_H + GRID_GAP),
        };
      });

      return next;
    });
  }, [sceneIdKey]);

  useEffect(() => {
    if (!script) {
      setCollapsedScenes({});
      return;
    }
    setCollapsedScenes(prev => {
      const next: Record<string, boolean> = {};
      for (const s of script.scenes) {
        next[s.id] = !!prev[s.id];
      }
      return next;
    });
  }, [sceneIdKey]);

  useEffect(() => {
    if (!script || !script.scenes.length) {
      setGridSceneTarget(null);
      return;
    }
    setGridSceneTarget(prev => script.scenes.find(s => s.id === prev)?.id || script.scenes[0].id);
  }, [sceneIdKey]);

  useEffect(() => {
    if (!isSessionReady) return;
    const payload: SessionPayload = {
      topic,
      format,
      videoLength,
      ffmpegFilters,
      stabilize,
      aiEnhance,
      engine,
      settings,
      script,
      scenePositions,
      gridRows,
      gridCols,
      gridSceneTarget,
      edgeVoiceId,
      currentStep,
      videoUrl,
      timelineEditMode,
      ollamaHost,
      ollamaModel,
      captionScale,
      captionBg,
      captionColor,
      captionHighlight,
      captionY,
      musicVolume,
      narrationVolume,
      colorStrength,
      transitionDuration,
      imageScale,
      previewText,
      previewTransition,
      previewFilter,
      previewAnimation,
      previewColorStrength,
      previewFilters,
      previewUrl,
      cleanupAgeDays,
      canvasZoom,
      openSections,
      gridFile,
      collapsedScenes,
    };
    persistSessionPayload(payload);
  }, [
    isSessionReady,
    topic,
    format,
    videoLength,
    ffmpegFilters,
    stabilize,
    aiEnhance,
    engine,
    settings,
    script,
    scenePositions,
    gridRows,
    gridCols,
    gridSceneTarget,
    edgeVoiceId,
    currentStep,
    videoUrl,
    timelineEditMode,
    ollamaHost,
    ollamaModel,
    captionScale,
    captionBg,
    captionColor,
    captionHighlight,
    captionY,
    musicVolume,
    narrationVolume,
    colorStrength,
    transitionDuration,
    imageScale,
    previewText,
    previewTransition,
    previewFilter,
    previewAnimation,
    previewColorStrength,
    previewFilters,
    previewUrl,
    cleanupAgeDays,
    canvasZoom,
    openSections,
    gridFile,
    collapsedScenes,
  ]);

  const handleCanvasWheel = (e: React.WheelEvent<HTMLDivElement>) => {
    // Ctrl+scroll: zoom (trackpad pinch also triggers ctrlKey on many browsers)
    if (!e.ctrlKey) return;
    e.preventDefault();
    const direction = e.deltaY > 0 ? -1 : 1;
    setCanvasZoom(z => clampZoom(Number((z + direction * 0.05).toFixed(2))));
  };

  const beginSceneDrag = (sceneId: string, e: React.PointerEvent<HTMLDivElement>) => {
    if (e.button !== 0) return;
    const target = e.target as HTMLElement | null;
    if (target?.closest('button, input, textarea, select, label, a')) return;
    e.preventDefault();
    const pos = scenePositions[sceneId] || { x: CANVAS_PAD, y: CANVAS_PAD };
    dragRef.current = { id: sceneId, startX: e.clientX, startY: e.clientY, origX: pos.x, origY: pos.y };
    setDraggingSceneId(sceneId);
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
  };

  const moveSceneDrag = (e: React.PointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current;
    if (!drag) return;
    const dx = (e.clientX - drag.startX) / canvasZoom;
    const dy = (e.clientY - drag.startY) / canvasZoom;
    setScenePositions(prev => ({
      ...prev,
      [drag.id]: {
        x: Math.max(0, drag.origX + dx),
        y: Math.max(0, drag.origY + dy),
      },
    }));
  };

  const endSceneDrag = () => {
    if (!dragRef.current) return;
    dragRef.current = null;
    setDraggingSceneId(null);
  };

  const addSceneCard = () => {
    if (!script) return;
    if (script.scenes.length >= maxScenes) return;
    const newId = Math.random().toString(36).substr(2, 9);
    const newScene: Scene = {
      id: newId,
      text: 'Nova narracao',
      visualPrompt: 'Nova descricao',
      narrationVolume: 1,
      trimStartMs: 0,
      trimEndMs: 0,
      audioOffsetMs: 0,
      sfxVolume: 0.35,
      status: 'completed',
      transition: settings.defaultTransition,
      filter: settings.defaultFilter,
      flowTo: null,
    };
    setScript({ ...script, scenes: [...script.scenes, newScene] });

    const idx = script.scenes.length;
    const col = idx % GRID_COLS;
    const row = Math.floor(idx / GRID_COLS);
    setScenePositions(prev => ({
      ...prev,
      [newId]: {
        x: CANVAS_PAD + col * (CARD_W + GRID_GAP),
        y: CANVAS_PAD + row * (CARD_H + GRID_GAP),
      },
    }));
  };

  useEffect(() => {
    // carrega preferências locais de roteiro
    const storedHost = localStorage.getItem('ollama_host');
    const storedModel = localStorage.getItem('ollama_model');
    if (storedHost) setOllamaHost(storedHost);
    if (storedModel) setOllamaModel(storedModel);

    const storedEdgeVoice = localStorage.getItem('edge_voice');
    if (storedEdgeVoice) setEdgeVoiceId(storedEdgeVoice);

    // carrega vozes do backend
    const loadVoices = async () => {
      try {
      const res = await apiFetch('/api/tts/voices');
        if (!res.ok) throw new Error('Falha ao listar vozes');
        const data = await res.json();
        const desiredVoice = storedEdgeVoice || edgeVoiceId;
        const voices = (data?.voices || [])
          .map((v: any) => ({
            id: String(v.id || ''),
            label: String(v.label || v.id || ''),
            gender: String(v.gender || 'Unknown'),
            locale: String(v.locale || ''),
          }))
          .filter((v: any) => v.id);

        voices.sort((a: any, b: any) => {
          const localeRank = (l: string) => (l === 'pt-BR' ? 0 : l === 'pt-PT' ? 1 : 2);
          const la = localeRank(String(a.locale || ''));
          const lb = localeRank(String(b.locale || ''));
          if (la !== lb) return la - lb;

          const ga = a.gender === 'Male' ? 0 : a.gender === 'Female' ? 1 : 2;
          const gb = b.gender === 'Male' ? 0 : b.gender === 'Female' ? 1 : 2;
          if (ga !== gb) return ga - gb;
          return String(a.label).localeCompare(String(b.label));
        });

        setEdgeVoices(voices);
        if (voices.length && !voices.some((v: any) => v.id === desiredVoice)) setEdgeVoiceId(voices[0].id);
      } catch (err) {
        setEdgeVoices(staticEdgeVoices);
        if (staticEdgeVoices.length) {
          const fallback =
            storedEdgeVoice && staticEdgeVoices.some(v => v.id === storedEdgeVoice)
              ? storedEdgeVoice
              : staticEdgeVoices[0].id;
          setEdgeVoiceId(fallback);
        }
      }
    };
    loadVoices();
  }, []);

  useEffect(() => {
    if (!gridFile) {
      setGridTiles([]);
      setGridError(null);
      setGridProcessing(false);
      return;
    }

    let cancelled = false;
    const processGrid = async () => {
      setGridProcessing(true);
      setGridError(null);
      try {
        const image = await new Promise<HTMLImageElement>((resolve, reject) => {
          const img = new Image();
          img.crossOrigin = 'anonymous';
          img.onload = () => resolve(img);
          img.onerror = (err) => reject(err);
          img.src = gridFile;
        });

        if (gridRows <= 0 || gridCols <= 0) {
          throw new Error('Linhas e colunas devem ser maiores que zero');
        }

        const totalWidth = image.naturalWidth;
        const totalHeight = image.naturalHeight;
        const baseWidth = Math.floor(totalWidth / gridCols);
        const baseHeight = Math.floor(totalHeight / gridRows);
        const widths = Array.from({ length: gridCols }, (_, col) =>
          col === gridCols - 1 ? totalWidth - baseWidth * (gridCols - 1) : baseWidth
        );
        const heights = Array.from({ length: gridRows }, (_, row) =>
          row === gridRows - 1 ? totalHeight - baseHeight * (gridRows - 1) : baseHeight
        );

        const colOffsets = widths.reduce<number[]>((acc, width, index) => {
          acc[index] = index === 0 ? 0 : acc[index - 1] + widths[index - 1];
          return acc;
        }, []);
        const rowOffsets = heights.reduce<number[]>((acc, height, index) => {
          acc[index] = index === 0 ? 0 : acc[index - 1] + heights[index - 1];
          return acc;
        }, []);

        const tiles: GridTile[] = [];
        for (let row = 0; row < gridRows; row += 1) {
          for (let col = 0; col < gridCols; col += 1) {
            const sw = widths[col];
            const sh = heights[row];
            const sx = colOffsets[col];
            const sy = rowOffsets[row];
            const canvas = document.createElement('canvas');
            canvas.width = sw;
            canvas.height = sh;
            const ctx = canvas.getContext('2d');
            if (!ctx) continue;
            ctx.drawImage(image, sx, sy, sw, sh, 0, 0, sw, sh);
            tiles.push({
              id: `tile-${row}-${col}-${Date.now()}`,
              row,
              col,
              dataUrl: canvas.toDataURL('image/png'),
            });
          }
        }

        if (!cancelled) {
          setGridTiles(tiles);
        }
      } catch (err: any) {
        if (!cancelled) {
          setGridError(err?.message || 'Erro ao processar imagem em grade');
          setGridTiles([]);
        }
      } finally {
        if (!cancelled) {
          setGridProcessing(false);
        }
      }
    };

    processGrid();
    return () => {
      cancelled = true;
    };
  }, [gridFile, gridCols, gridRows]);

  const loadOllamaModels = async () => {
    setIsLoadingModels(true);
    setOllamaError(null);
    try {
      const base = ollamaHost.trim();
      const proxied = base.includes('127.0.0.1') || base.includes('localhost');
      const url = proxied ? `/ollama/api/tags` : `${base.replace(/\/$/, '')}/api/tags`;
      const res = await fetch(url, { method: 'GET' });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      const names = (data?.models || data || []).map((m: any) => m.name).filter(Boolean);
      setOllamaModels(names);
      if (names.length && !names.includes(ollamaModel)) setOllamaModel(names[0]);
      pushLog(`Ollama OK - ${names.length} modelos`, 'info');
    } catch (err: any) {
      setOllamaError(err?.message || 'Falha ao consultar modelos');
      setOllamaModels([]);
      pushLog(`Ollama falhou: ${err?.message || err}`, 'error');
    } finally {
      setIsLoadingModels(false);
    }
  };

  const previewVoice = async () => {
    setIsPreviewingVoice(true);
    try {
      if (!ensureApiOrWarn()) return;
      localStorage.setItem('edge_voice', edgeVoiceId);
      const res = await apiFetch(`/api/tts/preview?text=${encodeURIComponent(previewText || 'Previa de voz')}&voice=${encodeURIComponent(edgeVoiceId)}`);
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err?.detail || `HTTP ${res.status}`);
      }
      const usedVoice = res.headers.get('x-voice-used') || edgeVoiceId;
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const audio = new Audio(url);
      audio.play();
      pushLog(`Previa de voz ${usedVoice}`, 'info');
    } catch (err: any) {
      const msg = err?.message || String(err);
      const isConn = /failed to fetch|network|ECONNREFUSED|API indisponível/i.test(msg);
      if (isConn) {
        // ja avisado por ensureApiOrWarn; nao spammar
        setStatusMessage(prev => prev || 'Inicie o backend para habilitar prévia de voz.');
      } else {
        pushLog(`Falha na prévia de voz: ${msg}`, 'error');
      }
    } finally {
      setIsPreviewingVoice(false);
    }
  };

  const previewVideoEffects = async () => {
    if (isPreviewRendering) return;
    if (!ensureApiOrWarn()) return;
    setIsPreviewRendering(true);
    setStatusMessage('Gerando prévia curta de efeitos...');
    try {
      const payload = {
        format,
        transition: previewTransition,
        filter: previewFilter,
        colorStrength: previewColorStrength,
        animationType: previewAnimation,
        captionText: previewText || 'Prévia de efeitos',
        duration: 3.5,
        ffmpegFilters: previewFilters || undefined,
        stabilize,
        aiEnhance,
        engine,
      };
      const res = await apiFetch('/api/render/preview', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err?.detail || `HTTP ${res.status}`);
      }
      const data = await res.json();
      const url = data.web_url && apiBase ? `${apiBase}${data.web_url}` : data.output;
      setPreviewUrl(url || null);
      pushLog('Prévia de efeitos gerada (3-4s)', 'info');
    } catch (err: any) {
      const msg = err?.message || String(err);
      pushLog(`Falha ao gerar prévia: ${msg}`, 'error');
      setStatusMessage(`Prévia falhou: ${msg}`);
    } finally {
      setIsPreviewRendering(false);
    }
  };

  const handleAiError = (error: any, context: string) => {
    console.error(`${context} failed:`, error);
    const errorMessage = String(error?.message || error || 'Erro desconhecido');
    pushLog(`${context} falhou: ${errorMessage}`, 'error');
    setStatusMessage(`${context} falhou: ${errorMessage}`);
  };

  const cleanupCache = async () => {
    if (!ensureApiOrWarn()) return;
    setIsCleaningCache(true);
    setCleanupResult(null);
    try {
      const days = Math.max(1, Number.isFinite(cleanupAgeDays) ? cleanupAgeDays : 7);
      const res = await apiFetch(`/api/cleanup?max_age_days=${days}`, { method: 'POST' });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err?.detail || `HTTP ${res.status}`);
      }
      const data = await res.json();
      const summary = `Removidos ${data.removed ?? '?'} | Mantidos ${data.kept ?? '?'} | Limite ${data.max_age_days ?? days}d`;
      setCleanupResult(summary);
      pushLog(`Limpeza concluída: ${summary}`, 'info');
    } catch (error: any) {
      handleAiError(error, 'Limpeza de cache');
    } finally {
      setIsCleaningCache(false);
    }
  };

  const restoreSession = async () => {
    const stored = await loadSessionSerialized(SESSION_KEY);
    if (!stored) {
      pushLog('Nenhuma sessao salva', 'warn');
      return;
    }
    try {
      const payload = JSON.parse(stored) as SessionPayload;
      if (applySessionPayload(payload)) {
        pushLog('Sessao restaurada', 'info');
      }
    } catch (err: any) {
      pushLog(`Erro ao restaurar sessao: ${err?.message || err}`, 'error');
    }
  };

  const restorePreviousSession = async () => {
    const stored = await loadSessionSerialized(SESSION_PREV_KEY);
    if (!stored) {
      pushLog('Nenhuma sessao anterior salva', 'warn');
      return;
    }
    try {
      const payload = JSON.parse(stored) as SessionPayload;
      if (applySessionPayload(payload)) {
        pushLog('Sessao anterior restaurada', 'info');
      }
    } catch (err: any) {
      pushLog(`Erro ao restaurar sessao anterior: ${err?.message || err}`, 'error');
    }
  };

  const buildSceneVisualPrompt = (sceneText: string, sceneIndex: number): string => {
    const cleanText = String(sceneText || '').replace(/\s+/g, ' ').trim();
    const narrative = cleanText || `CENA ${sceneIndex + 1}`;
    return `Ilustracao cinematografica (${settings.theme}) da cena ${sceneIndex + 1}: ${narrative}. Composicao detalhada, iluminacao dramatica, profundidade de campo, enquadramento claro, atmosfera coerente com o roteiro, sem texto na imagem, alta qualidade.`;
  };

  const buildSceneVisualPromptEn = (sceneText: string, sceneIndex: number): string => {
    const cleanText = String(sceneText || '').replace(/\s+/g, ' ').trim();
    const narrative = cleanText || `SCENE ${sceneIndex + 1}`;
    return `Cinematic illustration (${settings.theme}) for scene ${sceneIndex + 1}: ${narrative}. Detailed composition, dramatic lighting, depth of field, clear framing, mood consistent with the storyline, no text in image, high quality.`;
  };

  const getFormatGuide = (value: '16:9' | '9:16') => (
    value === '9:16'
      ? { target: '1080x1920', max: '2160x3840' }
      : { target: '1920x1080', max: '3840x2160' }
  );

  const buildAllScenePromptsText = (lang: 'pt' | 'en' = 'pt') => {
    if (!script?.scenes?.length) return '';
    const guide = getFormatGuide(format);
    const projectTitle = String(script.title || '').trim() || 'Projeto sem titulo';
    const projectTopic = String(topic || '').trim() || projectTitle;
    const style = String(settings.theme || '').trim() || 'Cinematographic';

    const lines: string[] = lang === 'en'
      ? [
          `PROJECT: ${projectTitle}`,
          `TOPIC: ${projectTopic}`,
          `GLOBAL STYLE: ${style}`,
          `FINAL VIDEO FORMAT: ${format} (${guide.target})`,
          `MAX RECOMMENDED RESOLUTION: ${guide.max}`,
          '',
          'GENERAL IMAGE GENERATION INSTRUCTIONS:',
          `- Keep ${format} aspect ratio for every image.`,
          '- No text, no logos, no watermark, no borders.',
          '- Cinematic visual language, rich details, clear subject and depth.',
          '- Keep visual consistency across scenes (palette, style and lighting).',
          '- Export as high-quality PNG or JPG.',
          '',
          'SCENE PROMPTS:',
          '',
        ]
      : [
          `PROJETO: ${projectTitle}`,
          `TEMA: ${projectTopic}`,
          `ESTILO GLOBAL: ${style}`,
          `FORMATO FINAL DE VIDEO: ${format} (${guide.target})`,
          `RESOLUCAO MAXIMA RECOMENDADA: ${guide.max}`,
          '',
          'INSTRUCOES GERAIS PARA GERAR AS IMAGENS:',
          `- Manter proporcao ${format} em todas as imagens.`,
          '- Sem texto, sem logos, sem watermark e sem bordas.',
          '- Linguagem cinematografica, alto nivel de detalhe e boa profundidade.',
          '- Manter consistencia visual entre as cenas (paleta, estilo e iluminacao).',
          '- Exportar em PNG ou JPG de alta qualidade.',
          '',
          'PROMPTS POR CENA:',
          '',
        ];

    script.scenes.forEach((scene, idx) => {
      const prompt = lang === 'en'
        ? buildSceneVisualPromptEn(scene.text, idx)
        : (String(scene.visualPrompt || '').trim() || buildSceneVisualPrompt(scene.text, idx));
      lines.push(lang === 'en' ? `SCENE ${idx + 1}:` : `CENA ${idx + 1}:`);
      lines.push(prompt);
      lines.push('');
    });

    return lines.join('\n');
  };

  const generateScript = async () => {
    if (!topic.trim()) return;
    setIsGeneratingScript(true);
    setCurrentStep('scripting');
    setStatusMessage('IA projetando roteiro...');
    pushLog('Iniciando geracao de roteiro', 'info');

    const normalizeSceneText = (value: string) =>
      String(value || '')
        .toLowerCase()
        .replace(/[^\p{L}\p{N}\s]/gu, ' ')
        .replace(/\s+/g, ' ')
        .trim();

    const tokenSet = (value: string) =>
      new Set(
        normalizeSceneText(value)
          .split(' ')
          .filter((t) => t.length > 2)
      );

    const jaccard = (a: Set<string>, b: Set<string>) => {
      if (!a.size || !b.size) return 0;
      let inter = 0;
      for (const item of a) {
        if (b.has(item)) inter += 1;
      }
      return inter / (a.size + b.size - inter);
    };

    const hasLowDiversity = (scenes: Array<{ text?: string; visualPrompt?: string }>) => {
      if (!scenes || scenes.length < 2) return false;
      const texts = scenes.map((s) => normalizeSceneText(s.text || ''));
      const visuals = scenes.map((s) => normalizeSceneText(s.visualPrompt || ''));
      const uniqueTexts = new Set(texts.filter(Boolean)).size;
      const uniqueVisuals = new Set(visuals.filter(Boolean)).size;

      if (uniqueTexts / scenes.length < 0.75) return true;
      if (uniqueVisuals / scenes.length < 0.75) return true;

      for (let i = 0; i < scenes.length; i += 1) {
        for (let j = i + 1; j < scenes.length; j += 1) {
          const textSim = jaccard(tokenSet(scenes[i].text || ''), tokenSet(scenes[j].text || ''));
          const visualSim = jaccard(tokenSet(scenes[i].visualPrompt || ''), tokenSet(scenes[j].visualPrompt || ''));
          if (textSim > 0.72 || visualSim > 0.72) {
            return true;
          }
        }
      }
      return false;
    };

    const buildPrompt = (attempt: number) => `Voce e um roteirista criativo para videos curtos.
Tema central: "${topic}".
Direcao visual global: "${settings.theme}".

Objetivo:
- Criar no maximo ${maxScenes} cenas.
- Cada cena deve trazer uma ideia nova, sem copiar texto de outras cenas.
- Nao repetir a mesma frase-base em todas as cenas.
- Narrativa com progressao: abertura -> desenvolvimento -> virada -> fechamento.
- Variar cenario, foco, angulo e detalhe visual entre cenas.
- Evitar texto generico e evitar repeticao literal do tema.

Regras de saida (obrigatorio):
- Responda SOMENTE JSON valido.
- Estrutura: {"title":"...","scenes":[{"id":"1","text":"...","visualPrompt":"..."}]}.
- "text" deve ser narracao da cena.
- "visualPrompt" deve descrever imagem/video da cena com detalhes diferentes das outras.
- Nao inclua markdown, comentarios ou texto fora do JSON.
${attempt > 1 ? '- IMPORTANTE: a tentativa anterior ficou repetitiva. Aumente ainda mais a variedade entre cenas.' : ''}`;

    try {
      localStorage.setItem('ollama_host', ollamaHost);
      localStorage.setItem('ollama_model', ollamaModel);

      let data: any = null;
      let lastError: any = null;

      for (let attempt = 1; attempt <= 3; attempt += 1) {
        const response = await fetch(`${ollamaHost.replace(/\/$/, '')}/api/generate`, {
          method: 'POST',
          body: JSON.stringify({
            model: ollamaModel,
            prompt: buildPrompt(attempt),
            stream: false,
            format: 'json',
          }),
        });

        if (!response.ok) {
          throw new Error(`Ollama HTTP ${response.status}`);
        }

        const resData = await response.json();
        const parsed = JSON.parse(String(resData.response || '{}'));
        if (!parsed?.title || !Array.isArray(parsed?.scenes) || parsed.scenes.length === 0) {
          lastError = new Error('Resposta sem formato de roteiro valido');
          continue;
        }

        const candidateScenes = parsed.scenes.slice(0, maxScenes).map((s: any, idx: number) => ({
          id: String(s.id ?? idx + 1),
          text: String(s.text || '').trim(),
          visualPrompt: String(s.visualPrompt || '').trim() || buildSceneVisualPrompt(String(s.text || ''), idx),
        }));

        if (hasLowDiversity(candidateScenes)) {
          lastError = new Error('Roteiro repetitivo');
          pushLog(`Roteiro repetitivo detectado (tentativa ${attempt}/3), regenerando...`, 'warn');
          continue;
        }

        data = { ...parsed, scenes: candidateScenes };
        break;
      }

      if (!data) {
        throw lastError || new Error('Nao foi possivel gerar roteiro criativo');
      }

      const scenesWithStatus = data.scenes.map((s: any, idx: number) => ({
        ...s,
        id: String(s.id ?? idx + 1),
        status: 'completed',
        narrationVolume: Math.max(0, Math.min(2, Number(s.narrationVolume) || 1)),
        trimStartMs: Math.max(0, Math.min(5000, Math.floor(Number(s.trimStartMs) || 0))),
        trimEndMs: Math.max(0, Math.min(5000, Math.floor(Number(s.trimEndMs) || 0))),
        audioOffsetMs: Math.max(-3000, Math.min(3000, Math.floor(Number(s.audioOffsetMs) || 0))),
        localSfx: typeof s.localSfx === 'string' ? s.localSfx : undefined,
        sfxVolume: Math.max(0, Math.min(2, Number(s.sfxVolume) || 0.35)),
        transition: settings.defaultTransition,
        filter: settings.defaultFilter,
        flowTo: s.flowTo ? String(s.flowTo) : null,
      }));
      setScript({ ...data, scenes: scenesWithStatus });
      setCurrentStep('assets');
      pushLog(`Roteiro pronto (${data.scenes.length} cenas)`, 'info');
    } catch (error: any) {
      setCurrentStep('topic');
      handleAiError(error, 'Roteiro');
    } finally {
      setIsGeneratingScript(false);
      setStatusMessage('');
    }
  };
  const blobToDataUrl = (blob: Blob): Promise<string> =>
    new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => {
        if (typeof reader.result === 'string') {
          resolve(reader.result);
        } else {
          reject(new Error('Falha ao converter áudio'));
        }
      };
      reader.onerror = () => reject(new Error('Falha ao ler áudio'));
      reader.readAsDataURL(blob);
    });

  const generateNarration = async (sceneId: string, text: string) => {
    if (!apiBase) {
      setStatusMessage('API indisponível para prévia de voz');
      return;
    }
    setIsGeneratingNarration(sceneId);
    try {
      const res = await apiFetch(
        `/api/tts/preview?text=${encodeURIComponent(text)}&voice=${encodeURIComponent(edgeVoiceId)}`
      );
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err?.detail || `HTTP ${res.status}`);
      }
      const blob = await res.blob();
      const dataUrl = await blobToDataUrl(blob);
      updateScene(sceneId, { narrationAudio: dataUrl });
      pushLog(`Narracao gerada para cena ${sceneId}`, 'info');
    } catch (error: any) {
      handleAiError(error, 'Narracao');
    } finally {
      setIsGeneratingNarration(null);
    }
  };

  const playNarration = (source: string) => {
    const audio = new Audio(source);
    audio.play();
  };

  const updateScene = (id: string, updates: Partial<Scene>) => {
    setScript(prev => {
      if (!prev) return prev;
      return {
        ...prev,
        scenes: prev.scenes.map(s => (s.id === id ? { ...s, ...updates } : s)),
      };
    });
  };

  const deleteScene = (id: string) => {
    setScript(prev => {
      if (!prev) return prev;
      return { ...prev, scenes: prev.scenes.filter(s => s.id !== id) };
    });
    setScenePositions(prev => {
      const next = { ...prev };
      delete next[id];
      return next;
    });
  };

  const duplicateScene = (scene: Scene) => {
    const newScene = {
      ...scene,
      id: Math.random().toString(36).substr(2, 9),
      narrationAudio: undefined,
      narrationVolume: scene.narrationVolume ?? 1,
      trimStartMs: scene.trimStartMs ?? 0,
      trimEndMs: scene.trimEndMs ?? 0,
      audioOffsetMs: scene.audioOffsetMs ?? 0,
      flowTo: null,
    };
    setScript(prev => {
      if (!prev) return prev;
      const newScenes = [...prev.scenes];
      const index = newScenes.findIndex(s => s.id === scene.id);
      newScenes.splice(index + 1, 0, newScene);
      return { ...prev, scenes: newScenes };
    });
    setScenePositions(prev => {
      const base = prev[scene.id] || { x: CANVAS_PAD, y: CANVAS_PAD };
      return {
        ...prev,
        [newScene.id]: { x: base.x + 24, y: base.y + 24 },
      };
    });
  };

  const applyEffectsToAllScenes = () => {
    setScript(prev => {
      if (!prev) return prev;
      return {
        ...prev,
        scenes: prev.scenes.map(s => ({
          ...s,
          transition: settings.defaultTransition,
          filter: settings.defaultFilter,
        })),
      };
    });
  };

  const handleFileUpload = (type: 'scene_image' | 'scene_sfx' | 'global_theme' | 'global_music', id: string, file: File) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      const result = reader.result as string;
      if (type === 'scene_image') {
        updateScene(id, { localImage: result });
      } else if (type === 'scene_sfx') {
        updateScene(id, { localSfx: result, sfxVolume: 0.35 });
      } else if (type === 'global_theme') {
        setSettings({ ...settings, localThemeRef: result });
      } else if (type === 'global_music') {
        setSettings({ ...settings, localBackgroundMusic: result });
      }
    };
    reader.readAsDataURL(file);
  };

  const handleGridUpload = (file: File | null) => {
    if (!file) {
      setGridFile(null);
      return;
    }
    const reader = new FileReader();
    reader.onload = () => {
      setGridFile(String(reader.result || ''));
    };
    reader.onerror = (err) => {
      setGridFile(null);
      pushLog(`Falha ao carregar grade: ${err}`,'error');
    };
    reader.readAsDataURL(file);
  };

  const copyScenePrompt = async (scene: Scene, sceneIndex: number) => {
    if (!navigator.clipboard) {
      pushLog('Clipboard indisponivel', 'warn');
      return;
    }
    try {
      const prompt = String(scene.visualPrompt || '').trim() || buildSceneVisualPrompt(scene.text, sceneIndex);
      await navigator.clipboard.writeText(prompt);
      pushLog(`Prompt da cena ${sceneIndex + 1} copiado`, 'info');
    } catch (err: any) {
      pushLog(`Erro ao copiar prompt: ${err?.message || err}`, 'error');
    }
  };

  const copyAllScenePrompts = async () => {
    if (!script?.scenes?.length) {
      pushLog('Nao ha cenas para copiar', 'warn');
      return;
    }
    if (!navigator.clipboard) {
      pushLog('Clipboard indisponivel', 'warn');
      return;
    }
    try {
      const compiled = buildAllScenePromptsText('pt');
      if (!compiled) {
        pushLog('Nao foi possivel montar o texto dos prompts', 'warn');
        return;
      }
      await navigator.clipboard.writeText(compiled);
      pushLog(`Prompts de ${script.scenes.length} cenas copiados com instrucoes`, 'info');
    } catch (err: any) {
      pushLog(`Erro ao copiar prompts: ${err?.message || err}`, 'error');
    }
  };

  const copyAllScenePromptsEn = async () => {
    if (!script?.scenes?.length) {
      pushLog('Nao ha cenas para copiar', 'warn');
      return;
    }
    if (!navigator.clipboard) {
      pushLog('Clipboard indisponivel', 'warn');
      return;
    }
    try {
      const compiled = buildAllScenePromptsText('en');
      if (!compiled) {
        pushLog('Nao foi possivel montar o texto dos prompts', 'warn');
        return;
      }
      await navigator.clipboard.writeText(compiled);
      pushLog(`Prompts EN de ${script.scenes.length} cenas copiados`, 'info');
    } catch (err: any) {
      pushLog(`Erro ao copiar prompts EN: ${err?.message || err}`, 'error');
    }
  };

  const copyTileAsBase64 = async (tile: GridTile) => {
    if (!navigator.clipboard) {
      pushLog('Clipboard indisponível', 'warn');
      return;
    }
    try {
      await navigator.clipboard.writeText(tile.dataUrl);
      pushLog('Tile copiado para área de transferência', 'info');
    } catch (err: any) {
      pushLog(`Erro ao copiar tile: ${err?.message || err}`, 'error');
    }
  };

  const applyTileToScene = (tile: GridTile) => {
    if (!gridSceneTarget) {
      pushLog('Selecione uma cena antes de aplicar o tile', 'warn');
      return;
    }
    updateScene(gridSceneTarget, { localImage: tile.dataUrl });
    pushLog(`Tile ${tile.row + 1}x${tile.col + 1} aplicado na cena`, 'info');
  };

  const handleImportScript = (file: File) => {
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const json = JSON.parse(String(reader.result || '{}'));
        if (!json?.title || !Array.isArray(json?.scenes)) throw new Error('Formato inválido');
        const scenesWithStatus = json.scenes.map((s: any, idx: number) => ({
          ...s,
          status: s.status ?? 'completed',
          visualPrompt: String(s.visualPrompt || '').trim() || buildSceneVisualPrompt(String(s.text || ''), idx),
          narrationVolume: Math.max(0, Math.min(2, Number(s.narrationVolume) || 1)),
          trimStartMs: Math.max(0, Math.min(5000, Math.floor(Number(s.trimStartMs) || 0))),
          trimEndMs: Math.max(0, Math.min(5000, Math.floor(Number(s.trimEndMs) || 0))),
          audioOffsetMs: Math.max(-3000, Math.min(3000, Math.floor(Number(s.audioOffsetMs) || 0))),
          localSfx: typeof s.localSfx === 'string' ? s.localSfx : undefined,
          sfxVolume: Math.max(0, Math.min(2, Number(s.sfxVolume) || 0.35)),
          transition: s.transition || settings.defaultTransition,
          filter: s.filter || settings.defaultFilter,
          flowTo: typeof s.flowTo === 'string' ? s.flowTo : null,
        }));
        setScript({ ...json, scenes: scenesWithStatus });
        setCurrentStep('assets');
        setImportError(null);
      } catch (err: any) {
        setImportError(err?.message || 'Erro ao importar roteiro');
      }
    };
    reader.readAsText(file, 'utf-8');
  };

  const downloadScript = () => {
    if (!script) return;
    const blob = new Blob([JSON.stringify(script, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${script.title || 'roteiro'}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const resetProject = () => {
    setTopic('');
    setScript(null);
    setVideoUrl(null);
    setTimelineEditMode('pre');
    setCurrentStep('topic');
    setStatusMessage('');
    setScenePositions({});
    localStorage.removeItem(SESSION_KEY);
    localStorage.removeItem(SESSION_PREV_KEY);
    if (hasIndexedDb()) {
      void idbRemoveKey(SESSION_KEY);
      void idbRemoveKey(SESSION_PREV_KEY);
    }
  };

  const generateFullVideo = async () => {
    if (!script) return;
    if (!apiBase) {
      setStatusMessage('API indisponível para render. Inicie o backend: "cd video_factory; .\\.venv\\Scripts\\Activate.ps1; uvicorn api:app --reload --port 8000".');
      pushLog('Backend nao encontrado. Inicie uvicorn em 127.0.0.1:8000 (ou 8001+) e recarregue.', 'error');
      return;
    }

    setIsGeneratingVideo(true);
    setCurrentStep('rendering');
    setRenderProgress(0);
    setRenderStage('Iniciando');
    setRenderRealPct(0);
    setRenderRealInfo('');
    setRenderStaleSeconds(0);
    setRenderElapsedSeconds(0);
    renderStartedAtRef.current = Date.now();
    setStatusMessage('Iniciando render com motor local...');

    let producedUrl: string | null = null;
    try {
      const hasLocalAssets = script.scenes.some(s => s.localImage || s.localSfx) || settings.localThemeRef || settings.localBackgroundMusic;

      if (!hasLocalAssets) {
        setStatusMessage('Adicione imagens ou áudio local antes de processar. O motor local precisa de assets locais para criar o vídeo.');
        pushLog('Render abortado: faltam assets locais.', 'warn');
        setCurrentStep('assets');
        return;
      }

      const payload = {
        scenes: script.scenes.map(s => ({
          id: String(s.id),
          text: s.text,
          visualPrompt: s.visualPrompt,
          localImage: s.localImage,
          narrationVolume: s.narrationVolume ?? 1,
          trimStartMs: s.trimStartMs ?? 0,
          trimEndMs: s.trimEndMs ?? 0,
          audioOffsetMs: s.audioOffsetMs ?? 0,
          localSfx: s.localSfx,
          sfxVolume: s.sfxVolume ?? 0.35,
          animationType: (s as any).animationType,
          transition: s.transition || settings.defaultTransition,
          filter: s.filter || settings.defaultFilter,
          flowTo: s.flowTo ? String(s.flowTo) : null,
        })),
        format,
        voice: edgeVoiceId,
        scriptTitle: script.title,
        backgroundMusic: settings.localBackgroundMusic
          || (settings.backgroundMusic && settings.backgroundMusic.startsWith('data:') ? settings.backgroundMusic : undefined),
        transitionStyle: settings.defaultTransition || 'mixed',
        colorFilter: settings.defaultFilter || 'none',
        musicVolume,
        narrationVolume,
        colorStrength,
        transitionDuration,
        imageScale,
        captionFontScale: captionScale,
        captionBgOpacity: captionBg,
        captionColor,
        captionHighlightColor: captionHighlight,
        captionYPct: captionY / 100,
        ffmpegFilters: ffmpegFilters || undefined,
        stabilize,
        aiEnhance,
        engine,
      };

      const startRes = await apiFetch('/api/render/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!startRes.ok) {
        const err = await startRes.json().catch(() => ({}));
        const msg = err?.detail ? JSON.stringify(err.detail) : `HTTP ${startRes.status}`;
        throw new Error(`Falha ao iniciar render: ${msg}`);
      }
      const { task_id } = await startRes.json();
      setRenderTaskId(task_id);
      setStatusMessage('Render em andamento...');
      pushLog(`Render iniciado (task ${task_id.slice(0, 6)}...)`, 'info');

      let done = false;
      while (!done) {
        const stRes = await apiFetch(`/api/render/status/${task_id}`);
        if (!stRes.ok) throw new Error(`Falha ao consultar status: HTTP ${stRes.status}`);
        const st = await stRes.json();
        const stageLabel = st.stage || '';
        const staleSeconds = Number(st.stale_seconds || 0);
        const stageProgressPct = Math.max(0, Math.min(100, Number(st.stage_progress || 0) * 100));
        const detail = (st.render_detail || null) as any;
        const backendPct = Math.max(0, Math.min(100, Number(st.progress || 0) * 100));
        const syntheticPct = stageLabel === 'finalizando'
          ? Math.min(99.9, Math.max(backendPct, 98.5 + Math.min(staleSeconds, 120) * 0.012))
          : backendPct;
        const fallbackMsg = stageLabel === 'finalizando'
          ? `Finalizando arquivo de vídeo. Render ativo (${Math.round(staleSeconds)}s sem novos frames). Não travou.`
          : staleSeconds >= 12
            ? `Render ativo... ${Math.round(staleSeconds)}s sem nova atualização.`
            : '';
        setRenderStage(stageLabel);
        setRenderStaleSeconds(staleSeconds);
        setRenderProgress(syntheticPct);
        setRenderRealPct(stageProgressPct);
        if (String(stageLabel).toLowerCase() === 'render') {
          const current = Number(detail?.current || 0);
          const total = Number(detail?.total || 0);
          const fps = Number(detail?.fps || 0);
          if (total > 0) {
            setRenderRealInfo(`Frames ${current}/${total}${fps > 0 ? ` • ${fps.toFixed(1)} fps` : ''}`);
          } else {
            setRenderRealInfo(`Montagem ${stageProgressPct.toFixed(1)}%`);
          }
        } else if (String(stageLabel).toLowerCase() === 'tts') {
          setRenderRealInfo(`Narrações ${stageProgressPct.toFixed(1)}%`);
        } else if (String(stageLabel).toLowerCase() === 'post' || String(stageLabel).toLowerCase() === 'finalizando') {
          setRenderRealInfo(`Finalização ${stageProgressPct.toFixed(1)}%`);
        } else {
          setRenderRealInfo('');
        }
        setStatusMessage(st.message || fallbackMsg);
        if (st.done) {
          done = true;
          if (st.error) {
            if (st.error_log_url) {
              const logUrl = st.error_log_url.startsWith('/') && apiBase ? `${apiBase}${st.error_log_url}` : st.error_log_url;
              pushLog(`Log do erro: ${logUrl}`, 'error');
            }
            throw new Error(st.error_log_url ? `${st.error} | log: ${st.error_log_url}` : st.error);
          }
          const rawUrl = st.web_url || st.output || null;
          const finalUrl = toPlayableVideoUrl(rawUrl);
          producedUrl = finalUrl;
          setVideoUrl(finalUrl);
          setRenderProgress(100);
          setRenderStage('Concluído');
          setRenderRealPct(100);
          setRenderRealInfo('Montagem concluída');
          setStatusMessage('Render concluído');
          pushLog(`Render concluído: ${finalUrl || rawUrl || st.output}`, 'info');
          setTimelineEditMode('post');
          setWorkspacePage('studio');
          setCurrentStep('done');
        } else {
          const pollDelayMs = staleSeconds >= 20 ? 3000 : 1500;
          await new Promise(r => setTimeout(r, pollDelayMs));
        }
      }
    } catch (error: any) {
      setCurrentStep('assets');
      handleAiError(error, 'Vídeo');
    } finally {
      setIsGeneratingVideo(false);
      setRenderStage(prev => (prev === 'Concluído' ? prev : ''));
      setRenderTaskId(null);
      if (!producedUrl) {
        setRenderRealPct(0);
        setRenderRealInfo('');
      }
      setRenderStaleSeconds(0);
      renderStartedAtRef.current = null;
      if (!producedUrl) {
        setStatusMessage('');
        setRenderProgress(0);
        setRenderElapsedSeconds(0);
      }
    }
  };

  const testRenderPipeline = async () => {
    if (isGeneratingVideo || isTestingRender) return;
    setIsTestingRender(true);
    setStatusMessage('Testando pipeline local com payload padrao...');
    try {
      const testRequest = {
        scenes: [
          { id: 'test-1', text: 'Cena de teste automática gerada no backend.', animationType: 'kenburns', transition: settings.defaultTransition, filter: settings.defaultFilter },
          { id: 'test-2', text: 'Segunda cena para validar legendas karaokê.', animationType: 'zoom_in', transition: settings.defaultTransition, filter: settings.defaultFilter },
        ],
        format,
        voice: edgeVoiceId,
        scriptTitle: 'Teste local',
        musicVolume,
        narrationVolume,
        colorStrength,
        transitionDuration,
        imageScale,
        captionFontScale: captionScale,
        captionBgOpacity: captionBg,
        captionColor,
        captionHighlightColor: captionHighlight,
        captionYPct: captionY / 100,
      };
      const response = await apiFetch('/api/render', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(testRequest),
      });
      if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        throw new Error(err?.detail || `HTTP ${response.status}`);
      }
      const data = await response.json();
      const rawUrl = data.web_url || data.output;
      const finalUrl = toPlayableVideoUrl(rawUrl);
      setVideoUrl(finalUrl);
      setRenderProgress(100);
      setRenderStage('Concluído');
      setRenderRealPct(100);
      setRenderRealInfo('Montagem concluída');
      setTimelineEditMode('post');
      setWorkspacePage('studio');
      setCurrentStep('done');
      pushLog(`Teste de render concluído: ${finalUrl}`, 'info');
      setStatusMessage(`Teste completo. Saída: ${finalUrl}`);
    } catch (error: any) {
      handleAiError(error, 'Teste de render');
      setStatusMessage('Falha no teste de render.');
    } finally {
      setIsTestingRender(false);
    }
  };

  const progress = ((stepOrder.indexOf(currentStep) + 1) / stepOrder.length) * 100;
  const renderProgressLabel = (renderStage || '').toLowerCase() === 'finalizando'
    ? renderProgress.toFixed(1)
    : renderProgress.toFixed(0);
  const formatRenderElapsed = (seconds: number) => {
    const safe = Math.max(0, Math.floor(Number(seconds) || 0));
    const mm = Math.floor(safe / 60);
    const ss = safe % 60;
    return `${String(mm).padStart(2, '0')}:${String(ss).padStart(2, '0')}`;
  };
  const renderStageLower = String(renderStage || '').toLowerCase();
  const renderIsWaiting = renderStageLower === 'finalizando' || renderStaleSeconds >= 10;
  const estimateSceneDuration = (scene: Scene) => {
    const words = String(scene.text || '').trim().split(/\s+/).filter(Boolean).length;
    const base = Math.max(1.2, (words / 2.7) + 0.6);
    const trimStart = Math.max(0, Number(scene.trimStartMs || 0)) / 1000;
    const trimEnd = Math.max(0, Number(scene.trimEndMs || 0)) / 1000;
    const offset = Number(scene.audioOffsetMs || 0) / 1000;
    const adjusted = base - trimStart - trimEnd + Math.max(0, offset);
    return Math.max(0.8, Math.min(22, adjusted));
  };

  const toPlayableVideoUrl = (rawUrl?: string | null): string | null => {
    if (!rawUrl) return null;
    const raw = String(rawUrl).trim();
    if (!raw) return null;
    if (/^https?:\/\//i.test(raw) || raw.startsWith('blob:') || raw.startsWith('data:')) return raw;
    if (raw.startsWith('/')) return apiBase ? `${apiBase}${raw}` : raw;

    const normalized = raw.replace(/\\/g, '/');
    const lower = normalized.toLowerCase();
    const marker = '/assets/';
    const idx = lower.lastIndexOf(marker);
    if (idx >= 0) {
      const webPath = normalized.slice(idx);
      return apiBase ? `${apiBase}${webPath}` : webPath;
    }
    const marker2 = 'assets/';
    const idx2 = lower.lastIndexOf(marker2);
    if (idx2 >= 0) {
      const webPath = `/assets/${normalized.slice(idx2 + marker2.length)}`;
      return apiBase ? `${apiBase}${webPath}` : webPath;
    }
    return raw;
  };
  const timelineSceneMeta = useMemo(() => {
    if (!script) return [];
    let cursor = 0;
    return script.scenes.map((scene, idx) => {
      const durationSec = estimateSceneDuration(scene);
      const startSec = cursor;
      cursor += durationSec;
      return {
        scene,
        idx,
        durationSec,
        startSec,
        widthPx: Math.max(220, Math.round(durationSec * 72)),
      };
    });
  }, [script]);
  const timelineTotalSeconds = useMemo(
    () => timelineSceneMeta.reduce((acc, item) => acc + item.durationSec, 0),
    [timelineSceneMeta],
  );
  const timelineDurationSafe = Math.max(0.1, timelineTotalSeconds);
  const timelinePixelsPerSecond = 86 * timelineZoom;
  const timelineContentWidth = Math.max(980, Math.round(timelineDurationSafe * timelinePixelsPerSecond) + 80);
  const timelineHeaderHeight = Math.max(42, Math.round(timelineTrackHeight * 0.62));
  const timelineTickStep = timelineTotalSeconds > 180 ? 10 : timelineTotalSeconds > 90 ? 5 : timelineTotalSeconds > 40 ? 2 : 1;
  const timelineTicks = useMemo(() => {
    const ticks: number[] = [];
    const cap = Math.ceil(timelineDurationSafe / timelineTickStep) * timelineTickStep;
    for (let t = 0; t <= cap; t += timelineTickStep) ticks.push(t);
    return ticks;
  }, [timelineDurationSafe, timelineTickStep]);
  const selectedTimelineScene = useMemo(
    () => timelineSceneMeta.find(item => item.scene.id === timelineSelectedSceneId) || timelineSceneMeta[0] || null,
    [timelineSceneMeta, timelineSelectedSceneId],
  );
  const activeTimelineScene = useMemo(() => {
    const found = timelineSceneMeta.find(
      item => timelinePlayheadSec >= item.startSec && timelinePlayheadSec < item.startSec + item.durationSec,
    );
    return found || timelineSceneMeta[timelineSceneMeta.length - 1] || null;
  }, [timelineSceneMeta, timelinePlayheadSec]);

  useEffect(() => {
    if (!script || !script.scenes.length) {
      setTimelineSelectedSceneId(null);
      return;
    }
    setTimelineSelectedSceneId(prev => script.scenes.some(s => s.id === prev) ? prev : script.scenes[0].id);
  }, [sceneIdKey]);

  useEffect(() => {
    setTimelinePlayheadSec(prev => Math.max(0, Math.min(prev, timelineDurationSafe)));
    if (timelineDurationSafe <= 0.11) setIsTimelinePlaying(false);
  }, [timelineDurationSafe]);

  useEffect(() => {
    if (!isTimelinePlaying) return;
    if (timelineDurationSafe <= 0.11) {
      setIsTimelinePlaying(false);
      return;
    }
    let rafId = 0;
    let lastTs = performance.now();
    const tick = (ts: number) => {
      const dt = Math.max(0, (ts - lastTs) / 1000);
      lastTs = ts;
      setTimelinePlayheadSec(prev => {
        const next = prev + dt;
        if (next >= timelineDurationSafe) {
          setIsTimelinePlaying(false);
          return timelineDurationSafe;
        }
        return next;
      });
      rafId = requestAnimationFrame(tick);
    };
    rafId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafId);
  }, [isTimelinePlaying, timelineDurationSafe]);

  const formatTimelineTime = (seconds: number) => {
    const safe = Math.max(0, Number(seconds) || 0);
    const min = Math.floor(safe / 60);
    const sec = Math.floor(safe % 60);
    const tenth = Math.floor((safe % 1) * 10);
    return `${String(min).padStart(2, '0')}:${String(sec).padStart(2, '0')}.${tenth}`;
  };

  const seekTimeline = (next: number) => {
    setIsTimelinePlaying(false);
    setTimelinePlayheadSec(Math.max(0, Math.min(next, timelineDurationSafe)));
  };

  useEffect(() => {
    if (!isGeneratingVideo) return;
    const tick = () => {
      const startedAt = renderStartedAtRef.current;
      if (!startedAt) return;
      setRenderElapsedSeconds((Date.now() - startedAt) / 1000);
    };
    tick();
    const intervalId = window.setInterval(tick, 1000);
    return () => window.clearInterval(intervalId);
  }, [isGeneratingVideo]);

  const moveSceneToTime = (sceneId: string, targetSec: number) => {
    if (!script || script.scenes.length <= 1) return;
    const movingScene = script.scenes.find(s => s.id === sceneId);
    if (!movingScene) return;
    const remainingScenes = script.scenes.filter(s => s.id !== sceneId);
    const durationById = new Map(timelineSceneMeta.map(item => [item.scene.id, item.durationSec]));

    const safeTarget = Math.max(0, Math.min(Number(targetSec) || 0, timelineDurationSafe));
    let insertIdx = 0;
    let cursor = 0;
    for (let i = 0; i < remainingScenes.length; i += 1) {
      const s = remainingScenes[i];
      const dur = durationById.get(s.id) ?? estimateSceneDuration(s);
      const midpoint = cursor + dur / 2;
      if (safeTarget >= midpoint) insertIdx = i + 1;
      cursor += dur;
    }

    const reordered = [
      ...remainingScenes.slice(0, insertIdx),
      movingScene,
      ...remainingScenes.slice(insertIdx),
    ];
    setScript({ ...script, scenes: reordered });
    setTimelineSelectedSceneId(sceneId);
  };

  const startTimelineClipDrag = (
    e: React.PointerEvent<HTMLButtonElement>,
    sceneId: string,
    startSec: number,
    durationSec: number,
  ) => {
    const canvasEl = e.currentTarget.parentElement as HTMLElement | null;
    if (!canvasEl) return;
    e.preventDefault();
    e.stopPropagation();
    setIsTimelinePlaying(false);
    setTimelineSelectedSceneId(sceneId);
    const maxSec = Math.max(0, timelineDurationSafe - durationSec);
    const normalizedStart = Math.max(0, Math.min(startSec, maxSec));
    timelineDragSecRef.current = normalizedStart;
    setTimelineDrag({
      sceneId,
      startSec: normalizedStart,
      durationSec,
      originClientX: e.clientX,
      minSec: 0,
      maxSec,
      currentSec: normalizedStart,
      pointerId: e.pointerId,
    });
  };

  useEffect(() => {
    if (!timelineDrag) return;
    const onMove = (ev: PointerEvent) => {
      if (ev.pointerId !== timelineDrag.pointerId) return;
      const deltaSec = (ev.clientX - timelineDrag.originClientX) / timelinePixelsPerSecond;
      const next = Math.max(timelineDrag.minSec, Math.min(timelineDrag.maxSec, timelineDrag.startSec + deltaSec));
      timelineDragSecRef.current = next;
      setTimelineDrag(prev => (prev ? { ...prev, currentSec: next } : prev));
    };
    const onUp = (ev: PointerEvent) => {
      if (ev.pointerId !== timelineDrag.pointerId) return;
      const finalSec = timelineDragSecRef.current;
      moveSceneToTime(timelineDrag.sceneId, finalSec);
      setTimelineDrag(null);
    };
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
    window.addEventListener('pointercancel', onUp);
    return () => {
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
      window.removeEventListener('pointercancel', onUp);
    };
  }, [timelineDrag, timelinePixelsPerSecond, moveSceneToTime]);

  return (
    <div className="min-h-screen bg-[#020617] selection:bg-blue-500/30">
      <div className="max-w-[1700px] mx-auto p-4 md:p-6 flex flex-col lg:flex-row gap-6">
        {/* SIDEBAR */}
        <aside className="w-full lg:w-[360px] xl:w-[380px] shrink-0 flex flex-col gap-4">
          <div className="glass p-4 rounded-2xl border-white/5 shadow-2xl">
            <div className="flex flex-col gap-3">
              <div>
                <h1 className="text-2xl font-extrabold font-display gradient-text tracking-tighter">VelozzVideo</h1>
                <p className="text-gray-500 text-[11px] font-bold uppercase tracking-[0.18em]">Fluxo Local + API</p>
              </div>
              <div className="grid grid-cols-2 gap-2">
                <button
                  onClick={() => setWorkspacePage('studio')}
                  className={`text-[10px] font-black uppercase px-3 py-2 rounded-xl border transition-all ${
                    workspacePage === 'studio'
                      ? 'border-blue-400 bg-blue-500/20 text-white'
                      : 'border-gray-700 text-gray-300 hover:text-white hover:border-blue-400'
                  }`}
                >
                  Studio
                </button>
                <button
                  onClick={() => {
                    setTimelineEditMode(videoUrl ? 'post' : 'pre');
                    setWorkspacePage('timeline');
                  }}
                  className={`text-[10px] font-black uppercase px-3 py-2 rounded-xl border transition-all ${
                    workspacePage === 'timeline'
                      ? 'border-cyan-400 bg-cyan-500/20 text-white'
                      : 'border-gray-700 text-gray-300 hover:text-white hover:border-cyan-400'
                  }`}
                >
                  Timeline
                </button>
              </div>
              <div className="grid grid-cols-3 gap-2">
                <button onClick={restoreSession} className="text-[9px] font-bold uppercase px-2 py-2 rounded-xl border border-green-500/40 text-green-300 hover:text-white hover:border-green-400 transition-all leading-tight text-center">Recuperar sessao</button>
                <button onClick={restorePreviousSession} className="text-[9px] font-bold uppercase px-2 py-2 rounded-xl border border-amber-500/40 text-amber-300 hover:text-white hover:border-amber-400 transition-all leading-tight text-center">Sessao anterior</button>
                <button onClick={resetProject} className="text-[9px] font-bold uppercase px-2 py-2 rounded-xl border border-red-500/40 text-red-300 hover:text-white hover:border-red-400 transition-all leading-tight text-center">Resetar</button>
              </div>
            </div>
            <div className="mt-3 grid grid-cols-3 gap-2 text-[10px] text-gray-400"> 
              <div className="px-2 py-1 rounded-lg bg-black/40 border border-gray-800">Etapa: <span className="text-white">{currentStep}</span></div>
              <div className="px-2 py-1 rounded-lg bg-black/40 border border-gray-800">Form.: {format}</div>
              <div className="px-2 py-1 rounded-lg bg-black/40 border border-gray-800">Duracao: {videoLength === '1m' ? '1m' : '5m'}</div>
            </div>
            <div className="mt-3 h-2 w-full bg-gray-900 rounded-full overflow-hidden">
              <div className="h-full bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500" style={{ width: `${progress}%` }}></div>
            </div>
          </div>

          <div className="glass p-5 rounded-3xl border-blue-500/15 flex flex-col gap-3 shadow-2xl">
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-black uppercase tracking-[0.2em] text-blue-400">Status de Producao</span>
              <span className="text-[10px] font-mono text-gray-400">{renderProgressLabel}%</span>
            </div>
            <div className="h-2 w-full bg-gray-900 rounded-full overflow-hidden">
              <div className="h-full bg-gradient-to-r from-blue-500 via-cyan-400 to-emerald-400 transition-all duration-300" style={{ width: `${Math.max(0, Math.min(100, renderProgress))}%` }}></div>
            </div>
            <div className="grid grid-cols-2 gap-2 text-[10px] text-gray-300">
              <div className="bg-black/30 border border-gray-800 rounded-xl px-2 py-1.5">
                <div className="text-gray-500 uppercase tracking-widest text-[9px]">Etapa</div>
                <div className="text-white font-mono mt-0.5">{renderStage || currentStep}</div>
              </div>
              <div className="bg-black/30 border border-gray-800 rounded-xl px-2 py-1.5">
                <div className="text-gray-500 uppercase tracking-widest text-[9px]">Task</div>
                <div className="text-white font-mono mt-0.5">{renderTaskId ? renderTaskId.slice(0, 8) : "-"}</div>
              </div>
              <div className="bg-black/30 border border-gray-800 rounded-xl px-2 py-1.5">
                <div className="text-gray-500 uppercase tracking-widest text-[9px]">Cards</div>
                <div className="text-white font-mono mt-0.5">{script?.scenes.length || 0}</div>
              </div>
              <div className="bg-black/30 border border-gray-800 rounded-xl px-2 py-1.5">
                <div className="text-gray-500 uppercase tracking-widest text-[9px]">Assets</div>
                <div className="text-emerald-300 font-mono mt-0.5">{script ? script.scenes.filter(s => s.localImage).length : 0}</div>
              </div>
            </div>
            {statusMessage && (
              <div className="text-[10px] text-gray-300 bg-black/30 border border-gray-800 rounded-xl px-3 py-2">
                {statusMessage}
              </div>
            )}
          </div>

          {script && workspacePage === 'studio' && (
            <div className="glass p-5 rounded-3xl border-blue-500/10 flex flex-col gap-4 shadow-2xl">
              <h3 className="font-black text-xs text-blue-500 uppercase tracking-[0.3em]">Painel de Render</h3>

              <div className="pt-1 text-[11px] space-y-3">
                <div className="flex items-center justify-between gap-2">
                  <span className="text-gray-500 font-mono">Roteiro</span>
                  <div className="flex gap-2">
                    <label className="text-[9px] font-bold px-3 py-2 bg-gray-900 rounded-xl border border-gray-800 cursor-pointer hover:border-blue-500/50 transition-all">
                      Importar
                      <input type="file" accept="application/json" className="hidden" onChange={(e) => e.target.files?.[0] && handleImportScript(e.target.files[0])} />
                    </label>
                    <button onClick={downloadScript} disabled={!script} className="text-[9px] font-bold px-3 py-2 bg-gray-900 rounded-xl border border-gray-800 text-gray-400 hover:text-white hover:border-blue-500/50 disabled:opacity-40 transition-all">Exportar</button>
                  </div>
                </div>
                {importError && <p className="text-xs text-red-400">{importError}</p>}
              </div>

              <div className="pt-3 border-t border-gray-800 text-[11px] space-y-2">
                <div className="flex items-center justify-between gap-2">
                  <span className="text-gray-500 font-mono">Limpar cache</span>
                  <div className="flex items-center gap-2">
                    <input
                      type="number"
                      min={1}
                      className="w-16 bg-black/40 border border-gray-800 rounded-lg px-2 py-1 text-[10px] text-white outline-none focus:border-blue-500/40"
                      value={cleanupAgeDays}
                      onChange={(e) => setCleanupAgeDays(Math.max(1, Number(e.target.value) || 7))}
                      title="Dias maximos para manter assets"
                    />
                    <button
                      onClick={cleanupCache}
                      disabled={isCleaningCache}
                      className="text-[9px] font-bold px-3 py-2 bg-gray-900 rounded-xl border border-gray-800 text-gray-400 hover:text-white hover:border-blue-500/50 disabled:opacity-50 transition-all"
                    >
                      {isCleaningCache ? 'Limpando...' : 'Executar'}
                    </button>
                  </div>
                </div>
                {cleanupResult && <p className="text-[10px] text-green-300">{cleanupResult}</p>}
              </div>

              <div className="pt-3 border-t border-gray-800 text-[11px] space-y-3">
                <div className="flex items-start justify-between gap-3">
                  <span className="text-gray-500 font-mono">Volumes</span>
                  <div className="flex-1 grid grid-cols-2 gap-3">
                    <label className="flex flex-col gap-1 text-[10px] text-gray-400">
                      Musica {Math.round(musicVolume * 100)}%
                      <input type="range" min={0} max={1} step={0.05} value={musicVolume} onChange={(e)=>setMusicVolume(Number(e.target.value))} />
                    </label>
                    <label className="flex flex-col gap-1 text-[10px] text-gray-400">
                      Narracao {Math.round(narrationVolume * 100)}%
                      <input type="range" min={0.3} max={1.5} step={0.05} value={narrationVolume} onChange={(e)=>setNarrationVolume(Number(e.target.value))} />
                    </label>
                  </div>
                </div>

                <div className="flex items-start justify-between gap-3">
                  <span className="text-gray-500 font-mono">Legendas</span>
                  <div className="flex-1 space-y-2">
                    <div className="grid grid-cols-2 gap-2 text-[10px] text-gray-400">
                      <label className="flex flex-col gap-1">
                        Tamanho {captionScale.toFixed(2)}x
                        <input type="range" min={0.7} max={1.4} step={0.05} value={captionScale} onChange={(e)=>setCaptionScale(Number(e.target.value))} />
                      </label>
                      <label className="flex flex-col gap-1">
                        Opacidade BG {Math.round(captionBg*100)}%
                        <input type="range" min={0} max={1} step={0.05} value={captionBg} onChange={(e)=>setCaptionBg(Number(e.target.value))} />
                      </label>
                      <label className="flex flex-col gap-1">
                        Posicao Y {captionY}%
                        <input type="range" min={60} max={90} step={1} value={captionY} onChange={(e)=>setCaptionY(Number(e.target.value))} />
                      </label>
                    </div>
                    <div className="flex items-center gap-3 text-[10px] text-gray-300">
                      <span>Cor</span>
                      <input type="color" value={captionColor} onChange={(e)=>setCaptionColor(e.target.value)} />
                      <span>Destaque</span>
                      <input type="color" value={captionHighlight} onChange={(e)=>setCaptionHighlight(e.target.value)} />
                    </div>
                    <div className="mt-1 p-3 rounded-xl border border-gray-800 bg-black/40 text-center text-[11px]">
                      <span style={{backgroundColor:`rgba(0,0,0,${captionBg})`, padding:'6px 10px', borderRadius:'10px', display:'inline-block', color: captionColor}}>
                        LEGENDAS PREVIA <span style={{color: captionHighlight}}>DESTAQUE</span>
                      </span>
                    </div>
                  </div>
                </div>

                <div className="flex items-start justify-between gap-3">
                  <span className="text-gray-500 font-mono">Efeitos</span>
                  <div className="flex-1 grid grid-cols-2 gap-3 text-[10px] text-gray-400">
                    <label className="flex flex-col gap-1">
                      Forca de cor {colorStrength.toFixed(2)}
                      <input type="range" min={0} max={1.2} step={0.05} value={colorStrength} onChange={(e)=>setColorStrength(Number(e.target.value))} />
                    </label>
                    <label className="flex flex-col gap-1">
                      Duracao de transicao {transitionDuration.toFixed(2)}s
                      <input type="range" min={0.2} max={1.2} step={0.05} value={transitionDuration} onChange={(e)=>setTransitionDuration(Number(e.target.value))} />
                    </label>
                    <label className="flex flex-col gap-1 col-span-2">
                      Escala da imagem {imageScale.toFixed(2)}x
                      <input type="range" min={0.8} max={1.2} step={0.02} value={imageScale} onChange={(e)=>setImageScale(Number(e.target.value))} />
                    </label>
                  </div>
                </div>
              </div>

              <div className="pt-4 border-t border-gray-800">
                <button
                  onClick={generateFullVideo}
                  disabled={isGeneratingVideo}
                  className="w-full bg-blue-600 hover:bg-blue-500 py-4 rounded-[1.4rem] font-black text-xs uppercase tracking-[0.22em] transition-all active:scale-[0.98] shadow-xl shadow-blue-900/30 disabled:opacity-50"
                >
                  {isGeneratingVideo ? <Loader /> : "Processar Projeto"}
                </button>
                <button
                  onClick={testRenderPipeline}
                  disabled={isGeneratingVideo || isTestingRender}
                  className="w-full mt-3 border border-blue-500/40 text-blue-200 hover:text-white hover:border-blue-300 px-4 py-3 rounded-[1.4rem] font-black text-xs uppercase tracking-[0.22em] transition-all disabled:opacity-50"
                >
                  {isTestingRender ? <Loader /> : "Forcar teste (ver logs)"}
                </button>
              </div>
            </div>
          )}

          {/* Roteiro Local */}
          <div className="glass p-5 rounded-3xl flex flex-col gap-3 border-white/5 relative overflow-hidden">
            <div className="flex items-center justify-between">
              <p className="text-[11px] font-black uppercase text-blue-400">Roteiro Local (Ollama)</p>
              <button onClick={() => toggleSection('ollama')} className="text-[11px] text-gray-400 hover:text-white">{openSections.ollama ? '−' : '+'}</button>
            </div>
            {openSections.ollama && (
              <>
                <label className="text-[10px] text-gray-500 uppercase font-bold">Host</label>
                <input value={ollamaHost} onChange={(e) => setOllamaHost(e.target.value)} className="w-full bg-black/40 border border-gray-800 rounded-xl px-3 py-2 text-xs text-white focus:border-blue-500/40 outline-none" />
                <div className="flex items-center gap-2">
                  <div className="flex-1">
                    <label className="text-[10px] text-gray-500 uppercase font-bold">Modelo</label>
                    <input value={ollamaModel} onChange={(e) => setOllamaModel(e.target.value)} className="w-full bg-black/40 border border-gray-800 rounded-xl px-3 py-2 text-xs text-white focus:border-blue-500/40 outline-none" />
                  </div>
                  <button onClick={loadOllamaModels} disabled={isLoadingModels} className="text-[10px] font-bold uppercase px-3 py-2 mt-5 rounded-xl border border-blue-500/40 text-blue-200 hover:text-white hover:border-blue-400 transition-all disabled:opacity-50">
                    {isLoadingModels ? '...' : 'Listar'}
                  </button>
                </div>
                {ollamaError && <p className="text-[10px] text-red-400">{ollamaError}</p>}
                {!!ollamaModels.length && (
                  <ul className="space-y-1 text-[10px] text-gray-300 bg-black/30 rounded-xl border border-gray-800 p-2">
                    {ollamaModels.map(m => (
                      <li key={m}>
                        <button
                          onClick={() => setOllamaModel(m)}
                          className={`w-full text-left px-3 py-2 rounded-lg border ${
                            ollamaModel === m
                              ? 'border-blue-500 bg-blue-500/20 text-white'
                              : 'border-gray-700 bg-black/40 hover:border-blue-500/40 hover:text-white'
                          }`}
                        >
                          {m}
                        </button>
                      </li>
                    ))}
                  </ul>
                )}
                <p className="text-[11px] text-gray-500">Máx {maxScenes} cenas (vídeo {videoLength === '1m' ? '~1 min' : '~5 min'}). Sem chave de API.</p>
              </>
            )}
          </div>

          {/* Tema Visual */}
          <div className="glass p-5 rounded-3xl flex flex-col gap-4 border-white/5 relative overflow-hidden group">
            <div className="flex justify-between items-center">
              <label className="text-[10px] font-black text-blue-500 uppercase tracking-widest">Tema Visual</label>
              <button onClick={() => toggleSection('theme')} className="text-[11px] text-gray-400 hover:text-white">{openSections.theme ? '−' : '+'}</button>
            </div>
            {openSections.theme && (
              <>
                <select 
                  value={settings.theme} 
                  onChange={(e) => setSettings({...settings, theme: e.target.value})}
                  className="bg-black/40 text-xs p-3 rounded-xl border border-gray-800 text-gray-200 outline-none focus:border-blue-500/50 transition-all"
                >
                  <option value="Cinematographic">Cinematográfico</option>
                  <option value="Cyberpunk Noir">Cyberpunk Noir</option>
                  <option value="Ultra-Realistic Professional">Profissional Realista</option>
                  <option value="Anime Aesthetic">Anime Aesthetic</option>
                  <option value="Vibrant 3D Render">Vibrant 3D Render</option>
                </select>
                <div className="flex items-center justify-between gap-3">
                  <span className="text-[10px] text-gray-400 uppercase tracking-widest">Referência local</span>
                  <label className="cursor-pointer inline-flex items-center gap-2 px-3 py-1 bg-blue-500/10 border border-blue-500/30 rounded-xl text-[9px] font-bold text-blue-200 hover:border-blue-300 hover:text-white transition-all">
                    Upload
                    <input type="file" className="hidden" accept="image/*" onChange={(e) => e.target.files?.[0] && handleFileUpload('global_theme', '', e.target.files[0])} />
                  </label>
                </div>
                {settings.localThemeRef && (
                  <div className="h-12 w-full rounded-xl overflow-hidden border border-blue-500/30">
                    <img src={settings.localThemeRef} className="w-full h-full object-cover grayscale group-hover:grayscale-0 transition-all" />
                  </div>
                )}
              </>
            )}
          </div>

          {/* Efeitos de Vídeo (MoviePy) */}
          <div className="glass p-5 rounded-3xl flex flex-col gap-3 border-blue-500/10 relative overflow-hidden">
            <div className="flex items-center justify-between">
              <p className="text-[11px] font-black uppercase text-cyan-300">Efeitos de Vídeo (MoviePy)</p>
              <button onClick={() => toggleSection('effects')} className="text-[11px] text-gray-400 hover:text-white">{openSections.effects ? '−' : '+'}</button>
            </div>
            {openSections.effects && (
              <>
                <label className="text-[10px] text-gray-500 uppercase font-bold">Transicao padrao</label>
                <select
                  value={settings.defaultTransition || 'fade'}
                  onChange={(e) => setSettings({ ...settings, defaultTransition: e.target.value })}
                  className="bg-black/40 text-xs p-3 rounded-xl border border-gray-800 text-gray-200 outline-none focus:border-blue-500/50 transition-all"
                >
                  {transitionOptions.map(opt => (
                    <option key={opt.id} value={opt.id}>{opt.label}</option>
                  ))}
                </select>

                <label className="text-[10px] text-gray-500 uppercase font-bold">Filtro global</label>
                <select
                  value={settings.defaultFilter || 'none'}
                  onChange={(e) => setSettings({ ...settings, defaultFilter: e.target.value })}
                  className="bg-black/40 text-xs p-3 rounded-xl border border-gray-800 text-gray-200 outline-none focus:border-blue-500/50 transition-all"
                >
                  {filterOptions.map(opt => (
                    <option key={opt.id} value={opt.id}>{opt.label}</option>
                  ))}
                </select>

                <div className="flex gap-2">
                  <button
                    onClick={applyEffectsToAllScenes}
                    className="flex-1 text-[10px] font-black uppercase px-3 py-2 rounded-xl bg-blue-600/80 hover:bg-blue-500 text-white tracking-[0.18em] transition-all"
                  >
                    Aplicar nos cards
                  </button>
                  <div className="flex-1 text-[10px] text-gray-400 bg-black/30 border border-gray-800 rounded-xl p-2 leading-relaxed">
                    Inclui fades, crossfade, slides e filtros cinematográficos, bw, vhs, matte, vibrant.
                  </div>
                </div>

                <div className="mt-3 p-3 rounded-2xl bg-black/30 border border-cyan-500/20">
                  <div className="flex items-center justify-between mb-2">
                    <p className="text-[10px] uppercase font-black text-cyan-300">Lab de Efeitos (Prévia rápida)</p>
                    <span className="text-[10px] text-gray-500">MoviePy / FFmpeg</span>
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    <div className="flex flex-col gap-1">
                      <label className="text-[10px] text-gray-500 uppercase font-bold">Transicao</label>
                      <select value={previewTransition} onChange={e => setPreviewTransition(e.target.value)} className="bg-black/40 text-xs p-2.5 rounded-xl border border-gray-800 text-gray-200 outline-none focus:border-cyan-400/70 transition-all">
                        {transitionOptions.map(opt => <option key={opt.id} value={opt.id}>{opt.label}</option>)}
                      </select>
                    </div>
                    <div className="flex flex-col gap-1">
                      <label className="text-[10px] text-gray-500 uppercase font-bold">Animacao</label>
                      <select value={previewAnimation} onChange={e => setPreviewAnimation(e.target.value)} className="bg-black/40 text-xs p-2.5 rounded-xl border border-gray-800 text-gray-200 outline-none focus:border-cyan-400/70 transition-all">
                        {animationOptions.map(opt => <option key={opt.id} value={opt.id}>{opt.label}</option>)}
                      </select>
                    </div>
                    <div className="flex flex-col gap-1">
                      <label className="text-[10px] text-gray-500 uppercase font-bold">Filtro</label>
                      <select value={previewFilter} onChange={e => setPreviewFilter(e.target.value)} className="bg-black/40 text-xs p-2.5 rounded-xl border border-gray-800 text-gray-200 outline-none focus:border-cyan-400/70 transition-all">
                        {filterOptions.map(opt => <option key={opt.id} value={opt.id}>{opt.label}</option>)}
                      </select>
                    </div>
                    <div className="flex flex-col gap-1">
                      <label className="text-[10px] text-gray-500 uppercase font-bold">Força do filtro ({previewColorStrength.toFixed(2)})</label>
                      <input type="range" min={0} max={1.5} step={0.05} value={previewColorStrength} onChange={e => setPreviewColorStrength(parseFloat(e.target.value))} className="accent-cyan-400" />
                    </div>
                  </div>
                  <div className="mt-2 flex flex-col gap-1">
                    <label className="text-[10px] text-gray-500 uppercase font-bold">FFmpeg filtergraph (opcional)</label>
                    <textarea value={previewFilters} onChange={e => setPreviewFilters(e.target.value)} rows={2} className="w-full bg-black/40 border border-gray-800 rounded-xl px-3 py-2 text-[11px] text-gray-100 outline-none focus:border-cyan-400/70" placeholder="ex: unsharp=luma_msize_x=5:luma_amount=1.2,eq=saturation=1.2" />
                  </div>
                  <div className="flex items-center gap-2 mt-3">
                    <button onClick={previewVideoEffects} disabled={isPreviewRendering} className="px-3 py-2 rounded-xl bg-cyan-600/80 hover:bg-cyan-500 text-white text-[10px] font-black uppercase tracking-[0.16em] transition-all disabled:opacity-50">
                      {isPreviewRendering ? 'Gerando...' : 'Gerar prévia'}
                    </button>
                    <span className="text-[10px] text-gray-400">MP4 de 3-4s para validar transicao + filtro + animacao + FFmpeg.</span>
                  </div>
                  {previewUrl && (
                    <div className="mt-3 rounded-xl overflow-hidden border border-cyan-500/30 bg-black/40 p-2">
                      <video src={previewUrl} className="w-full rounded-lg" controls muted loop playsInline />
                    </div>
                  )}
                </div>

              </>
            )}
          </div>

          {/* Edicao avancada (FFmpeg / Estabilizacao) */}
          <div className="glass p-5 rounded-3xl flex flex-col gap-4 border-emerald-500/10 relative overflow-hidden">
            <div className="flex items-center justify-between">
              <p className="text-[11px] font-black uppercase text-emerald-300">Edicao avancada (pos)</p>
              <button onClick={() => toggleSection('advanced')} className="text-[11px] text-gray-400 hover:text-white">{openSections.advanced ? '−' : '+'}</button>
            </div>
            {openSections.advanced && (
              <>
                <label className="text-[10px] text-gray-500 uppercase font-bold">FFmpeg filtergraph (aplica no MP4 final)</label>
                <textarea value={ffmpegFilters} onChange={e => setFfmpegFilters(e.target.value)} rows={3} className="w-full bg-black/40 border border-gray-800 rounded-xl px-3 py-2 text-[11px] text-gray-100 outline-none focus:border-emerald-400/70" placeholder="ex: unsharp=luma_msize_x=5:luma_amount=1.2,eq=saturation=1.15,curves=psfile=film_lut.acv" />
                <div className="grid grid-cols-2 gap-3">
                  <div className="flex items-center gap-2">
                    <input id="stabilize" type="checkbox" checked={stabilize} onChange={e => setStabilize(e.target.checked)} className="accent-emerald-400" />
                    <label htmlFor="stabilize" className="text-[11px] text-gray-200">Estabilizar (VidGear, opcional)</label>
                  </div>
                  <div className="flex items-center gap-2">
                    <input id="aiEnhance" type="checkbox" checked={aiEnhance} onChange={e => setAiEnhance(e.target.checked)} className="accent-emerald-400" />
                    <label htmlFor="aiEnhance" className="text-[11px] text-gray-200">AI Enhance (placeholder)</label>
                  </div>
                  <div className="flex flex-col gap-1">
                    <label className="text-[10px] text-gray-500 uppercase font-bold">Motor</label>
                    <select value={engine} onChange={e => setEngine(e.target.value)} className="bg-black/40 text-xs p-2.5 rounded-xl border border-gray-800 text-gray-200 outline-none focus:border-emerald-400/70 transition-all">
                      <option value="moviepy">MoviePy (padrao)</option>
                      <option value="movielite">MovieLite (experimental)</option>
                    </select>
                  </div>
                  <div className="text-[10px] text-gray-400 bg-black/30 border border-gray-800 rounded-xl p-2 leading-relaxed">
                    Os filtros FFmpeg serao aplicados apos a montagem. Se ffmpeg nao estiver instalado, o render segue sem erro.
                  </div>
                </div>
              </>
            )}
          </div>

          {/* Áudio local */}
          <div className="glass p-5 rounded-3xl flex flex-col gap-4 border-purple-500/15 relative group overflow-hidden">
            <div className="absolute inset-0 pointer-events-none opacity-10 blur-3xl bg-[radial-gradient(circle_at_30%_20%,#8b5cf6,transparent_40%),radial-gradient(circle_at_80%_0%,#3b82f6,transparent_35%)]"></div>
            <div className="flex justify-between items-center relative z-10">
              <label className="text-[10px] font-black text-purple-300 uppercase tracking-[0.18em]">Áudio de Fundo (Local)</label>
              <div className="flex items-center gap-2">
                <label className="cursor-pointer inline-flex items-center gap-2 px-3 py-1 bg-purple-600/20 border border-purple-500/40 rounded-xl text-[9px] font-bold text-purple-200 hover:border-purple-300 hover:text-white transition-all">
                 <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 4v16m8-8H4"/></svg>
                 Selecionar
                 <input type="file" className="hidden" accept="audio/*" onChange={(e) => e.target.files?.[0] && handleFileUpload('global_music', '', e.target.files[0])} />
                </label>
                <button onClick={() => toggleSection('audio')} className="text-[11px] text-gray-300 hover:text-white relative z-10">{openSections.audio ? '−' : '+'}</button>
              </div>
            </div>
            {openSections.audio && (
              <>
                <div className="text-[11px] text-gray-400 relative z-10 leading-relaxed">
                  Arraste ou clique para definir a trilha. Fontes online estao desativadas para priorizar conteudo local e seguro.
                </div>
                {settings.localBackgroundMusic ? (
                  <div className="flex items-center gap-2 px-3 py-2 bg-purple-500/15 rounded-xl border border-purple-500/30 relative z-10">
                    <svg className="w-4 h-4 text-purple-300" fill="currentColor" viewBox="0 0 24 24"><path d="M12 3v10.55c-.59-.34-1.27-.55-2-.55-2.21 0-4 1.79-4 4s1.79 4 4 4 4-1.79 4-4V7h4V3h-6z"/></svg>
                    <span className="text-[10px] font-mono text-purple-100 truncate">Áudio local carregado</span>
                  </div>
                ) : (
                  <div className="text-[10px] text-yellow-300 font-semibold relative z-10">Nenhum áudio local selecionado.</div>
                )}
              </>
            )}
          </div>

          {/* Voz Edge TTS */}
          <div className="glass p-5 rounded-3xl border-white/5 flex flex-col gap-3">
            <div className="flex items-center justify-between">
              <p className="text-[11px] font-black uppercase text-blue-300">Voz (Edge TTS)</p>
              <div className="flex items-center gap-2">
                <span className="text-[10px] text-gray-500">pt-*</span>
                <button onClick={() => toggleSection('voice')} className="text-[11px] text-gray-400 hover:text-white">{openSections.voice ? '−' : '+'}</button>
              </div>
            </div>
            {openSections.voice && (
              <>
                <div className="flex flex-col gap-2 text-[10px] text-gray-400 mb-1">
                  <div className="flex items-center justify-between">
                      <span>
                        Voz selecionada:{' '}
                      {selectedEdgeVoice?.label || edgeVoiceId}
                      {selectedEdgeVoice?.locale ? ` (${selectedEdgeVoice.locale})` : ''}
                      {selectedEdgeVoice?.gender && selectedEdgeVoice.gender !== 'Unknown' ? ` - ${selectedEdgeVoice.gender}` : ''}
                      </span>
                    <button onClick={previewVoice} disabled={isPreviewingVoice} className="px-2 py-1 rounded-lg border border-blue-500/40 text-blue-200 hover:text-white hover:border-blue-300 transition-all disabled:opacity-50">
                      {isPreviewingVoice ? '...' : 'Ouvir prévia'}
                    </button>
                  </div>
                  <input
                    value={previewText}
                    onChange={(e) => setPreviewText(e.target.value)}
                    className="w-full bg-black/40 border border-gray-800 rounded-lg px-3 py-2 text-[10px] text-white focus:border-blue-500/40 outline-none"
                    placeholder="Digite o texto para prévia"
                  />
                </div>
                <ul className="space-y-1 text-[10px] text-gray-300 bg-black/30 rounded-xl border border-gray-800 p-2 max-h-40 overflow-auto custom-scrollbar">
                  {edgeVoiceOptions.map(v => (
                    <li key={v.id}>
                      <button
                        onClick={() => {
                          setEdgeVoiceId(v.id);
                          localStorage.setItem('edge_voice', v.id);
                        }}
                        className={`w-full text-left px-3 py-2 rounded-lg border transition-all ${
                          edgeVoiceId === v.id
                            ? 'border-blue-500 bg-blue-500/20 text-white'
                            : 'border-gray-700 bg-black/40 text-gray-300 hover:border-blue-500/40 hover:text-white'
                        }`}
                      >
                        {v.label} {v.locale ? `(${v.locale})` : ''} {v.gender && v.gender !== 'Unknown' ? `- ${v.gender}` : ''}
                      </button>
                    </li>
                  ))}
                </ul>
                <p className="text-[10px] text-gray-500">PT-BR: Antonio (M), Francisca/Thalita (F). PT-PT: Duarte (M), Raquel (F).</p>
              </>
            )}
          </div>

          <div className="glass p-5 rounded-3xl border-blue-500/15 flex flex-col gap-3 max-h-64 overflow-hidden">
            <div className="flex justify-between items-center">
              <span className="text-[10px] font-black uppercase tracking-[0.2em] text-blue-400">Painel Backend</span>
              <div className="flex gap-2 items-center">
                <button onClick={loadOllamaModels} disabled={isLoadingModels} className="text-[10px] font-bold uppercase px-3 py-1 rounded-lg border border-blue-500/40 text-blue-200 hover:border-blue-300 hover:text-white transition-all disabled:opacity-50">Ping</button>
                <button onClick={() => toggleSection('backend')} className="text-[11px] text-gray-400 hover:text-white">{openSections.backend ? '−' : '+'}</button>
              </div>
            </div>
            {openSections.backend && (
              <>
                  <div className="grid grid-cols-2 gap-2 text-[10px] text-gray-300">
                    <div className="flex items-center gap-2">
                      <span className="w-2 h-2 rounded-full" style={{ background: ollamaError ? '#f87171' : '#34d399' }}></span>
                      Ollama {ollamaError ? 'Offline' : 'Online'}
                    </div>
                    <div>Modelos: {ollamaModels.length || '—'}</div>
                    <div>Áudio: {settings.localBackgroundMusic ? 'Local carregado' : 'Pendente'}</div>
                    <div>Voice Preview (Edge): {isPreviewingVoice ? 'Local gerando...' : 'Disponível'}</div>
                  </div>
                <div className="border border-blue-500/10 rounded-xl bg-black/30 h-28 overflow-auto custom-scrollbar text-[11px] text-gray-200 font-mono px-3 py-2 space-y-1">
                  {logFeed.length === 0 && <div className="text-gray-500 text-[10px]">Sem logs recentes.</div>}
                  {logFeed.map(log => (
                    <div key={log.id} className="flex gap-2 items-center">
                      <span className="text-gray-500">{log.time}</span>
                      <span className={log.level === 'error' ? 'text-red-400' : log.level === 'warn' ? 'text-yellow-300' : 'text-blue-300'}>
                        [{log.level.toUpperCase()}]
                      </span>
                      <span className="text-gray-100">{log.msg}</span>
                    </div>
                  ))}
                </div>
              </>
            )}
          </div>
        </aside>

        {/* MAIN WORKSPACE */}
        <main className="min-w-0 flex-1 flex flex-col gap-6">
          {workspacePage === 'timeline' ? (
            <>
              <div className="glass p-5 rounded-2xl border-cyan-500/20 shadow-2xl flex flex-col gap-3">
                <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-3">
                  <div>
                    <h2 className="text-sm font-black uppercase tracking-[0.2em] text-cyan-300">Timeline de Edicao</h2>
                    <p className="text-[11px] text-gray-400">
                      {timelineEditMode === 'post'
                        ? 'Pos-producao ativa: ajuste manual e renderize novamente.'
                        : 'Pre-producao ativa: ajuste narracao, imagem e efeito sonoro por cena.'}
                    </p>
                  </div>
                  <div className="flex gap-2">
                    <div className="flex rounded-xl border border-cyan-500/30 overflow-hidden">
                      <button
                        onClick={() => setTimelineEditMode('pre')}
                        className={`px-3 py-2 text-[10px] font-black uppercase tracking-widest transition-all ${
                          timelineEditMode === 'pre' ? 'bg-cyan-500/30 text-white' : 'bg-black/50 text-gray-300 hover:text-white'
                        }`}
                      >
                        Pre
                      </button>
                      <button
                        onClick={() => setTimelineEditMode('post')}
                        className={`px-3 py-2 text-[10px] font-black uppercase tracking-widest transition-all ${
                          timelineEditMode === 'post' ? 'bg-cyan-500/30 text-white' : 'bg-black/50 text-gray-300 hover:text-white'
                        }`}
                      >
                        Pos
                      </button>
                    </div>
                    <button
                      onClick={() => setShowTimelineInspector(prev => !prev)}
                      className={`px-3 py-2 rounded-xl border text-[10px] font-black uppercase tracking-widest transition-all ${
                        showTimelineInspector
                          ? 'border-cyan-300 bg-cyan-500/20 text-white'
                          : 'border-gray-700 bg-black/40 text-gray-300 hover:text-white hover:border-cyan-300'
                      }`}
                    >
                      {showTimelineInspector ? 'Ocultar inspector' : 'Mostrar inspector'}
                    </button>
                    <button
                      onClick={() => {
                        setShowTimelineInspector(false);
                        setShowTimelineRenderPanel(false);
                      }}
                      className="px-3 py-2 rounded-xl border border-cyan-500/30 bg-cyan-500/10 text-[10px] font-black uppercase tracking-widest text-cyan-200 hover:text-white hover:border-cyan-300 transition-all"
                    >
                      Foco camadas
                    </button>
                    <button
                      onClick={() => setWorkspacePage('studio')}
                      className="px-3 py-2 rounded-xl border border-gray-700 bg-black/40 text-[10px] font-black uppercase tracking-widest text-gray-300 hover:text-white hover:border-blue-400 transition-all"
                    >
                      Voltar ao studio
                    </button>
                    <button
                      onClick={generateFullVideo}
                      disabled={!script || isGeneratingVideo}
                      className="px-4 py-2 rounded-xl bg-blue-600 hover:bg-blue-500 text-[10px] font-black uppercase tracking-widest disabled:opacity-50 transition-all"
                    >
                      {isGeneratingVideo ? 'Renderizando...' : 'Renderizar do timeline'}
                    </button>
                  </div>
                </div>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-[10px]">
                  <div className="bg-black/30 border border-gray-800 rounded-xl px-3 py-2 text-gray-300">Cenas: <span className="text-white font-mono">{script?.scenes.length || 0}</span></div>
                  <div className="bg-black/30 border border-gray-800 rounded-xl px-3 py-2 text-gray-300">Duracao estimada: <span className="text-white font-mono">{Math.round(timelineTotalSeconds)}s</span></div>
                  <div className="bg-black/30 border border-gray-800 rounded-xl px-3 py-2 text-gray-300">Musica global: <span className="text-white font-mono">{settings.localBackgroundMusic ? 'Local' : 'Nenhuma'}</span></div>
                  <div className="bg-black/30 border border-gray-800 rounded-xl px-3 py-2 text-gray-300">{timelineEditMode === 'post' ? 'Video pronto' : 'Volume musica'}: <span className="text-white font-mono">{timelineEditMode === 'post' ? (videoUrl ? 'SIM' : 'NAO') : `${Math.round(musicVolume * 100)}%`}</span></div>
                </div>
              </div>

              {script && (
                <div className="glass p-5 rounded-2xl border-blue-500/15 flex flex-col gap-4 shadow-2xl">
                  <div className="flex items-center justify-between">
                    <h3 className="font-black text-xs text-blue-500 uppercase tracking-[0.3em]">Painel de Render</h3>
                    <button
                      onClick={() => setShowTimelineRenderPanel(prev => !prev)}
                      className="px-3 py-2 rounded-xl border border-blue-500/40 bg-blue-500/10 text-[10px] font-black uppercase tracking-widest text-blue-200 hover:text-white hover:border-blue-300 transition-all"
                    >
                      {showTimelineRenderPanel ? 'Ocultar' : 'Expandir'}
                    </button>
                  </div>
                  {!showTimelineRenderPanel && (
                    <p className="text-[10px] text-gray-500">Painel recolhido para ampliar área de trabalho das camadas.</p>
                  )}

                  {showTimelineRenderPanel && (
                  <>
                  <div className="pt-1 text-[11px] space-y-3">
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-gray-500 font-mono">Roteiro</span>
                      <div className="flex gap-2">
                        <label className="text-[9px] font-bold px-3 py-2 bg-gray-900 rounded-xl border border-gray-800 cursor-pointer hover:border-blue-500/50 transition-all">
                          Importar
                          <input type="file" accept="application/json" className="hidden" onChange={(e) => e.target.files?.[0] && handleImportScript(e.target.files[0])} />
                        </label>
                        <button onClick={downloadScript} disabled={!script} className="text-[9px] font-bold px-3 py-2 bg-gray-900 rounded-xl border border-gray-800 text-gray-400 hover:text-white hover:border-blue-500/50 disabled:opacity-40 transition-all">Exportar</button>
                      </div>
                    </div>
                    {importError && <p className="text-xs text-red-400">{importError}</p>}
                  </div>

                  <div className="pt-3 border-t border-gray-800 text-[11px] space-y-2">
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-gray-500 font-mono">Limpar cache</span>
                      <div className="flex items-center gap-2">
                        <input
                          type="number"
                          min={1}
                          className="w-16 bg-black/40 border border-gray-800 rounded-lg px-2 py-1 text-[10px] text-white outline-none focus:border-blue-500/40"
                          value={cleanupAgeDays}
                          onChange={(e) => setCleanupAgeDays(Math.max(1, Number(e.target.value) || 7))}
                          title="Dias maximos para manter assets"
                        />
                        <button
                          onClick={cleanupCache}
                          disabled={isCleaningCache}
                          className="text-[9px] font-bold px-3 py-2 bg-gray-900 rounded-xl border border-gray-800 text-gray-400 hover:text-white hover:border-blue-500/50 disabled:opacity-50 transition-all"
                        >
                          {isCleaningCache ? 'Limpando...' : 'Executar'}
                        </button>
                      </div>
                    </div>
                    {cleanupResult && <p className="text-[10px] text-green-300">{cleanupResult}</p>}
                  </div>

                  <div className="pt-3 border-t border-gray-800 text-[11px] space-y-3">
                    <div className="flex items-start justify-between gap-3">
                      <span className="text-gray-500 font-mono">Volumes</span>
                      <div className="flex-1 grid grid-cols-2 gap-3">
                        <label className="flex flex-col gap-1 text-[10px] text-gray-400">
                          Musica {Math.round(musicVolume * 100)}%
                          <input type="range" min={0} max={1} step={0.05} value={musicVolume} onChange={(e)=>setMusicVolume(Number(e.target.value))} />
                        </label>
                        <label className="flex flex-col gap-1 text-[10px] text-gray-400">
                          Narracao {Math.round(narrationVolume * 100)}%
                          <input type="range" min={0.3} max={1.5} step={0.05} value={narrationVolume} onChange={(e)=>setNarrationVolume(Number(e.target.value))} />
                        </label>
                      </div>
                    </div>

                    <div className="flex items-start justify-between gap-3">
                      <span className="text-gray-500 font-mono">Legendas</span>
                      <div className="flex-1 space-y-2">
                        <div className="grid grid-cols-2 gap-2 text-[10px] text-gray-400">
                          <label className="flex flex-col gap-1">
                            Tamanho {captionScale.toFixed(2)}x
                            <input type="range" min={0.7} max={1.4} step={0.05} value={captionScale} onChange={(e)=>setCaptionScale(Number(e.target.value))} />
                          </label>
                          <label className="flex flex-col gap-1">
                            Opacidade BG {Math.round(captionBg*100)}%
                            <input type="range" min={0} max={1} step={0.05} value={captionBg} onChange={(e)=>setCaptionBg(Number(e.target.value))} />
                          </label>
                          <label className="flex flex-col gap-1">
                            Posicao Y {captionY}%
                            <input type="range" min={60} max={90} step={1} value={captionY} onChange={(e)=>setCaptionY(Number(e.target.value))} />
                          </label>
                        </div>
                        <div className="flex items-center gap-3 text-[10px] text-gray-300">
                          <span>Cor</span>
                          <input type="color" value={captionColor} onChange={(e)=>setCaptionColor(e.target.value)} />
                          <span>Destaque</span>
                          <input type="color" value={captionHighlight} onChange={(e)=>setCaptionHighlight(e.target.value)} />
                        </div>
                        <div className="mt-1 p-3 rounded-xl border border-gray-800 bg-black/40 text-center text-[11px]">
                          <span style={{backgroundColor:`rgba(0,0,0,${captionBg})`, padding:'6px 10px', borderRadius:'10px', display:'inline-block', color: captionColor}}>
                            LEGENDAS PREVIA <span style={{color: captionHighlight}}>DESTAQUE</span>
                          </span>
                        </div>
                      </div>
                    </div>

                    <div className="flex items-start justify-between gap-3">
                      <span className="text-gray-500 font-mono">Efeitos</span>
                      <div className="flex-1 grid grid-cols-2 gap-3 text-[10px] text-gray-400">
                        <label className="flex flex-col gap-1">
                          Forca de cor {colorStrength.toFixed(2)}
                          <input type="range" min={0} max={1.2} step={0.05} value={colorStrength} onChange={(e)=>setColorStrength(Number(e.target.value))} />
                        </label>
                        <label className="flex flex-col gap-1">
                          Duracao de transicao {transitionDuration.toFixed(2)}s
                          <input type="range" min={0.2} max={1.2} step={0.05} value={transitionDuration} onChange={(e)=>setTransitionDuration(Number(e.target.value))} />
                        </label>
                        <label className="flex flex-col gap-1 col-span-2">
                          Escala da imagem {imageScale.toFixed(2)}x
                          <input type="range" min={0.8} max={1.2} step={0.02} value={imageScale} onChange={(e)=>setImageScale(Number(e.target.value))} />
                        </label>
                      </div>
                    </div>
                  </div>
                  </>
                  )}
                </div>
              )}

              {!script ? (
                <div className="glass p-8 rounded-2xl border-white/10 text-center text-gray-400">
                  Gere ou importe um roteiro para editar na timeline.
                </div>
              ) : (
                <div className={`grid grid-cols-1 gap-4 ${showTimelineInspector ? 'xl:grid-cols-[330px_minmax(0,1fr)]' : 'xl:grid-cols-1'}`}>
                  {showTimelineInspector && (
                  <div className="glass p-4 rounded-2xl border-white/10 shadow-2xl">
                    <div className="flex items-center justify-between mb-3">
                      <h3 className="text-[11px] font-black uppercase tracking-[0.2em] text-gray-300">Inspector</h3>
                      <span className="text-[10px] font-mono text-cyan-300">
                        {selectedTimelineScene ? `Cena ${selectedTimelineScene.idx + 1}` : '--'}
                      </span>
                    </div>
                    {selectedTimelineScene ? (
                      <div className="space-y-3">
                        <div className="rounded-xl border border-gray-800 bg-black/40 overflow-hidden h-36">
                          {timelineEditMode === 'post' && videoUrl ? (
                            <video key={`tl-preview-${videoUrl}`} controls className="w-full h-full object-contain">
                              <source src={videoUrl} type="video/mp4" />
                            </video>
                          ) : selectedTimelineScene.scene.localImage ? (
                            <img src={selectedTimelineScene.scene.localImage} className="w-full h-full object-cover" />
                          ) : (
                            <div className="w-full h-full flex items-center justify-center text-[11px] text-gray-500">Sem imagem da cena</div>
                          )}
                        </div>
                        <div className="text-[10px] text-gray-400 font-mono">
                          Inicio {formatTimelineTime(selectedTimelineScene.startSec)} | Fim {formatTimelineTime(selectedTimelineScene.startSec + selectedTimelineScene.durationSec)}
                        </div>
                        <textarea
                          className="w-full h-32 bg-black/50 border border-gray-800 rounded-xl p-3 text-[11px] text-gray-200 outline-none focus:border-blue-500/40 resize-none"
                          value={selectedTimelineScene.scene.text}
                          onChange={(e) => updateScene(selectedTimelineScene.scene.id, { text: e.target.value })}
                        />
                        <div className="grid grid-cols-2 gap-2">
                          <button
                            onClick={() => generateNarration(selectedTimelineScene.scene.id, selectedTimelineScene.scene.text)}
                            disabled={isGeneratingNarration === selectedTimelineScene.scene.id}
                            className="text-[10px] px-2 py-2 rounded-lg border border-purple-500/40 text-purple-200 hover:text-white hover:border-purple-300 transition-all disabled:opacity-50"
                          >
                            {isGeneratingNarration === selectedTimelineScene.scene.id ? 'Gerando...' : 'Gerar voz'}
                          </button>
                          <button
                            onClick={() => selectedTimelineScene.scene.narrationAudio && playNarration(selectedTimelineScene.scene.narrationAudio)}
                            disabled={!selectedTimelineScene.scene.narrationAudio}
                            className="text-[10px] px-2 py-2 rounded-lg border border-blue-500/40 text-blue-200 hover:text-white hover:border-blue-300 transition-all disabled:opacity-40"
                          >
                            Ouvir narracao
                          </button>
                        </div>
                        <label className="block text-[10px] text-gray-400">
                          Volume narracao {Math.round((selectedTimelineScene.scene.narrationVolume ?? 1) * 100)}%
                          <input
                            type="range"
                            min={0}
                            max={1.8}
                            step={0.05}
                            value={selectedTimelineScene.scene.narrationVolume ?? 1}
                            onChange={(e) => updateScene(selectedTimelineScene.scene.id, { narrationVolume: Number(e.target.value) })}
                          />
                        </label>
                        <label className="block text-[10px] text-gray-400">
                          Trim inicio {Math.round(selectedTimelineScene.scene.trimStartMs ?? 0)}ms
                          <input
                            type="range"
                            min={0}
                            max={2500}
                            step={50}
                            value={selectedTimelineScene.scene.trimStartMs ?? 0}
                            onChange={(e) => updateScene(selectedTimelineScene.scene.id, { trimStartMs: Number(e.target.value) })}
                          />
                        </label>
                        <label className="block text-[10px] text-gray-400">
                          Trim fim {Math.round(selectedTimelineScene.scene.trimEndMs ?? 0)}ms
                          <input
                            type="range"
                            min={0}
                            max={2500}
                            step={50}
                            value={selectedTimelineScene.scene.trimEndMs ?? 0}
                            onChange={(e) => updateScene(selectedTimelineScene.scene.id, { trimEndMs: Number(e.target.value) })}
                          />
                        </label>
                        <label className="block text-[10px] text-gray-400">
                          Offset audio {Math.round(selectedTimelineScene.scene.audioOffsetMs ?? 0)}ms
                          <input
                            type="range"
                            min={-1500}
                            max={1500}
                            step={50}
                            value={selectedTimelineScene.scene.audioOffsetMs ?? 0}
                            onChange={(e) => updateScene(selectedTimelineScene.scene.id, { audioOffsetMs: Number(e.target.value) })}
                          />
                        </label>
                        <div className="grid grid-cols-3 gap-2">
                          <label className="text-[10px] px-2 py-2 rounded-lg border border-amber-500/40 text-amber-200 hover:text-white hover:border-amber-300 transition-all cursor-pointer text-center">
                            Upload SFX
                            <input type="file" className="hidden" accept="audio/*" onChange={(e) => e.target.files?.[0] && handleFileUpload('scene_sfx', selectedTimelineScene.scene.id, e.target.files[0])} />
                          </label>
                          <button
                            onClick={() => selectedTimelineScene.scene.localSfx && playNarration(selectedTimelineScene.scene.localSfx)}
                            disabled={!selectedTimelineScene.scene.localSfx}
                            className="text-[10px] px-2 py-2 rounded-lg border border-blue-500/40 text-blue-200 hover:text-white hover:border-blue-300 transition-all disabled:opacity-40"
                          >
                            Ouvir SFX
                          </button>
                          <button
                            onClick={() => updateScene(selectedTimelineScene.scene.id, { localSfx: undefined })}
                            disabled={!selectedTimelineScene.scene.localSfx}
                            className="text-[10px] px-2 py-2 rounded-lg border border-red-500/40 text-red-300 hover:text-white hover:border-red-300 transition-all disabled:opacity-40"
                          >
                            Remover
                          </button>
                        </div>
                        <label className="block text-[10px] text-gray-400">
                          Volume SFX {Math.round((selectedTimelineScene.scene.sfxVolume ?? 0.35) * 100)}%
                          <input
                            type="range"
                            min={0}
                            max={1.2}
                            step={0.05}
                            value={selectedTimelineScene.scene.sfxVolume ?? 0.35}
                            onChange={(e) => updateScene(selectedTimelineScene.scene.id, { sfxVolume: Number(e.target.value) })}
                          />
                        </label>
                        <label className="inline-flex w-full justify-center cursor-pointer text-[10px] px-2 py-2 rounded-lg border border-blue-500/40 text-blue-200 hover:text-white hover:border-blue-300 transition-all">
                          Trocar imagem
                          <input type="file" className="hidden" accept="image/*" onChange={(e) => e.target.files?.[0] && handleFileUpload('scene_image', selectedTimelineScene.scene.id, e.target.files[0])} />
                        </label>
                      </div>
                    ) : (
                      <div className="text-[11px] text-gray-500">Selecione um clip na timeline.</div>
                    )}
                  </div>
                  )}

                  <div className="glass p-4 rounded-2xl border-white/10 shadow-2xl overflow-hidden">
                    <div className="flex flex-wrap gap-2 items-center mb-3">
                      <button
                        onClick={() => setIsTimelinePlaying(prev => !prev)}
                        className="px-3 py-2 rounded-lg border border-cyan-500/40 bg-cyan-500/10 text-[10px] font-black uppercase tracking-widest text-cyan-200 hover:text-white hover:border-cyan-300 transition-all"
                      >
                        {isTimelinePlaying ? 'Pausar' : 'Play'}
                      </button>
                      <button
                        onClick={() => seekTimeline(activeTimelineScene ? activeTimelineScene.startSec : 0)}
                        className="px-3 py-2 rounded-lg border border-gray-700 bg-black/40 text-[10px] font-black uppercase tracking-widest text-gray-300 hover:text-white hover:border-blue-400 transition-all"
                      >
                        Ir para cena
                      </button>
                      <button
                        onClick={() => seekTimeline(0)}
                        className="px-3 py-2 rounded-lg border border-gray-700 bg-black/40 text-[10px] font-black uppercase tracking-widest text-gray-300 hover:text-white hover:border-blue-400 transition-all"
                      >
                        Início
                      </button>
                      <div className="text-[11px] font-mono text-gray-300 ml-auto">
                        {formatTimelineTime(timelinePlayheadSec)} / {formatTimelineTime(timelineDurationSafe)}
                      </div>
                    </div>
                    <div className="flex items-center gap-3 mb-4">
                      <input
                        type="range"
                        min={0}
                        max={timelineDurationSafe}
                        step={0.01}
                        value={timelinePlayheadSec}
                        onChange={(e) => seekTimeline(Number(e.target.value))}
                        className="flex-1 accent-cyan-400"
                      />
                      <div className="w-40 flex items-center gap-2">
                        <span className="text-[10px] text-gray-500 uppercase">Zoom</span>
                        <input
                          type="range"
                          min={0.5}
                          max={3}
                          step={0.1}
                          value={timelineZoom}
                          onChange={(e) => setTimelineZoom(Number(e.target.value))}
                          className="flex-1 accent-cyan-400"
                        />
                      </div>
                      <div className="w-44 flex items-center gap-2">
                        <span className="text-[10px] text-gray-500 uppercase">Altura</span>
                        <input
                          type="range"
                          min={64}
                          max={140}
                          step={2}
                          value={timelineTrackHeight}
                          onChange={(e) => setTimelineTrackHeight(Number(e.target.value))}
                          className="flex-1 accent-cyan-400"
                        />
                      </div>
                    </div>
                    <p className="text-[10px] text-gray-500 mb-3">Arraste os clips para reordenar as cenas na timeline. Use zoom/altura para trabalhar melhor com camadas de áudio e imagem.</p>

                    <div className="overflow-x-auto overflow-y-auto custom-scrollbar pb-2 max-h-[72vh]">
                      <div className="min-w-max space-y-2">
                        <div className="flex items-end gap-2">
                          <div className="w-24 text-[10px] uppercase tracking-widest text-gray-500">Tempo</div>
                          <div className="relative rounded-lg border border-gray-800 bg-black/50" style={{ width: `${timelineContentWidth}px`, height: `${timelineHeaderHeight}px` }}>
                            {timelineTicks.map(t => (
                              <div key={`tick-${t}`} className="absolute top-0 bottom-0 border-l border-white/10" style={{ left: `${Math.round(t * timelinePixelsPerSecond)}px` }}>
                                <span className="absolute top-1 left-1 text-[9px] text-gray-500 font-mono">{formatTimelineTime(t)}</span>
                              </div>
                            ))}
                            <div className="absolute top-0 bottom-0 w-[2px] bg-pink-400/80" style={{ left: `${Math.round(timelinePlayheadSec * timelinePixelsPerSecond)}px` }}></div>
                          </div>
                        </div>

                        {[{ id: 'video', label: 'Video' }, { id: 'narration', label: 'Narracao' }, { id: 'sfx', label: 'SFX' }].map(track => (
                          <div key={track.id} className="flex items-stretch gap-2">
                            <div className="w-24 rounded-lg border border-gray-800 bg-black/40 px-2 flex items-center text-[10px] uppercase tracking-widest text-gray-500" style={{ height: `${timelineTrackHeight}px` }}>{track.label}</div>
                            <div
                              className="relative rounded-lg border border-gray-800 bg-black/40"
                              style={{ width: `${timelineContentWidth}px`, height: `${timelineTrackHeight}px` }}
                              onClick={(e) => {
                                const rect = e.currentTarget.getBoundingClientRect();
                                const sec = (e.clientX - rect.left) / timelinePixelsPerSecond;
                                seekTimeline(sec);
                              }}
                            >
                              {timelineSceneMeta.map(({ scene, idx, startSec, durationSec }) => {
                                const isDraggingScene = timelineDrag?.sceneId === scene.id;
                                const renderStartSec = isDraggingScene ? (timelineDrag?.currentSec ?? startSec) : startSec;
                                const left = Math.round(renderStartSec * timelinePixelsPerSecond);
                                const width = Math.max(52, Math.round(durationSec * timelinePixelsPerSecond) - 2);
                                const selected = scene.id === timelineSelectedSceneId;
                                const muted = track.id === 'video'
                                  ? !scene.localImage
                                  : track.id === 'narration'
                                    ? !String(scene.text || '').trim()
                                    : !scene.localSfx;
                                const tone = track.id === 'video'
                                  ? 'from-blue-500/50 to-cyan-500/40 border-blue-400/40'
                                  : track.id === 'narration'
                                    ? 'from-purple-500/50 to-fuchsia-500/40 border-purple-400/40'
                                    : 'from-amber-500/50 to-orange-500/40 border-amber-400/40';
                                return (
                                  <button
                                    key={`${track.id}-${scene.id}`}
                                    onPointerDown={(e) => startTimelineClipDrag(e, scene.id, startSec, durationSec)}
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      setTimelineSelectedSceneId(scene.id);
                                      seekTimeline(startSec);
                                    }}
                                    className={`absolute top-1.5 bottom-1.5 rounded-md border bg-gradient-to-r px-2 text-left transition-all cursor-grab active:cursor-grabbing ${tone} ${selected ? 'ring-1 ring-white/60' : ''} ${muted ? 'opacity-30' : 'opacity-95 hover:opacity-100'} ${isDraggingScene ? 'z-20 shadow-[0_0_0_1px_rgba(255,255,255,0.5)]' : ''}`}
                                    style={{ left: `${left}px`, width: `${width}px` }}
                                  >
                                    <div className="text-[9px] font-black uppercase tracking-widest text-white/95">C{idx + 1}</div>
                                    <div className="text-[9px] font-mono text-white/80">
                                      {durationSec.toFixed(1)}s{track.id === 'narration' ? ` | T${Math.round(scene.trimStartMs ?? 0)}/${Math.round(scene.trimEndMs ?? 0)}ms` : ''}
                                    </div>
                                  </button>
                                );
                              })}
                              <div className="absolute top-0 bottom-0 w-[2px] bg-pink-400/80 pointer-events-none" style={{ left: `${Math.round(timelinePlayheadSec * timelinePixelsPerSecond)}px` }}></div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </>
          ) : (
            <>
          <div className="glass p-5 rounded-2xl border-white/5 shadow-2xl flex flex-col lg:flex-row gap-4 items-stretch">
            <div className="flex-1">
              <input
                type="text"
                className="w-full bg-black/40 border border-gray-800 rounded-2xl px-6 py-4 text-sm text-white focus:ring-1 focus:ring-blue-500/40 outline-none transition-all placeholder:text-gray-700 font-medium"
                placeholder="Digite o conceito ou tema para o vídeo..."
                value={topic}
                onChange={(e) => setTopic(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && generateScript()}
              />
            </div>
            <div className="flex flex-wrap gap-2 items-center">
              <select value={format} onChange={(e) => setFormat(e.target.value as any)} className="bg-gray-900 text-[10px] font-bold uppercase px-3 py-2 rounded-xl border border-gray-800 outline-none text-gray-400 hover:text-white transition-colors cursor-pointer">
                <option value="16:9">16:9</option>
                <option value="9:16">9:16</option>
              </select>
              <select value={videoLength} onChange={(e) => setVideoLength(e.target.value as '1m' | '5m')} className="bg-gray-900 text-[10px] font-bold uppercase px-3 py-2 rounded-xl border border-gray-800 outline-none text-gray-400 hover:text-white transition-colors cursor-pointer">
                <option value="1m">~1 min</option>
                <option value="5m">~5 min</option>
              </select>
              <button
                onClick={generateScript}
                disabled={isGeneratingScript || !topic.trim()}
                className="bg-blue-600 hover:bg-blue-500 px-6 py-3 rounded-2xl font-black text-[11px] transition-all flex items-center gap-2 shadow-lg shadow-blue-600/20 active:scale-95 disabled:opacity-50"
              >
                {isGeneratingScript ? <Loader /> : "Gerar Roteiro"}
              </button>
            </div>
          </div>

          {/* NODE FLOW */}
          <div className="glass rounded-3xl border-white/5 shadow-2xl p-6 overflow-hidden">
            <div className="flex flex-col md:flex-row md:items-end md:justify-between gap-3 mb-4">
              <div>
                <p className="text-[11px] font-black uppercase tracking-[0.22em] text-gray-400">Cards</p>
                <p className="text-[10px] text-gray-500">
                  {script ? `${script.scenes.length} cards — arraste para organizar. Ctrl+scroll para zoom.` : 'Gere um roteiro para comecar.'}
                </p>
              </div>
              <div className="flex flex-wrap items-center gap-2">
                <button onClick={zoomOut} className="px-3 py-2 rounded-xl border border-gray-800 bg-black/40 text-gray-300 hover:text-white hover:border-blue-500/40 transition-all" title="Zoom -">-</button>
                <input
                  type="range"
                  min={0.5}
                  max={1.6}
                  step={0.05}
                  value={canvasZoom}
                  onChange={(e) => setCanvasZoom(clampZoom(Number(e.target.value)))}
                  className="w-40 accent-blue-500"
                />
                <button onClick={zoomIn} className="px-3 py-2 rounded-xl border border-gray-800 bg-black/40 text-gray-300 hover:text-white hover:border-blue-500/40 transition-all" title="Zoom +">+</button>
                <span className="text-[10px] font-mono text-gray-500 w-14 text-right">{Math.round(canvasZoom * 100)}%</span>
                <button onClick={autoLayoutScenes} disabled={!script} className="px-3 py-2 rounded-xl border border-gray-800 bg-black/40 text-gray-300 hover:text-white hover:border-blue-500/40 transition-all disabled:opacity-40">Auto</button>
                <button onClick={resetCanvasView} className="px-3 py-2 rounded-xl border border-gray-800 bg-black/40 text-gray-300 hover:text-white hover:border-blue-500/40 transition-all">Reset</button>
                <button
                  onClick={copyAllScenePrompts}
                  disabled={!script || script.scenes.length === 0}
                  className="px-3 py-2 rounded-xl border border-cyan-500/40 bg-cyan-500/10 text-cyan-200 hover:text-white hover:border-cyan-300 transition-all disabled:opacity-40"
                >
                  Copiar prompts
                </button>
                <button
                  onClick={copyAllScenePromptsEn}
                  disabled={!script || script.scenes.length === 0}
                  className="px-3 py-2 rounded-xl border border-indigo-500/40 bg-indigo-500/10 text-indigo-200 hover:text-white hover:border-indigo-300 transition-all disabled:opacity-40"
                >
                  Copy prompts EN
                </button>
                <button onClick={addSceneCard} disabled={!script || script.scenes.length >= maxScenes} className="px-4 py-2 rounded-xl bg-blue-600 hover:bg-blue-500 text-white text-[10px] font-black uppercase tracking-[0.18em] transition-all disabled:opacity-40">+ Card</button>
              </div>
            </div>

            <div
              ref={canvasScrollRef}
              onWheel={handleCanvasWheel}
              className="relative h-[520px] md:h-[620px] xl:h-[700px] overflow-auto rounded-2xl border border-gray-800 bg-[radial-gradient(circle_at_1px_1px,rgba(148,163,184,0.10)_1px,transparent_1px)] [background-size:24px_24px]"
            >
              {!script && (
                <div className="absolute inset-0 flex items-center justify-center text-gray-600 text-sm">
                  Gere um roteiro para criar cards.
                </div>
              )}

              {script && (
                <div style={{ width: canvasBounds.w * canvasZoom, height: canvasBounds.h * canvasZoom, position: 'relative' }}>
                  <div style={{ width: canvasBounds.w, height: canvasBounds.h, transform: `scale(${canvasZoom})`, transformOrigin: '0 0', position: 'relative' }}>
                    <svg width={canvasBounds.w} height={canvasBounds.h} className="absolute inset-0 pointer-events-none">
                      <defs>
                        <marker id="arrow" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto" markerUnits="strokeWidth">
                          <path d="M0,0 L0,6 L6,3 z" fill="#60a5fa" />
                        </marker>
                      </defs>
                      {script.scenes.map(scene => {
                        if (!scene.flowTo) return null;
                        const from = scenePositions[scene.id];
                        const to = scenePositions[scene.flowTo];
                        if (!from || !to) return null;
                        const fromH = collapsedScenes[scene.id] ? CARD_H_COLLAPSED : CARD_H;
                        const toH = collapsedScenes[scene.flowTo] ? CARD_H_COLLAPSED : CARD_H;
                        const x1 = from.x + CARD_W / 2;
                        const y1 = from.y + fromH / 2;
                        const x2 = to.x + CARD_W / 2;
                        const y2 = to.y + toH / 2;
                        return (
                          <line
                            key={`${scene.id}->${scene.flowTo}`}
                            x1={x1}
                            y1={y1}
                            x2={x2}
                            y2={y2}
                            stroke="#60a5fa"
                            strokeWidth={2}
                            strokeDasharray="4 3"
                            markerEnd="url(#arrow)"
                            opacity={0.8}
                          />
                        );
                      })}
                    </svg>
                    {script.scenes.map((scene, idx) => {
                      const pos = scenePositions[scene.id] || { x: CANVAS_PAD, y: CANVAS_PAD };
                      const isCollapsed = !!collapsedScenes[scene.id];
                      const transitionValue = scene.transition || settings.defaultTransition || 'fade';
                      const filterValue = scene.filter || settings.defaultFilter || 'none';
                      const transitionLabel = transitionOptions.find(t => t.id === transitionValue)?.label || transitionValue;
                      const filterLabel = filterOptions.find(f => f.id === filterValue)?.label || filterValue;
                      const flowTargets = script.scenes.filter(s => s.id !== scene.id);
                      const inboundCount = script.scenes.filter(s => s.flowTo === scene.id).length;
                      return (
                        <div key={scene.id} className="absolute" style={{ left: pos.x, top: pos.y }}>
                          <div className={`relative w-72 glass rounded-2xl overflow-hidden border shadow-xl transition-all ${scene.localImage ? 'border-green-500/20' : 'border-gray-800'}`}>
                            {livePreview?.sceneId === scene.id && (
                              <div className="absolute inset-0 z-20 bg-black/70 backdrop-blur-sm text-white text-[11px] font-black uppercase tracking-[0.18em] flex items-center justify-center animate-pulse">
                                Pre-visualizacao: {livePreview.label}
                              </div>
                            )}
                            <div
                              title="Arraste para mover"
                              className={`flex items-center justify-between px-4 py-2 bg-black/40 border-b border-gray-800 select-none ${draggingSceneId === scene.id ? 'cursor-grabbing' : 'cursor-grab'}`}
                              onPointerDown={(e) => beginSceneDrag(scene.id, e)}
                              onPointerMove={moveSceneDrag}
                              onPointerUp={endSceneDrag}
                              onPointerCancel={endSceneDrag}
                            >
                              <div className="flex items-center gap-2">
                                <span className="w-5 h-5 rounded-full bg-gray-900 border border-gray-800 flex items-center justify-center text-[9px] font-black text-gray-400">{idx + 1}</span>
                                <span className="text-[10px] font-black uppercase tracking-widest text-gray-500">Cena</span>
                              </div>
                              <div className="flex items-center gap-2">
                                <span className="text-[9px] font-mono text-gray-600">{scene.localImage ? 'IMG' : 'SEM IMG'}</span>
                                <span className={`text-[9px] font-mono ${scene.localSfx ? 'text-amber-300' : 'text-gray-600'}`}>{scene.localSfx ? 'SFX' : 'SEM SFX'}</span>
                                <span className="text-[9px] px-2 py-1 rounded-full bg-blue-500/10 border border-blue-500/20 text-blue-100">{transitionValue}</span>
                                <span className="text-[9px] px-2 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-100">{filterValue}</span>
                                <button
                                  onPointerDown={(e) => e.stopPropagation()}
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    toggleSceneCollapse(scene.id);
                                  }}
                                  className="text-[9px] px-2 py-1 rounded-lg border border-gray-700 text-gray-300 hover:text-white hover:border-blue-400 transition-all"
                                  title={isCollapsed ? 'Expandir card' : 'Recolher card'}
                                >
                                  {isCollapsed ? 'Expandir' : 'Recolher'}
                                </button>
                              </div>
                            </div>

                            <div className={`${isCollapsed ? 'h-20' : 'h-28'} bg-gray-950 relative group/asset border-b border-gray-800`}>
                              {scene.localImage ? (
                                <img src={scene.localImage} className="w-full h-full object-cover transition-transform duration-700 group-hover/asset:scale-105" />
                              ) : (
                                <div className="w-full h-full flex items-center justify-center text-gray-800 bg-[conic-gradient(from_0deg_at_50%_50%,_#0a0a0a_0%,_#030712_100%)]">
                                  <svg className="w-10 h-10 opacity-10" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"></path></svg>
                                </div>
                              )}

                              <label className="absolute inset-0 bg-black/80 opacity-0 group-hover/asset:opacity-100 transition-all duration-200 flex flex-col items-center justify-center cursor-pointer backdrop-blur-sm">
                                <div className="w-10 h-10 rounded-full bg-blue-600 flex items-center justify-center mb-2 shadow-xl">
                                  <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M12 4v16m8-8H4"/></svg>
                                </div>
                                <span className="text-[11px] font-black uppercase tracking-widest text-white/70">Imagem</span>
                                <input type="file" className="hidden" accept="image/*" onChange={(e) => e.target.files?.[0] && handleFileUpload('scene_image', scene.id, e.target.files[0])} />
                              </label>
                            </div>

                            {isCollapsed ? (
                              <div className="px-4 py-3 flex flex-col gap-2">
                                <div className="text-[10px] text-gray-400 uppercase tracking-widest">Resumo</div>
                                <div className="text-[11px] text-gray-200 line-clamp-2">
                                  {scene.text || 'Sem narracao'}
                                </div>
                                <div className="text-[10px] text-cyan-200 bg-black/30 border border-gray-800 rounded-lg px-2 py-1 line-clamp-2">
                                  {(scene.visualPrompt || '').trim() || 'Sem prompt visual'}
                                </div>
                              </div>
                            ) : (
                            <div className="p-4 flex flex-col gap-3">
                              <div className="flex justify-between items-center">
                                <label className="text-[10px] font-black text-gray-500 uppercase tracking-widest">Narracao</label>
                                <button 
                                  onClick={() => generateNarration(scene.id, scene.text)}
                                  disabled={isGeneratingNarration === scene.id}
                                  className={`text-[9px] font-black px-3 py-1.5 rounded-xl border transition-all ${
                                    scene.narrationAudio 
                                    ? 'bg-purple-600 border-purple-400 text-white' 
                                    : 'bg-gray-900 border-gray-800 text-gray-500 hover:text-white'
                                  }`}
                                >
                                  {isGeneratingNarration === scene.id ? "..." : scene.narrationAudio ? "RE-GERAR" : "GERAR"}
                                </button>
                              </div>
                              
                              <div className="relative">
                                <textarea 
                                  className="w-full bg-black/40 rounded-xl p-3 text-xs text-gray-300 border border-gray-800 focus:border-blue-500/40 outline-none h-20 resize-none leading-relaxed"
                                  value={scene.text}
                                  onChange={(e) => updateScene(scene.id, { text: e.target.value })}
                                />
                                {scene.narrationAudio && (
                                  <button 
                                    onClick={() => playNarration(scene.narrationAudio!)} 
                                    className="absolute bottom-2 right-2 p-2 bg-blue-600/80 rounded-xl hover:bg-blue-500 transition-all shadow-xl"
                                    title="Ouvir"
                                  >
                                    <svg className="w-4 h-4 text-white" fill="currentColor" viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>
                                  </button>
                                )}
                              </div>

                              <div className="flex flex-col gap-2 border-t border-gray-800/40 pt-3">
                                <div className="flex items-center justify-between">
                                  <label className="text-[10px] font-black text-gray-500 uppercase tracking-widest">Efeito sonoro (SFX)</label>
                                  <div className="flex items-center gap-2">
                                    <label className="text-[9px] px-2 py-1 rounded-lg border border-amber-500/40 text-amber-200 hover:text-white hover:border-amber-300 transition-all cursor-pointer">
                                      Upload
                                      <input type="file" className="hidden" accept="audio/*" onChange={(e) => e.target.files?.[0] && handleFileUpload('scene_sfx', scene.id, e.target.files[0])} />
                                    </label>
                                    {scene.localSfx && (
                                      <>
                                        <button
                                          onClick={() => playNarration(scene.localSfx!)}
                                          className="text-[9px] px-2 py-1 rounded-lg border border-blue-500/40 text-blue-200 hover:text-white hover:border-blue-300 transition-all"
                                        >
                                          Ouvir
                                        </button>
                                        <button
                                          onClick={() => updateScene(scene.id, { localSfx: undefined })}
                                          className="text-[9px] px-2 py-1 rounded-lg border border-red-500/40 text-red-300 hover:text-white hover:border-red-300 transition-all"
                                        >
                                          Remover
                                        </button>
                                      </>
                                    )}
                                  </div>
                                </div>
                                <div className="text-[10px] text-gray-400">
                                  {scene.localSfx ? 'SFX carregado para esta cena' : 'Opcional: adicione som ambiente/pontual desta cena'}
                                </div>
                                <label className="text-[10px] text-gray-400 flex flex-col gap-1">
                                  Volume SFX {Math.round((scene.sfxVolume ?? 0.35) * 100)}%
                                  <input
                                    type="range"
                                    min={0}
                                    max={1.2}
                                    step={0.05}
                                    value={scene.sfxVolume ?? 0.35}
                                    onChange={(e) => updateScene(scene.id, { sfxVolume: Number(e.target.value) })}
                                  />
                                </label>
                              </div>

                              <div className="flex flex-col gap-2">
                                <div className="flex items-center justify-between">
                                  <label className="text-[10px] font-black text-gray-500 uppercase tracking-widest">Prompt visual</label>
                                  <button
                                    onClick={() => copyScenePrompt(scene, idx)}
                                    className="text-[9px] px-2 py-1 rounded-lg border border-cyan-500/40 text-cyan-200 hover:text-white hover:border-cyan-300 transition-all"
                                  >
                                    Copiar
                                  </button>
                                </div>
                                <textarea
                                  className="w-full bg-black/40 rounded-xl p-3 text-[11px] text-gray-200 border border-gray-800 focus:border-cyan-500/40 outline-none h-24 resize-none leading-relaxed"
                                  value={scene.visualPrompt || ''}
                                  onChange={(e) => updateScene(scene.id, { visualPrompt: e.target.value })}
                                  placeholder="Descreva aqui o prompt visual detalhado desta cena..."
                                />
                              </div>

                              <div className="grid grid-cols-2 gap-3 border-t border-gray-800/40 pt-3">
                                <div className="flex flex-col gap-1">
                                  <label className="text-[10px] uppercase text-gray-500 font-black">Transicao</label>
                                  <select
                                    value={transitionValue}
                                    onChange={(e) => updateScene(scene.id, { transition: e.target.value })}
                                    className="bg-black/40 text-xs p-2 rounded-xl border border-gray-800 text-gray-200 outline-none focus:border-blue-500/50 transition-all"
                                  >
                                    {transitionOptions.map(opt => (
                                      <option key={opt.id} value={opt.id}>{opt.label}</option>
                                    ))}
                                  </select>
                                  <button
                                    onClick={() => setLivePreview({ sceneId: scene.id, label: transitionLabel })}
                                    className="text-[9px] px-2 py-1 rounded-lg border border-blue-500/40 text-blue-200 hover:text-white hover:border-blue-300 transition-all"
                                  >
                                    Pré-visualizar
                                  </button>
                                </div>
                                <div className="flex flex-col gap-1">
                                  <label className="text-[10px] uppercase text-gray-500 font-black">Filtro</label>
                                  <select
                                    value={filterValue}
                                    onChange={(e) => updateScene(scene.id, { filter: e.target.value })}
                                    className="bg-black/40 text-xs p-2 rounded-xl border border-gray-800 text-gray-200 outline-none focus:border-blue-500/50 transition-all"
                                  >
                                    {filterOptions.map(opt => (
                                      <option key={opt.id} value={opt.id}>{opt.label}</option>
                                    ))}
                                  </select>
                                  <button
                                    onClick={() => setLivePreview({ sceneId: scene.id, label: filterLabel })}
                                    className="text-[9px] px-2 py-1 rounded-lg border border-emerald-500/40 text-emerald-200 hover:text-white hover:border-emerald-300 transition-all"
                                  >
                                    Pré-visualizar
                                  </button>
                                </div>
                                <div className="col-span-2 flex flex-col gap-1">
                                  <label className="text-[10px] uppercase text-gray-500 font-black">Fluxo entre cards</label>
                                  <select
                                    value={scene.flowTo || ''}
                                    onChange={(e) => updateScene(scene.id, { flowTo: e.target.value || null })}
                                    className="bg-black/40 text-xs p-2 rounded-xl border border-gray-800 text-gray-200 outline-none focus:border-blue-500/50 transition-all"
                                  >
                                    <option value="">Sem ligacao</option>
                                    {flowTargets.map((t) => (
                                      <option key={t.id} value={t.id}>
                                        Cena {script.scenes.findIndex(s => s.id === t.id) + 1}: {t.text.slice(0, 40) || '...'}
                                      </option>
                                    ))}
                                  </select>
                                  <div className="flex justify-between text-[10px] text-gray-500">
                                    <span>Saída: {scene.flowTo ? `→ Cena ${script.scenes.findIndex(s => s.id === scene.flowTo) + 1}` : 'isolado'}</span>
                                    <span>Entradas recebidas: {inboundCount}</span>
                                  </div>
                                </div>
                              </div>

                              <div className="flex justify-between gap-2 pt-3 border-t border-gray-800/50">
                                <button onClick={() => duplicateScene(scene)} className="flex-1 py-2 bg-gray-900/50 rounded-xl text-[9px] font-black text-gray-500 hover:text-white border border-gray-800 transition-all uppercase tracking-widest">Duplicar</button>
                                <button onClick={() => deleteScene(scene.id)} className="flex-1 py-2 bg-gray-900/50 rounded-xl text-[9px] font-black text-gray-500 hover:text-red-400 border border-gray-800 transition-all uppercase tracking-widest">Excluir</button>
                              </div>
                            </div>
                            )}
                          </div>
                        </div>
                      );
                    })}
                  </div>
          </div>
        )}
      </div>

      {/* Grid Splitter */}
      <div className="glass p-5 rounded-3xl border-white/5 flex flex-col gap-3">
        <div className="flex items-center justify-between">
          <p className="text-[11px] font-black uppercase text-blue-300">Grid Splitter</p>
          <button onClick={() => toggleSection('grid')} className="text-[11px] text-gray-400 hover:text-white">
            {openSections.grid ? '−' : '+'}
          </button>
        </div>
        {openSections.grid && (
          <>
            <div className="grid grid-cols-2 gap-2 text-[10px] text-gray-400">
              <label className="flex flex-col gap-1">
                <span className="font-black uppercase tracking-[0.3em] text-gray-500">Linhas</span>
                <input
                  type="number"
                  min={1}
                  max={6}
                  value={gridRows}
                  onChange={(e) => setGridRows(Math.max(1, Math.min(6, Number(e.target.value))))}
                  className="bg-black/40 border border-gray-800 rounded-xl px-3 py-2 text-[10px] text-white outline-none focus:border-blue-500/40"
                />
              </label>
              <label className="flex flex-col gap-1">
                <span className="font-black uppercase tracking-[0.3em] text-gray-500">Colunas</span>
                <input
                  type="number"
                  min={1}
                  max={6}
                  value={gridCols}
                  onChange={(e) => setGridCols(Math.max(1, Math.min(6, Number(e.target.value))))}
                  className="bg-black/40 border border-gray-800 rounded-xl px-3 py-2 text-[10px] text-white outline-none focus:border-blue-500/40"
                />
              </label>
            </div>

            <div className="flex flex-wrap gap-2 items-center text-[10px] text-gray-400">
              <label className="inline-flex items-center gap-2 px-3 py-2 bg-purple-600/20 border border-purple-500/40 rounded-xl cursor-pointer hover:border-purple-300 hover:text-white transition-all">
                Upload grade
                <input type="file" accept="image/*" className="hidden" onChange={(e) => handleGridUpload(e.target.files?.[0] || null)} />
              </label>
              <span>{gridProcessing ? 'Processando a grade...' : gridFile ? 'Grade carregada' : 'Nenhuma imagem'}</span>
              {gridError && <span className="text-red-400 text-[9px]">{gridError}</span>}
            </div>

            {script && script.scenes.length > 0 && (
              <div className="flex items-center gap-2 text-[10px] text-gray-300">
                <span className="font-black uppercase tracking-[0.3em]">Cena alvo</span>
                <select
                  value={gridSceneTarget || ''}
                  onChange={(e) => setGridSceneTarget(e.target.value || null)}
                  className="bg-black/40 border border-gray-800 rounded-xl px-3 py-2 text-[10px] text-white outline-none focus:border-blue-500/40"
                >
                  {script.scenes.map((scene, idx) => (
                    <option key={scene.id} value={scene.id}>
                      {`${idx + 1}. ${scene.visualPrompt?.slice(0, 18) || 'Cena'}`}
                    </option>
                  ))}
                </select>
              </div>
            )}

            <div className="grid grid-cols-3 gap-2">
              {gridTiles.length === 0 && !gridProcessing && (
                <div className="col-span-3 text-[10px] text-gray-500 text-center">Use o upload para fatiar uma grade.</div>
              )}
              {gridTiles.map(tile => (
                <div key={tile.id} className="bg-black/30 border border-gray-800 rounded-2xl overflow-hidden flex flex-col gap-2 text-[9px] text-gray-200">
                  <div className="h-20 overflow-hidden">
                    <img src={tile.dataUrl} className="w-full h-full object-cover" />
                  </div>
                  <div className="px-2 flex items-center justify-between">
                    <span>R{tile.row + 1} C{tile.col + 1}</span>
                    <span>{gridSceneTarget ? 'Target' : 'Sem cena'}</span>
                  </div>
                  <div className="px-2 flex gap-1">
                    <a
                      href={tile.dataUrl}
                      download={`grid-${tile.row + 1}x${tile.col + 1}.png`}
                      className="flex-1 text-center px-2 py-1 rounded-xl border border-gray-700 hover:border-blue-500 hover:text-white transition-all"
                    >
                      Baixar
                    </a>
                    <button
                      onClick={() => copyTileAsBase64(tile)}
                      className="flex-1 px-2 py-1 rounded-xl border border-gray-700 hover:border-blue-500 hover:text-white transition-all"
                    >
                      Copiar
                    </button>
                  </div>
                  <div className="px-2 pb-2">
                    <button
                      disabled={!gridSceneTarget}
                      onClick={() => applyTileToScene(tile)}
                      className="w-full text-[9px] font-black uppercase tracking-[0.2em] px-2 py-1 rounded-xl border border-blue-500/40 bg-blue-600/20 text-blue-200 hover:border-blue-300 hover:text-white transition-all disabled:opacity-40"
                    >
                      Aplicar à cena
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </>
        )}
      </div>
          </div>

          {/* RENDER */}
          {script && (
            <div className="pb-12">
              <div className="glass rounded-[2.5rem] p-6 flex flex-col gap-4 relative border border-gray-800/70 overflow-hidden shadow-2xl">
                {isGeneratingVideo && (
                  <div className="absolute inset-0 bg-black/90 backdrop-blur-2xl z-[60] flex flex-col items-center justify-center rounded-[2.5rem] p-12 text-center">
                    <div className="w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mb-6"></div>
                    <h2 className="text-3xl font-black mb-4 tracking-tighter uppercase">Processando</h2>
                    <p className="text-gray-400 font-mono text-xs max-w-sm uppercase tracking-widest opacity-60 mb-4">{statusMessage}</p>
                    <div className="w-full max-w-md text-left">
                      <div className="flex items-center justify-between text-[11px] font-mono text-gray-400 uppercase tracking-widest mb-1">
                        <span>{renderStage || 'Etapa'}</span>
                        <span>{renderProgressLabel}%</span>
                      </div>
                      <div className="relative w-full h-2 rounded-full bg-white/10 overflow-hidden">
                        <div className="h-full bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400" style={{ width: `${renderProgress}%` }}></div>
                        {renderIsWaiting && (
                          <div className="absolute inset-0 bg-white/15 animate-pulse pointer-events-none"></div>
                        )}
                      </div>
                      {String(renderStage || '').toLowerCase() === 'render' && (
                        <div className="mt-2">
                          <div className="flex items-center justify-between text-[10px] font-mono text-gray-500 mb-1">
                            <span>Montagem real</span>
                            <span>{renderRealPct.toFixed(1)}%</span>
                          </div>
                          <div className="w-full h-1.5 rounded-full bg-white/10 overflow-hidden">
                            <div className="h-full bg-gradient-to-r from-emerald-400 to-cyan-400" style={{ width: `${renderRealPct}%` }}></div>
                          </div>
                          {renderRealInfo && (
                            <div className="mt-1 text-[10px] font-mono text-emerald-300">{renderRealInfo}</div>
                          )}
                        </div>
                      )}
                      <div className="mt-2 flex items-center justify-between text-[10px] font-mono text-gray-500">
                        <span className={`${renderIsWaiting ? 'text-cyan-300' : 'text-gray-500'}`}>
                          {renderIsWaiting
                            ? `Render ativo. ${Math.round(renderStaleSeconds)}s sem novos frames, aguarde.`
                            : renderRealInfo || 'Recebendo progresso do backend.'}
                        </span>
                        <span>{formatRenderElapsed(renderElapsedSeconds)}</span>
                      </div>
                    </div>
                  </div>
                )}

                <div className="flex items-start justify-between gap-3">
                  <div>
                    <h3 className="text-[11px] font-black uppercase tracking-[0.22em] text-gray-400">Saida do Video</h3>
                    <p className="text-[10px] text-gray-500">{videoUrl ? 'Preview pronto.' : 'Quando renderizar, a previa aparece aqui.'}</p>
                  </div>
                  {videoUrl && (
                    <div className="flex items-center gap-2">
                      <button onClick={() => setVideoUrl(null)} className="px-3 py-2 rounded-xl border border-gray-800 bg-black/40 text-[10px] font-black uppercase tracking-widest text-gray-300 hover:text-white hover:border-blue-500/40 transition-all">Reiniciar</button>
                      <button
                        onClick={() => {
                          setTimelineEditMode('post');
                          setWorkspacePage('timeline');
                        }}
                        className="px-3 py-2 rounded-xl border border-cyan-500/40 bg-cyan-500/10 text-[10px] font-black uppercase tracking-widest text-cyan-200 hover:text-white hover:border-cyan-300 transition-all"
                      >
                        Reeditar timeline
                      </button>
                      <a href={videoUrl} download="velozz_render.mp4" className="px-4 py-2 rounded-xl bg-white text-black text-[10px] font-black uppercase tracking-widest hover:bg-gray-200 transition-all shadow-xl active:scale-95">Baixar MP4</a>
                    </div>
                  )}
                </div>

                <div className={`mx-auto w-full bg-black rounded-2xl border border-gray-800 overflow-hidden ${format === '16:9' ? 'aspect-video max-w-[760px]' : 'aspect-[9/16] max-w-[340px]'}`}>
                  {videoUrl ? (
                    <video key={videoUrl} controls preload="metadata" playsInline className="w-full h-full object-contain">
                      <source src={videoUrl} type="video/mp4" />
                    </video>
                  ) : (
                    <div className="w-full h-full flex flex-col items-center justify-center text-center p-6 text-gray-500">
                      <div className="text-xs font-black uppercase tracking-widest opacity-70">Monitor</div>
                      <div className="text-[11px] mt-2 opacity-60">Sem video ainda.</div>
                    </div>
                  )}
                </div>
              </div>

            </div>
          )}
            </>
          )}
        </main>
      </div>
    </div>
  );
};

const Loader: React.FC = () => (
  <svg className="animate-spin h-5 w-5 text-white" viewBox="0 0 24 24">
    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
  </svg>
);

const root = createRoot(document.getElementById('root')!);
root.render(<App />);







