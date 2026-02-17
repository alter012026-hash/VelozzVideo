import React, { useState, useEffect, useMemo, useRef } from 'react';
import { createRoot } from 'react-dom/client';

// --- Types ---
interface Scene {
  id: string;
  text: string;
  visualPrompt: string;
  localImage?: string;
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

interface GridTile {
  id: string;
  row: number;
  col: number;
  dataUrl: string;
}

interface SessionPayload {
  topic: string;
  format: '16:9' | '9:16';
  videoLength: '1m' | '5m';
  settings: GeneralSettings;
  script: Script | null;
  scenePositions: Record<string, { x: number; y: number }>;
  gridRows: number;
  gridCols: number;
  gridSceneTarget: string | null;
  edgeVoiceId: string;
}

const SESSION_KEY = 'vv_session';

const App: React.FC = () => {
  const [topic, setTopic] = useState('');
  const [videoLength, setVideoLength] = useState<'1m' | '5m'>('1m');
  const [currentStep, setCurrentStep] = useState<WorkflowStep>('topic');
  const stepOrder: WorkflowStep[] = ['topic', 'scripting', 'assets', 'rendering', 'done'];
  const transitionOptions = [
    { id: 'fade', label: 'Fade in/out (padrão)' },
    { id: 'crossfade', label: 'Crossfade' },
    { id: 'slide_left', label: 'Slide Left' },
    { id: 'slide_right', label: 'Slide Right' },
    { id: 'slide_up', label: 'Slide Up' },
    { id: 'slide_down', label: 'Slide Down' },
    { id: 'none', label: 'Sem transição' },
  ];
  const filterOptions = [
    { id: 'none', label: 'Nenhum' },
    { id: 'cinematic', label: 'Cinematic quente' },
    { id: 'cool', label: 'Frio / teal' },
    { id: 'warm', label: 'Quente' },
    { id: 'bw', label: 'Preto e branco' },
    { id: 'vibrant', label: 'Vibrante' },
    { id: 'vhs', label: 'VHS grão leve' },
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
  const staticEdgeVoices = [
    { id: 'pt-BR-ThalitaMultilingualNeural', label: 'ThalitaMultilingualNeural', gender: 'Female', locale: 'pt-BR' },
    { id: 'pt-BR-FranciscaNeural', label: 'FranciscaNeural', gender: 'Female', locale: 'pt-BR' },
    { id: 'pt-BR-AntonioNeural', label: 'AntonioNeural', gender: 'Male', locale: 'pt-BR' },
    { id: 'pt-PT-DuarteNeural', label: 'DuarteNeural', gender: 'Male', locale: 'pt-PT' },
    { id: 'pt-PT-RaquelNeural', label: 'RaquelNeural', gender: 'Female', locale: 'pt-PT' },
  ];
  const edgeVoiceOptions = edgeVoices.length ? edgeVoices : staticEdgeVoices;
  const selectedEdgeVoice = edgeVoiceOptions.find(v => v.id === edgeVoiceId) || edgeVoiceOptions[0];
  const [openSections, setOpenSections] = useState({
    ollama: true,
    theme: true,
    effects: true,
    audio: true,
    voice: true,
    backend: true,
    grid: true,
  });

  const toggleSection = (key: keyof typeof openSections) =>
    setOpenSections(prev => ({ ...prev, [key]: !prev[key] }));

  const maxScenes = videoLength === '1m' ? 6 : 20; // estimativa: ~10s/cena ou ~15s/cena
  const CARD_W = 288;
  const CARD_H = 340;
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

  // resolve API base (auto fallback, scans common ports)
  useEffect(() => {
    let cancelled = false;
    const resolveApi = async () => {
      const saved = localStorage.getItem('vv_api_base');
      const envBase = (import.meta as any).env?.API_HOST || (import.meta as any).env?.VITE_API_HOST;
      const portRange = Array.from({ length: 11 }, (_, i) => 8000 + i);
      const portCandidates = portRange.flatMap(p => [`http://127.0.0.1:${p}`, `http://localhost:${p}`]);
      const candidates = [
        saved,
        envBase,
        window.location.origin,
        ...portCandidates,
      ].filter(Boolean) as string[];

      for (const base of candidates) {
        try {
          const ctrl = new AbortController();
          const timer = setTimeout(() => ctrl.abort(), 1200);
          const res = await fetch(`${base}/api/ping`, { signal: ctrl.signal });
          clearTimeout(timer);
          if (res.ok) {
            if (!cancelled) {
              setApiBase(base);
              localStorage.setItem('vv_api_base', base);
            }
            return;
          }
        } catch (err) {
          continue;
        }
      }
      setApiBase('');
    };
    resolveApi();
    return () => {
      cancelled = true;
    };
  }, []);

  const apiFetch = (path: string, options?: RequestInit) => {
    if (!apiBase) throw new Error('API indisponível');
    return fetch(`${apiBase}${path}`, options);
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

  const autoLayoutScenes = () => {
    if (!script) return;
    setScenePositions(() => {
      const next: Record<string, { x: number; y: number }> = {};
      script.scenes.forEach((s, idx) => {
        const col = idx % GRID_COLS;
        const row = Math.floor(idx / GRID_COLS);
        next[s.id] = {
          x: CANVAS_PAD + col * (CARD_W + GRID_GAP),
          y: CANVAS_PAD + row * (CARD_H + GRID_GAP),
        };
      });
      return next;
    });
    pushLog('Cards organizados (auto layout)', 'info');
  };

  // --- General Settings ---
  const [settings, setSettings] = useState<GeneralSettings>({
    theme: 'Cinematographic',
    backgroundMusic: 'Áudio Local',
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
    setScenePositions(payload.scenePositions || {});
    setGridRows(payload.gridRows || 2);
    setGridCols(payload.gridCols || 3);
    setGridSceneTarget(payload.gridSceneTarget || null);
    if (payload.edgeVoiceId) setEdgeVoiceId(payload.edgeVoiceId);
    if (payload.script) {
      const scenesWithStatus = payload.script.scenes.map((s, idx) => ({
        ...s,
        id: String((s as any).id ?? idx + 1),
        status: s.status ?? 'completed',
        transition: s.transition || payload.settings?.defaultTransition || settings.defaultTransition || 'fade',
        filter: s.filter || payload.settings?.defaultFilter || settings.defaultFilter || 'none',
        flowTo: typeof s.flowTo === 'string' ? String(s.flowTo) : null,
      }));
      setScript({ ...payload.script, scenes: scenesWithStatus });
      setCurrentStep('assets');
    } else {
      setScript(null);
      setCurrentStep('topic');
    }
    return true;
  };

  const canvasBounds = useMemo(() => {
    if (!script || script.scenes.length === 0) return { w: MIN_CANVAS_W, h: MIN_CANVAS_H };
    let maxX = 0;
    let maxY = 0;
    for (const s of script.scenes) {
      const pos = scenePositions[s.id] || { x: 0, y: 0 };
      if (pos.x > maxX) maxX = pos.x;
      if (pos.y > maxY) maxY = pos.y;
    }
    const w = Math.max(MIN_CANVAS_W, maxX + CARD_W + CANVAS_PAD);
    const h = Math.max(MIN_CANVAS_H, maxY + CARD_H + CANVAS_PAD);
    return { w, h };
  }, [script, scenePositions]);

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
    if (!script || !script.scenes.length) {
      setGridSceneTarget(null);
      return;
    }
    setGridSceneTarget(prev => script.scenes.find(s => s.id === prev)?.id || script.scenes[0].id);
  }, [sceneIdKey]);

  useEffect(() => {
    const payload: SessionPayload = {
      topic,
      format,
      videoLength,
      settings,
      script,
      scenePositions,
      gridRows,
      gridCols,
      gridSceneTarget,
      edgeVoiceId,
    };
    try {
      localStorage.setItem(SESSION_KEY, JSON.stringify(payload));
    } catch {
      // ignore quota issues
    }
  }, [topic, format, videoLength, settings, script, scenePositions, gridRows, gridCols, gridSceneTarget, edgeVoiceId]);

  const handleCanvasWheel = (e: React.WheelEvent<HTMLDivElement>) => {
    // Ctrl+scroll: zoom (trackpad pinch also triggers ctrlKey on many browsers)
    if (!e.ctrlKey) return;
    e.preventDefault();
    const direction = e.deltaY > 0 ? -1 : 1;
    setCanvasZoom(z => clampZoom(Number((z + direction * 0.05).toFixed(2))));
  };

  const beginSceneDrag = (sceneId: string, e: React.PointerEvent<HTMLDivElement>) => {
    if (e.button !== 0) return;
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
      text: 'Nova narração',
      visualPrompt: 'Nova descrição',
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
      pushLog(`Falha na prévia de voz: ${err?.message || err}`, 'error');
    } finally {
      setIsPreviewingVoice(false);
    }
  };

  const handleAiError = (error: any, context: string) => {
    console.error(`${context} failed:`, error);
    const errorMessage = String(error?.message || error || 'Erro desconhecido');
    pushLog(`${context} falhou: ${errorMessage}`, 'error');
    setStatusMessage(`${context} falhou: ${errorMessage}`);
  };

  const cleanupCache = async () => {
    if (!apiBase) {
      setStatusMessage('API indisponÃ­vel para limpeza de cache');
      return;
    }
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
      pushLog(`Limpeza concluÃ­da: ${summary}`, 'info');
    } catch (error: any) {
      handleAiError(error, 'Limpeza de cache');
    } finally {
      setIsCleaningCache(false);
    }
  };

  const restoreSession = () => {
    const stored = localStorage.getItem(SESSION_KEY);
    if (!stored) {
      pushLog('Nenhuma sessão salva', 'warn');
      return;
    }
    try {
      const payload = JSON.parse(stored) as SessionPayload;
      if (applySessionPayload(payload)) {
        pushLog('Sessão restaurada', 'info');
      }
    } catch (err: any) {
      pushLog(`Erro ao restaurar sessão: ${err?.message || err}`, 'error');
    }
  };

  const generateScript = async () => {
    if (!topic.trim()) return;
    setIsGeneratingScript(true);
    setCurrentStep('scripting');
    setStatusMessage('IA Projetando Roteiro e Sincronizando Assets...');
    pushLog('Iniciando geração de roteiro', 'info');

    const prompt = `Crie um roteiro de vídeo sobre "${topic}". 
    O tema visual geral é "${settings.theme}".
    Retorne JSON com "title" e array "scenes" (id, text, visualPrompt). Máximo ${maxScenes} cenas.`;

    try {
      localStorage.setItem('ollama_host', ollamaHost);
      localStorage.setItem('ollama_model', ollamaModel);
      const response = await fetch(`${ollamaHost.replace(/\/$/, '')}/api/generate`, {
        method: 'POST',
        body: JSON.stringify({ model: ollamaModel, prompt: `${prompt}. Responda apenas com o JSON puro.`, stream: false, format: 'json' }),
      });
      const resData = await response.json();
      const data = JSON.parse(String(resData.response));
      
      const scenesWithStatus = data.scenes.map((s: any, idx: number) => ({
        ...s,
        id: String(s.id ?? idx + 1),
        status: 'completed',
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
      pushLog(`Narração gerada para cena ${sceneId}`, 'info');
    } catch (error: any) {
      handleAiError(error, 'Narração');
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
    const newScene = { ...scene, id: Math.random().toString(36).substr(2, 9), narrationAudio: undefined, flowTo: null };
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

  const handleFileUpload = (type: 'scene_image' | 'global_theme' | 'global_music', id: string, file: File) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      const result = reader.result as string;
      if (type === 'scene_image') {
        updateScene(id, { localImage: result });
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
        const scenesWithStatus = json.scenes.map((s: any) => ({
          ...s,
          status: s.status ?? 'completed',
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
    setCurrentStep('topic');
    setStatusMessage('');
    setScenePositions({});
    localStorage.removeItem(SESSION_KEY);
  };

  const generateFullVideo = async () => {
    if (!script) return;
    if (!apiBase) {
      setStatusMessage('API indisponível para render');
      return;
    }

    setIsGeneratingVideo(true);
    setCurrentStep('rendering');
    setRenderProgress(0);
    setRenderStage('Iniciando');
    setStatusMessage('Iniciando render com motor local...');

    try {
      const hasLocalAssets = script.scenes.some(s => s.localImage) || settings.localThemeRef || settings.localBackgroundMusic;

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
          animationType: (s as any).animationType,
          transition: s.transition || settings.defaultTransition,
          filter: s.filter || settings.defaultFilter,
          flowTo: s.flowTo ? String(s.flowTo) : null,
        })),
        format,
        voice: edgeVoiceId,
        scriptTitle: script.title,
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
        setRenderStage(st.stage || '');
        setRenderProgress(Math.round((st.progress || 0) * 100));
        setStatusMessage(st.message || '');
        if (st.done) {
          done = true;
          if (st.error) throw new Error(st.error);
          const url = st.web_url || st.output || null;
          setVideoUrl(url);
          pushLog(`Render concluído: ${url || st.output}`, 'info');
          setCurrentStep('done');
        } else {
          await new Promise(r => setTimeout(r, 1500));
        }
      }
    } catch (error: any) {
      setCurrentStep('assets');
      handleAiError(error, 'Vídeo');
    } finally {
      setIsGeneratingVideo(false);
      setStatusMessage('');
      setRenderStage('');
      setRenderTaskId(null);
      setRenderProgress(0);
    }
  };

  const testRenderPipeline = async () => {
    if (isGeneratingVideo || isTestingRender) return;
    setIsTestingRender(true);
    setStatusMessage('Testando pipeline local com payload padrão...');
    try {
      const testRequest = {
        scenes: [
          { id: 'test-1', text: 'Cena de teste automática gerada no backend.', animationType: 'kenburns', transition: settings.defaultTransition, filter: settings.defaultFilter },
          { id: 'test-2', text: 'Segunda cena para validar legendas karaokê.', animationType: 'zoom_in', transition: settings.defaultTransition, filter: settings.defaultFilter },
        ],
        format,
        voice: edgeVoiceId,
        scriptTitle: 'Teste local',
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
      pushLog(`Teste de render concluído: ${data.output}`, 'info');
      setStatusMessage(`Teste completo. Saída: ${data.output}`);
    } catch (error: any) {
      handleAiError(error, 'Teste de render');
      setStatusMessage('Falha no teste de render.');
    } finally {
      setIsTestingRender(false);
    }
  };

  const progress = ((stepOrder.indexOf(currentStep) + 1) / stepOrder.length) * 100;

  return (
    <div className="min-h-screen bg-[#020617] selection:bg-blue-500/30">
      <div className="max-w-7xl mx-auto p-4 md:p-6 flex flex-col md:flex-row gap-6">
        {/* SIDEBAR */}
        <aside className="w-full md:w-80 flex flex-col gap-4">
          <div className="glass p-4 rounded-2xl border-white/5 shadow-2xl">
            <div className="flex items-center justify-between">
              <div>
                <h1 className="text-2xl font-extrabold font-display gradient-text tracking-tighter">VelozzVideo</h1>
                <p className="text-gray-500 text-[11px] font-bold uppercase tracking-[0.18em]">Fluxo Local + API</p>
              </div>
              <div className="flex gap-2">
                <button onClick={restoreSession} className="text-[10px] font-bold uppercase px-3 py-2 rounded-xl border border-green-500/40 text-green-300 hover:text-white hover:border-green-400 transition-all">Recuperar sessão</button>
                <button onClick={resetProject} className="text-[10px] font-bold uppercase px-3 py-2 rounded-xl border border-red-500/40 text-red-300 hover:text-white hover:border-red-400 transition-all">Resetar</button>
              </div>
            </div>
            <div className="mt-3 grid grid-cols-3 gap-2 text-[10px] text-gray-400">
              <div className="px-2 py-1 rounded-lg bg-black/40 border border-gray-800">Etapa: <span className="text-white">{currentStep}</span></div>
              <div className="px-2 py-1 rounded-lg bg-black/40 border border-gray-800">Form.: {format}</div>
              <div className="px-2 py-1 rounded-lg bg-black/40 border border-gray-800">Duração: {videoLength === '1m' ? '1m' : '5m'}</div>
            </div>
            <div className="mt-3 h-2 w-full bg-gray-900 rounded-full overflow-hidden">
              <div className="h-full bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500" style={{ width: `${progress}%` }}></div>
            </div>
          </div>

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
                <label className="text-[10px] text-gray-500 uppercase font-bold">Transição padrão</label>
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
                  Arraste ou clique para definir a trilha. Fontes online estão desativadas para priorizar conteúdo local e seguro.
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
        <main className="flex-1 flex flex-col gap-6">
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
                <button onClick={addSceneCard} disabled={!script || script.scenes.length >= maxScenes} className="px-4 py-2 rounded-xl bg-blue-600 hover:bg-blue-500 text-white text-[10px] font-black uppercase tracking-[0.18em] transition-all disabled:opacity-40">+ Card</button>
              </div>
            </div>

            <div
              ref={canvasScrollRef}
              onWheel={handleCanvasWheel}
              className="relative h-[520px] md:h-[600px] overflow-auto rounded-2xl border border-gray-800 bg-[radial-gradient(circle_at_1px_1px,rgba(148,163,184,0.10)_1px,transparent_1px)] [background-size:24px_24px]"
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
                        const x1 = from.x + CARD_W / 2;
                        const y1 = from.y + CARD_H / 2;
                        const x2 = to.x + CARD_W / 2;
                        const y2 = to.y + CARD_H / 2;
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
                      const transitionValue = scene.transition || settings.defaultTransition || 'fade';
                      const filterValue = scene.filter || settings.defaultFilter || 'none';
                      const transitionLabel = transitionOptions.find(t => t.id === transitionValue)?.label || transitionValue;
                      const filterLabel = filterOptions.find(f => f.id === filterValue)?.label || filterValue;
                      const flowTargets = script.scenes.filter(s => s.id !== scene.id);
                      const inboundCount = script.scenes.filter(s => s.flowTo === scene.id).length;
                      return (
                        <div key={scene.id} className="absolute" style={{ left: pos.x, top: pos.y }}>
                          <div className={`relative w-72 glass rounded-2xl overflow-hidden border shadow-xl transition-colors ${scene.localImage ? 'border-green-500/20' : 'border-gray-800'}`}>
                            {livePreview?.sceneId === scene.id && (
                              <div className="absolute inset-0 z-20 bg-black/70 backdrop-blur-sm text-white text-[11px] font-black uppercase tracking-[0.18em] flex items-center justify-center animate-pulse">
                                Pré-visualização: {livePreview.label}
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
                                <span className="text-[9px] px-2 py-1 rounded-full bg-blue-500/10 border border-blue-500/20 text-blue-100">{transitionValue}</span>
                                <span className="text-[9px] px-2 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-100">{filterValue}</span>
                              </div>
                            </div>

                            <div className="h-28 bg-gray-950 relative group/asset border-b border-gray-800">
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

                              <div className="grid grid-cols-2 gap-3 border-t border-gray-800/40 pt-3">
                                <div className="flex flex-col gap-1">
                                  <label className="text-[10px] uppercase text-gray-500 font-black">Transição</label>
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
                                    <option value="">Sem ligação</option>
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

          {/* RENDER + STATUS */}
          {script && (
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 pb-12">
              <div className="lg:col-span-7 glass rounded-[2.5rem] p-6 min-h-[360px] flex flex-col gap-4 relative border border-gray-800/70 overflow-hidden shadow-2xl">
                {isGeneratingVideo && (
                  <div className="absolute inset-0 bg-black/90 backdrop-blur-2xl z-[60] flex flex-col items-center justify-center rounded-[2.5rem] p-12 text-center">
                    <div className="w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mb-6"></div>
                    <h2 className="text-3xl font-black mb-4 tracking-tighter uppercase">Processando</h2>
                    <p className="text-gray-400 font-mono text-xs max-w-sm uppercase tracking-widest opacity-60 mb-4">{statusMessage}</p>
                    <div className="w-full max-w-md text-left">
                      <div className="flex items-center justify-between text-[11px] font-mono text-gray-400 uppercase tracking-widest mb-1">
                        <span>{renderStage || 'Etapa'}</span>
                        <span>{renderProgress.toFixed(0)}%</span>
                      </div>
                      <div className="w-full h-2 rounded-full bg-white/10 overflow-hidden">
                        <div className="h-full bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400" style={{ width: `${renderProgress}%` }}></div>
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
                      <a href={videoUrl} download="velozz_render.mp4" className="px-4 py-2 rounded-xl bg-white text-black text-[10px] font-black uppercase tracking-widest hover:bg-gray-200 transition-all shadow-xl active:scale-95">Baixar MP4</a>
                    </div>
                  )}
                </div>

                <div className={`w-full bg-black rounded-2xl border border-gray-800 overflow-hidden ${format === '16:9' ? 'aspect-video' : 'aspect-[9/16] max-w-[420px] mx-auto'}`}>
                  {videoUrl ? (
                    <video controls className="w-full h-full object-contain">
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

              <div className="lg:col-span-5 flex flex-col gap-4">
                <div className="glass p-5 rounded-[2rem] border-blue-500/10 flex flex-col gap-4 shadow-2xl">
                  <h3 className="font-black text-xs text-blue-500 uppercase tracking-[0.3em]">Status de Produção</h3>
                  <div className="space-y-3 text-[11px] font-mono">
                    <div className="flex justify-between"><span className="text-gray-500">TEMA:</span> <span className="text-white">{settings.theme}</span></div>
                    <div className="flex justify-between"><span className="text-gray-500">NODES:</span> <span className="text-white">{script.scenes.length}</span></div>
                    <div className="flex justify-between"><span className="text-gray-500">LOCAL:</span> <span className="text-green-500">{script.scenes.filter(s => s.localImage).length} ASSETS</span></div>
                    <div className="flex justify-between"><span className="text-gray-500">DURAÇÃO:</span> <span className="text-white">{videoLength === '1m' ? '~1 min' : '~5 min'} (máx {maxScenes})</span></div>
                    <div className="flex justify-between"><span className="text-gray-500">ÁUDIO FUNDO:</span> <span className={settings.localBackgroundMusic ? "text-green-400" : "text-red-400"}>{settings.localBackgroundMusic ? "Local" : "Não definido"}</span></div>
                  </div>

                  <div className="pt-4 border-t border-gray-800 text-[11px] space-y-3">
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
                          title="Dias mÃ¡ximos para manter assets"
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

                <div className="pt-4 border-t border-gray-800">
                  <button
                    onClick={generateFullVideo}
                    disabled={isGeneratingVideo}
                    className="w-full bg-blue-600 hover:bg-blue-500 py-4 rounded-[1.4rem] font-black text-xs uppercase tracking-[0.22em] transition-all active:scale-[0.98] shadow-xl shadow-blue-900/30"
                  >
                    {isGeneratingVideo ? <Loader /> : "Processar Projeto"}
                  </button>
                  <button
                    onClick={testRenderPipeline}
                    disabled={isGeneratingVideo || isTestingRender}
                    className="w-full mt-3 border border-blue-500/40 text-blue-200 hover:text-white hover:border-blue-300 px-4 py-3 rounded-[1.4rem] font-black text-xs uppercase tracking-[0.22em] transition-all"
                  >
                    {isTestingRender ? <Loader /> : "Forçar teste (ver logs)"}
                  </button>
                </div>
              </div>
            </div>
            </div>
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
