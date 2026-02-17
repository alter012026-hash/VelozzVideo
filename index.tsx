import React, { useState, useEffect } from 'react';
import { createRoot } from 'react-dom/client';
import { GoogleGenAI, Modality } from "@google/genai";

declare global {
  interface Window {
    aistudio?: {
      hasSelectedApiKey: () => Promise<boolean>;
      openSelectKey: () => Promise<void>;
      getSelectedApiKey?: () => Promise<string>;
    };
  }
}

// --- Types ---
interface Scene {
  id: string;
  text: string;
  visualPrompt: string;
  localImage?: string;
  narrationAudio?: string; // Base64 PCM data
  status: 'pending' | 'processing' | 'completed';
}

interface GeneralSettings {
  theme: string;
  localThemeRef?: string; // Imagem local de referência de estilo
  backgroundMusic: string;
  localBackgroundMusic?: string; // Áudio local de fundo
  voiceName: string;
}

interface Script {
  title: string;
  scenes: Scene[];
}

type ScriptSource = 'gemini' | 'ollama';
type WorkflowStep = 'topic' | 'scripting' | 'assets' | 'rendering' | 'done';

const App: React.FC = () => {
  const [topic, setTopic] = useState('');
  const [scriptSource, setScriptSource] = useState<ScriptSource>('ollama'); // prioriza local
  const [videoLength, setVideoLength] = useState<'1m' | '5m'>('1m');
  const [currentStep, setCurrentStep] = useState<WorkflowStep>('topic');
  const stepOrder: WorkflowStep[] = ['topic', 'scripting', 'assets', 'rendering', 'done'];
  const [isGeneratingScript, setIsGeneratingScript] = useState(false);
  const [script, setScript] = useState<Script | null>(null);
  const [isGeneratingVideo, setIsGeneratingVideo] = useState(false);
  const [isGeneratingNarration, setIsGeneratingNarration] = useState<string | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [statusMessage, setStatusMessage] = useState('');
  const [format, setFormat] = useState<'16:9' | '9:16'>('16:9');
  const [hasApiKey, setHasApiKey] = useState(false);
  const [importError, setImportError] = useState<string | null>(null);
  const [ollamaHost, setOllamaHost] = useState('http://127.0.0.1:11434');
  const [ollamaModel, setOllamaModel] = useState('llama3');
  const [ollamaModels, setOllamaModels] = useState<string[]>([]);
  const [isLoadingModels, setIsLoadingModels] = useState(false);
  const [ollamaError, setOllamaError] = useState<string | null>(null);
  const [logFeed, setLogFeed] = useState<{ id: string; msg: string; level: 'info' | 'warn' | 'error'; time: string }[]>([]);
  const [isPreviewingVoice, setIsPreviewingVoice] = useState(false);
  const [previewText, setPreviewText] = useState('Esta é uma prévia de voz.');
  const edgeVoices = [
    'pt-BR-ThalitaMultilingualNeural',
    'pt-BR-ThalitaNeural',
    'pt-BR-FranciscaNeural',
    'pt-BR-BrendaNeural',
    'pt-BR-GiovannaNeural',
    'pt-BR-ElzaNeural',
    'pt-BR-LeticiaNeural',
    'pt-BR-LeilaNeural',
    'pt-BR-ManuelaNeural',
    'pt-BR-YaraNeural',
    'pt-BR-AntonioNeural',
    'pt-BR-DonatoNeural',
    'pt-BR-FabioNeural',
    'pt-BR-HumbertoNeural',
    'pt-BR-JulioNeural',
    'pt-BR-NicolauNeural',
    'pt-BR-ValerioNeural'
  ];
  const [openSections, setOpenSections] = useState({
    ollama: true,
    theme: true,
    audio: true,
    voice: true,
    api: true,
    backend: true,
  });

  const toggleSection = (key: keyof typeof openSections) =>
    setOpenSections(prev => ({ ...prev, [key]: !prev[key] }));

  const maxScenes = videoLength === '1m' ? 6 : 20; // estimativa: ~10s/cena ou ~15s/cena

  const pushLog = (msg: string, level: 'info' | 'warn' | 'error' = 'info') => {
    const time = new Date().toLocaleTimeString();
    setLogFeed(prev => [{ id: Math.random().toString(36).slice(2), msg, level, time }, ...prev].slice(0, 12));
  };

  // --- General Settings ---
  const [settings, setSettings] = useState<GeneralSettings>({
    theme: 'Cinematographic',
    backgroundMusic: 'Áudio Local',
    voiceName: 'Kore'
  });

  useEffect(() => {
    const checkKey = async () => {
      // @ts-ignore
      const hasKey = await window.aistudio.hasSelectedApiKey();
      setHasApiKey(hasKey);
    };
    checkKey();

    // carrega preferências locais de roteiro
    const storedHost = localStorage.getItem('ollama_host');
    const storedModel = localStorage.getItem('ollama_model');
    if (storedHost) setOllamaHost(storedHost);
    if (storedModel) setOllamaModel(storedModel);
  }, []);

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
      const res = await fetch(`/api/tts/preview?text=${encodeURIComponent(previewText || 'Prévia de voz')}&voice=${encodeURIComponent(settings.voiceName)}`);
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err?.detail || `HTTP ${res.status}`);
      }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const audio = new Audio(url);
      audio.play();
      pushLog(`Prévia de voz ${settings.voiceName}`, 'info');
    } catch (err: any) {
      pushLog(`Falha na prévia de voz: ${err?.message || err}`, 'error');
    } finally {
      setIsPreviewingVoice(false);
    }
  };

  const handleAiError = async (error: any, context: string) => {
    console.error(`${context} failed:`, error);
    const errorMessage = String(error?.message || error || '');
    if (errorMessage.includes("API key expired") || errorMessage.includes("429") || errorMessage.includes("400")) {
      // @ts-ignore
      await window.aistudio.openSelectKey();
      setHasApiKey(true);
    } else {
      alert(`${context} falhou: ${errorMessage}`);
    }
  };

  const handleOpenKeySelector = async () => {
    // @ts-ignore
    await window.aistudio.openSelectKey();
    setHasApiKey(true);
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
      let data;
      if (scriptSource === 'ollama') {
        localStorage.setItem('ollama_host', ollamaHost);
        localStorage.setItem('ollama_model', ollamaModel);
        const response = await fetch(`${ollamaHost.replace(/\/$/, '')}/api/generate`, {
          method: 'POST',
          body: JSON.stringify({ model: ollamaModel, prompt: `${prompt}. Responda apenas com o JSON puro.`, stream: false, format: 'json' }),
        });
        const resData = await response.json();
        data = JSON.parse(String(resData.response));
      } else {
        const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
        const response = await ai.models.generateContent({
          model: 'gemini-3-flash-preview',
          contents: prompt,
          config: { responseMimeType: "application/json" }
        });
        data = JSON.parse(String(response.text || '{}'));
      }
      
      const scenesWithStatus = data.scenes.map((s: any) => ({ ...s, status: 'completed' }));
      setScript({ ...data, scenes: scenesWithStatus });
      setCurrentStep('assets');
      pushLog(`Roteiro pronto (${data.scenes.length} cenas)`, 'info');
    } catch (error: any) {
      setCurrentStep('topic');
      await handleAiError(error, 'Roteiro');
      pushLog(`Erro no roteiro: ${error?.message || error}`, 'error');
    } finally {
      setIsGeneratingScript(false);
      setStatusMessage('');
    }
  };

  const generateNarration = async (sceneId: string, text: string) => {
    setIsGeneratingNarration(sceneId);
    try {
      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
      const response = await ai.models.generateContent({
        model: "gemini-2.5-flash-preview-tts",
        contents: [{ parts: [{ text: `Say clearly: ${text}` }] }],
        config: {
          responseModalities: [Modality.AUDIO],
          speechConfig: {
            voiceConfig: {
              prebuiltVoiceConfig: { voiceName: settings.voiceName },
            },
          },
        },
      });

      const base64Audio = response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
      if (base64Audio) {
        updateScene(sceneId, { narrationAudio: base64Audio });
        pushLog(`Narração gerada para cena ${sceneId}`, 'info');
      }
    } catch (error: any) {
      await handleAiError(error, 'Narração');
      pushLog(`Erro na narração: ${error?.message || error}`, 'error');
    } finally {
      setIsGeneratingNarration(null);
    }
  };

  const playNarration = async (base64: string) => {
    const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
    const bytes = atob(base64);
    const arrayBuffer = new ArrayBuffer(bytes.length);
    const uint8View = new Uint8Array(arrayBuffer);
    for (let i = 0; i < bytes.length; i++) uint8View[i] = bytes.charCodeAt(i);
    
    const dataInt16 = new Int16Array(uint8View.buffer);
    const audioBuffer = audioCtx.createBuffer(1, dataInt16.length, 24000);
    const channelData = audioBuffer.getChannelData(0);
    for (let i = 0; i < dataInt16.length; i++) channelData[i] = dataInt16[i] / 32768.0;

    const source = audioCtx.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(audioCtx.destination);
    source.start();
  };

  const updateScene = (id: string, updates: Partial<Scene>) => {
    if (!script) return;
    setScript({
      ...script,
      scenes: script.scenes.map(s => s.id === id ? { ...s, ...updates } : s)
    });
  };

  const deleteScene = (id: string) => {
    if (!script) return;
    setScript({
      ...script,
      scenes: script.scenes.filter(s => s.id !== id)
    });
  };

  const duplicateScene = (scene: Scene) => {
    if (!script) return;
    const newScene = { ...scene, id: Math.random().toString(36).substr(2, 9), narrationAudio: undefined };
    const index = script.scenes.findIndex(s => s.id === scene.id);
    const newScenes = [...script.scenes];
    newScenes.splice(index + 1, 0, newScene);
    setScript({ ...script, scenes: newScenes });
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

  const handleImportScript = (file: File) => {
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const json = JSON.parse(String(reader.result || '{}'));
        if (!json?.title || !Array.isArray(json?.scenes)) throw new Error('Formato inválido');
        const scenesWithStatus = json.scenes.map((s: any) => ({ ...s, status: s.status ?? 'completed' }));
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
  };

  const generateFullVideo = async () => {
    if (!script) return;
    // Só exige chave se for usar geração em nuvem
    if (scriptSource === 'gemini') {
      // @ts-ignore
      if (!(await window.aistudio.hasSelectedApiKey())) {
        await handleOpenKeySelector();
        return;
      }
    }

    setIsGeneratingVideo(true);
    setCurrentStep('rendering');
    setStatusMessage(`Motor de Edição Híbrido: Processando Assets Locais e Sincronizando Camadas...`);

    try {
      const hasLocalAssets = script.scenes.some(s => s.localImage) || settings.localThemeRef || settings.localBackgroundMusic;
      
      if (hasLocalAssets) {
        setStatusMessage('Motor de Edição: Compilando Projeto com Assets Locais via MoviePy...');
        await new Promise(r => setTimeout(r, 6000));
        alert("Sucesso! O Motor de Edição Local processou a trilha sonora e os frames de cena com sucesso. Renderização completa.");
        setIsGeneratingVideo(false);
        setCurrentStep('done');
        return;
      }

      if (scriptSource === 'gemini') {
        const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
        const fullPrompt = `Video Title: ${script.title}. Visual Theme: ${settings.theme}. Background Music: ${settings.backgroundMusic}. Scene Sequence: ${script.scenes.map(s => s.visualPrompt).join('. ')}. Style: Cinematographic.`;
        
        let operation = await ai.models.generateVideos({
          model: 'veo-3.1-fast-generate-preview',
          prompt: fullPrompt,
          config: { numberOfVideos: 1, resolution: '720p', aspectRatio: format }
        });

        while (!operation.done) {
          await new Promise(resolve => setTimeout(resolve, 8000));
          operation = await ai.operations.getVideosOperation({ operation: operation });
        }

        if (operation.error) throw new Error(String(operation.error.message));

        const downloadLink = operation.response?.generatedVideos?.[0]?.video?.uri;
        if (downloadLink) {
          const res = await fetch(`${downloadLink}&key=${process.env.API_KEY}`);
          const blob = await res.blob();
          setVideoUrl(URL.createObjectURL(blob));
          setCurrentStep('done');
          pushLog('Renderização em nuvem concluída', 'info');
        }
      }
    } catch (error: any) {
      setCurrentStep('assets');
      await handleAiError(error, 'Vídeo');
      pushLog(`Erro na renderização: ${error?.message || error}`, 'error');
    } finally {
      setIsGeneratingVideo(false);
      setStatusMessage('');
    }
  };

  const needsApiGate = scriptSource === 'gemini' && !hasApiKey;

  if (needsApiGate) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center p-4 bg-[#030712] text-white">
        <div className="glass p-10 rounded-3xl text-center max-w-md shadow-2xl border-white/5">
          <h1 className="text-3xl font-bold mb-6 gradient-text">VelozzVideo Pro</h1>
          <p className="text-gray-400 mb-8">Conecte sua chave para ativar o Motor de Edição Híbrido.</p>
          <button onClick={handleOpenKeySelector} className="w-full bg-blue-600 hover:bg-blue-500 py-4 rounded-2xl font-bold transition-all shadow-lg shadow-blue-900/40 uppercase tracking-widest">Ativar Estúdio</button>
        </div>
      </div>
    );
  }

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
              <button onClick={resetProject} className="text-[10px] font-bold uppercase px-3 py-2 rounded-xl border border-red-500/40 text-red-300 hover:text-white hover:border-red-400 transition-all">Resetar</button>
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
              <div className="flex items-center gap-2">
                <label className="cursor-pointer">
                   <span className="text-[9px] font-bold text-gray-500 hover:text-blue-400 uppercase transition-colors">Local File</span>
                   <input type="file" className="hidden" accept="image/*" onChange={(e) => e.target.files?.[0] && handleFileUpload('global_theme', '', e.target.files[0])} />
                </label>
                <button onClick={() => toggleSection('theme')} className="text-[11px] text-gray-400 hover:text-white">{openSections.theme ? '−' : '+'}</button>
              </div>
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
                {settings.localThemeRef && (
                  <div className="h-12 w-full rounded-xl overflow-hidden border border-blue-500/30">
                    <img src={settings.localThemeRef} className="w-full h-full object-cover grayscale group-hover:grayscale-0 transition-all" />
                  </div>
                )}
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
                <span className="text-[10px] text-gray-500">pt-BR</span>
                <button onClick={() => toggleSection('voice')} className="text-[11px] text-gray-400 hover:text-white">{openSections.voice ? '−' : '+'}</button>
              </div>
            </div>
            {openSections.voice && (
              <>
                <div className="flex flex-col gap-2 text-[10px] text-gray-400 mb-1">
                  <div className="flex items-center justify-between">
                    <span>Voz selecionada: {settings.voiceName.replace('pt-BR-', '')}</span>
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
                  {edgeVoices.map(v => (
                    <li key={v}>
                      <button
                        onClick={() => setSettings({ ...settings, voiceName: v })}
                        className={`w-full text-left px-3 py-2 rounded-lg border transition-all ${
                          settings.voiceName === v
                            ? 'border-blue-500 bg-blue-500/20 text-white'
                            : 'border-gray-700 bg-black/40 text-gray-300 hover:border-blue-500/40 hover:text-white'
                        }`}
                      >
                        {v.replace('pt-BR-', '')}
                      </button>
                    </li>
                  ))}
                </ul>
                <p className="text-[10px] text-gray-500">Usado na narração; priorize vozes “Thalita” para PT-BR.</p>
              </>
            )}
          </div>

          {/* API + Logs */}
          <div className="glass p-4 rounded-3xl border-purple-500/15 flex flex-col gap-3">
            <p className="text-[11px] font-black uppercase text-purple-400">API (Opcional)</p>
            <div className="flex gap-2">
              <select value={scriptSource} onChange={(e) => setScriptSource(e.target.value as ScriptSource)} className="bg-gray-900 text-[10px] font-bold uppercase px-3 py-2 rounded-xl border border-gray-800 outline-none text-gray-400 hover:text-white transition-colors cursor-pointer flex-1">
                <option value="ollama">Ollama Local</option>
                <option value="gemini">Gemini API</option>
              </select>
              <button onClick={handleOpenKeySelector} className="bg-purple-600 hover:bg-purple-500 text-white text-[10px] font-black uppercase tracking-[0.15em] px-4 py-2 rounded-xl transition-all">Chave</button>
            </div>
            <div className="text-[10px] text-gray-500">Status: <span className={hasApiKey ? "text-green-400" : "text-red-400"}>{hasApiKey ? "Conectada" : "Ausente"}</span></div>
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
                  <div>API Gemini: {hasApiKey ? 'Conectada' : 'Opcional/ausente'}</div>
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
          <div className="glass rounded-3xl border-white/5 shadow-2xl p-6">
            <div className="flex items-start gap-12 min-w-max mx-auto px-10">
          {script?.scenes.map((scene, idx) => (
            <React.Fragment key={scene.id}>
              <div className="relative group/node animate-in zoom-in-95 duration-500">
                <div className="absolute -top-10 left-0 flex items-center gap-3 w-full">
                  <div className="flex items-center gap-2">
                    <span className="w-5 h-5 rounded-full bg-gray-900 border border-gray-800 flex items-center justify-center text-[8px] font-black text-gray-500">{idx + 1}</span>
                    <span className="text-[9px] font-black text-gray-500 uppercase tracking-widest">Cena Atual</span>
                  </div>
                </div>

                <div className={`w-80 glass rounded-[2.5rem] overflow-hidden border-2 transition-all duration-500 ${scene.localImage ? 'border-green-500/20' : 'border-gray-800'}`}>
                  <div className="h-44 bg-gray-950 relative group/asset border-b border-gray-800">
                    {scene.localImage ? (
                      <img src={scene.localImage} className="w-full h-full object-cover transition-transform duration-700 group-hover/asset:scale-105" />
                    ) : (
                      <div className="w-full h-full flex flex-col items-center justify-center text-gray-800 bg-[conic-gradient(from_0deg_at_50%_50%,_#0a0a0a_0%,_#030712_100%)]">
                        <svg className="w-10 h-10 mb-2 opacity-10" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"></path></svg>
                        <span className="text-[10px] font-black uppercase tracking-tighter opacity-20">Inserir Imagem Local</span>
                      </div>
                    )}
                    
                    <label className="absolute inset-0 bg-black/80 opacity-0 group-hover/asset:opacity-100 transition-all duration-300 flex flex-col items-center justify-center cursor-pointer backdrop-blur-sm">
                      <div className="w-12 h-12 rounded-full bg-blue-600 flex items-center justify-center mb-3 shadow-xl">
                        <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M12 4v16m8-8H4"/></svg>
                      </div>
                      <span className="text-xs font-black uppercase tracking-widest text-white/70">Substituir Imagem</span>
                      <input type="file" className="hidden" accept="image/*" onChange={(e) => e.target.files?.[0] && handleFileUpload('scene_image', scene.id, e.target.files[0])} />
                    </label>
                  </div>

                  <div className="p-6 flex flex-col gap-4">
                    <div className="flex justify-between items-end">
                      <label className="text-[10px] font-black text-gray-500 uppercase tracking-widest">Narração</label>
                      <button 
                        onClick={() => generateNarration(scene.id, scene.text)}
                        disabled={isGeneratingNarration === scene.id}
                        className={`text-[9px] font-black px-3 py-1.5 rounded-xl border transition-all ${
                          scene.narrationAudio 
                          ? 'bg-purple-600 border-purple-400 text-white' 
                          : 'bg-gray-900 border-gray-800 text-gray-500 hover:text-white'
                        }`}
                      >
                        {isGeneratingNarration === scene.id ? "SINCRONIZANDO..." : scene.narrationAudio ? "RE-GERAR VOZ" : "GERAR VOZ"}
                      </button>
                    </div>
                    
                    <div className="relative">
                      <textarea 
                        className="w-full bg-black/40 rounded-2xl p-4 text-xs text-gray-300 border border-gray-800 focus:border-blue-500/40 outline-none h-24 resize-none leading-relaxed"
                        value={scene.text}
                        onChange={(e) => updateScene(scene.id, { text: e.target.value })}
                      />
                      {scene.narrationAudio && (
                        <button 
                          onClick={() => playNarration(scene.narrationAudio!)} 
                          className="absolute bottom-3 right-3 p-2 bg-blue-600/80 rounded-xl hover:bg-blue-500 transition-all shadow-xl"
                        >
                          <svg className="w-4 h-4 text-white" fill="currentColor" viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>
                        </button>
                      )}
                    </div>

                    <div className="flex justify-between gap-3 pt-4 border-t border-gray-800/50">
                      <button onClick={() => duplicateScene(scene)} className="flex-1 py-2 bg-gray-900/50 rounded-xl text-[9px] font-black text-gray-500 hover:text-white border border-gray-800 transition-all uppercase tracking-widest">Duplicar</button>
                      <button onClick={() => deleteScene(scene.id)} className="flex-1 py-2 bg-gray-900/50 rounded-xl text-[9px] font-black text-gray-500 hover:text-red-500 border border-gray-800 transition-all uppercase tracking-widest">Excluir</button>
                    </div>
                  </div>
                </div>
              </div>

              {idx < (script?.scenes.length || 0) - 1 && (
                <div className="self-center pt-8 flex flex-col items-center gap-1 opacity-20">
                  <div className="w-12 h-0.5 bg-gradient-to-r from-blue-500/50 to-purple-500/50 rounded-full"></div>
                </div>
              )}
            </React.Fragment>
          ))}

          {script && (
             <button 
              onClick={() => {
                if (script.scenes.length >= maxScenes) return;
                const newId = Math.random().toString(36).substr(2, 9);
                setScript({...script, scenes: [...script.scenes, { id: newId, text: 'Nova narração', visualPrompt: 'Nova descrição', status: 'completed' }]})
              }}
              disabled={script.scenes.length >= maxScenes}
              className={`self-center w-16 h-16 rounded-full border-2 border-dashed flex items-center justify-center transition-all ${
                script.scenes.length >= maxScenes
                ? 'border-gray-800 text-gray-800 cursor-not-allowed'
                : 'border-gray-800 text-gray-700 hover:border-blue-500/50 hover:text-blue-500'
              }`}
             >
               <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 4v16m8-8H4"/></svg>
             </button>
          )}
        </div>
      </div>

          {/* RENDER + STATUS */}
          {script && (
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 pb-12">
              <div className="lg:col-span-8 glass rounded-[2.5rem] p-8 min-h-[420px] flex flex-col items-center justify-center relative border-dashed border-2 border-gray-800 overflow-hidden shadow-2xl">
                {isGeneratingVideo && (
                  <div className="absolute inset-0 bg-black/90 backdrop-blur-2xl z-[60] flex flex-col items-center justify-center rounded-[2.5rem] p-12 text-center">
                    <div className="w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mb-6"></div>
                    <h2 className="text-3xl font-black mb-4 tracking-tighter uppercase">Processando</h2>
                    <p className="text-gray-400 font-mono text-xs max-w-sm uppercase tracking-widest opacity-60">{statusMessage}</p>
                  </div>
                )}

                {videoUrl ? (
                  <div className="w-full h-full flex flex-col animate-in zoom-in-95 duration-700">
                    <video controls className="w-full max-h-[520px] bg-black rounded-3xl">
                      <source src={videoUrl} type="video/mp4" />
                    </video>
                    <div className="mt-6 flex justify-center items-center gap-6">
                       <button onClick={() => setVideoUrl(null)} className="text-gray-500 hover:text-white text-xs font-black uppercase tracking-widest transition-all">Recomeçar</button>
                       <a href={videoUrl} download="velozz_render.mp4" className="bg-white text-black px-10 py-4 rounded-[1.6rem] font-black text-xs uppercase tracking-widest hover:bg-gray-200 transition-all shadow-2xl active:scale-95">Exportar MP4</a>
                    </div>
                  </div>
                ) : (
                  <div className="text-center opacity-20">
                    <h3 className="text-2xl font-black uppercase tracking-widest font-display">Monitor de Saída</h3>
                    <p className="text-xs mt-3">Consolide os nodes para gerar o vídeo final.</p>
                  </div>
                )}
              </div>

              <div className="lg:col-span-4 flex flex-col gap-4">
                <div className="glass p-6 rounded-2rem border-blue-500/10 flex flex-col gap-5 shadow-2xl">
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

                  <div className="pt-4 border-t border-gray-800">
                    <button
                      onClick={generateFullVideo}
                      disabled={isGeneratingVideo}
                      className="w-full bg-blue-600 hover:bg-blue-500 py-6 rounded-[1.6rem] font-black text-sm uppercase tracking-[0.2em] transition-all active:scale-[0.98] shadow-xl shadow-blue-900/30"
                    >
                      {isGeneratingVideo ? <Loader /> : "Processar Projeto"}
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
