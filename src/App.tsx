/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState, useEffect, useRef } from 'react';
import { 
  Wifi, 
  Settings, 
  Activity, 
  Play, 
  Square, 
  Crosshair, 
  Radio,
  Eye,
  Camera,
  ServerCrash,
  Video
} from 'lucide-react';
import '@tensorflow/tfjs-backend-webgl';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import '@tensorflow/tfjs';

interface Target {
  id: string;
  type: string;
  x: number; // percentage
  y: number; // percentage
  width?: number; // percentage
  height?: number; // percentage
  speed: number;
  altitude: number;
  threatLevel: 'LOW' | 'MEDIUM' | 'HIGH';
}

export default function App() {
  const [connectionMode, setConnectionMode] = useState<'sim' | 'webcam' | 'mjpeg'>('sim');
  const [isConnected, setIsConnected] = useState(false);
  const [isScanning, setIsScanning] = useState(false);
  const [showWifiHelp, setShowWifiHelp] = useState(false);
  
  const [cameras, setCameras] = useState<{ip: string, port: string}[]>(Array(6).fill(null).map((_, i) => ({
    ip: i === 0 ? '192.168.1.100' : `192.168.1.10${i}`,
    port: '8080'
  })));
  const [activeCamIndex, setActiveCamIndex] = useState(0);

  const activeCam = cameras[activeCamIndex];
  const streamUrl = `http://${activeCam.ip}:${activeCam.port}/stream`;

  const updateCamera = (index: number, field: 'ip' | 'port', value: string) => {
    setCameras(prev => prev.map((cam, i) => i === index ? { ...cam, [field]: value } : cam));
  };
  
  const [targets, setTargets] = useState<Target[]>([]);
  const [logs, setLogs] = useState<{time: string, msg: string, type: 'info'|'warn'|'alert'}[]>([]);
  const [modelLoading, setModelLoading] = useState(false);

  const videoRef = useRef<HTMLVideoElement>(null);
  const modelRef = useRef<cocoSsd.ObjectDetection | null>(null);
  const requestRef = useRef<number>();

  useEffect(() => {
    addLog('System initialized. Select video source and initialize link.', 'info');
  }, []);

  const addLog = (msg: string, type: 'info'|'warn'|'alert') => {
    const time = new Date().toLocaleTimeString('en-US', { hour12: false });
    setLogs(prev => [{ time, msg, type }, ...prev].slice(0, 50));
  };

  const loadModel = async () => {
    if (modelRef.current) return;
    setModelLoading(true);
    addLog('Loading neural network model (COCO-SSD)...', 'info');
    try {
      modelRef.current = await cocoSsd.load({ base: 'lite_mobilenet_v2' });
      addLog('Neural network loaded successfully.', 'info');
    } catch (e) {
      addLog('Failed to load neural network.', 'alert');
      console.error(e);
    }
    setModelLoading(false);
  };

  const toggleConnection = async () => {
    if (isConnected) {
      if (videoRef.current?.srcObject) {
        const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
        tracks.forEach(track => track.stop());
        videoRef.current.srcObject = null;
      }
      setIsConnected(false);
      setIsScanning(false);
      addLog('Disconnected from stream.', 'warn');
      setTargets([]);
    } else {
      if (connectionMode === 'webcam') {
        try {
          const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { facingMode: { ideal: "environment" } } 
          });
          if (videoRef.current) {
            videoRef.current.srcObject = stream;
            videoRef.current.play();
          }
          setIsConnected(true);
          addLog('Local camera stream connected.', 'info');
          loadModel();
        } catch (e) {
          addLog('Failed to access local camera.', 'alert');
        }
      } else {
        setIsConnected(true);
        addLog(`Connected to stream at: ${activeCam.ip}:${activeCam.port}`, 'info');
        if (connectionMode === 'sim') {
           // Sim needs no extra load
        }
      }
    }
  };

  const toggleScanning = () => {
    if (!isConnected) return;
    if (isScanning) {
      setIsScanning(false);
      addLog('Sky scanning suspended.', 'warn');
      setTargets([]);
    } else {
      setIsScanning(true);
      addLog('Optical sky tracking engaged.', 'info');
    }
  };

  // Webcam object tracking loop
  useEffect(() => {
    let active = true;
    
    const track = async () => {
      if (!active) return;
      if (videoRef.current && modelRef.current && videoRef.current.readyState === 4) {
        try {
          const predictions = await modelRef.current.detect(videoRef.current);
          
          const newTargets = predictions.filter(p => p.score > 0.4).map((pred) => {
            const videoWidth = videoRef.current!.videoWidth || 640;
            const videoHeight = videoRef.current!.videoHeight || 480;
            
            const [x, y, width, height] = pred.bbox;
            const centerX = x + width / 2;
            const centerY = y + height / 2;

            const xPct = (centerX / videoWidth) * 100;
            const yPct = (centerY / videoHeight) * 100;
            const wPct = (width / videoWidth) * 100;
            const hPct = (height / videoHeight) * 100;

            const isPerson = pred.class === 'person';
            const typeStr = isPerson ? 'PERSON' : pred.class.toUpperCase();
            const threat = isPerson ? 'HIGH' : ['AIRPLANE', 'BIRD'].includes(typeStr) ? 'MEDIUM' : 'LOW';

            return {
              id: `TRK-${pred.class.substring(0,3).toUpperCase()}-${Math.floor(pred.score * 100)}`,
              type: typeStr,
              x: xPct,
              y: yPct,
              width: wPct,
              height: hPct,
              speed: Math.floor(Math.random() * 50 + 10),
              altitude: Math.floor(Math.random() * 100 + 5),
              threatLevel: threat as any
            };
          });

          if (active) {
            setTargets(newTargets);
            const hasHighThreat = newTargets.some(t => t.threatLevel === 'HIGH');
            if (hasHighThreat && Math.random() > 0.95) {
               addLog(`HIGH THREAT: Target in view`, 'alert');
            }
          }
        } catch (err) {
          console.error("Detection error:", err);
        }
      }
      
      if (active) {
        requestRef.current = requestAnimationFrame(track);
      }
    };

    if (isScanning && connectionMode === 'webcam' && isConnected) {
      track();
    }

    return () => {
      active = false;
      if (requestRef.current) cancelAnimationFrame(requestRef.current);
    };
  }, [isScanning, isConnected, connectionMode]);

  // Simulation effect for 'sim' mode
  useEffect(() => {
    let interval: ReturnType<typeof setInterval>;
    if (isScanning && isConnected && connectionMode === 'sim') {
      interval = setInterval(() => {
        if (Math.random() > 0.4) {
          const rand = Math.random();
          const simType = rand > 0.7 ? 'PERSON' : rand > 0.4 ? 'PLANE' : rand > 0.2 ? 'BIRD' : 'OTHER';
          const threat = simType === 'PERSON' ? 'HIGH' : ['PLANE', 'BIRD'].includes(simType) ? 'MEDIUM' : 'LOW';
          const newTarget: Target = {
            id: `TRK-${Math.floor(Math.random() * 9000) + 1000}`,
            type: simType,
            x: Math.random() * 80 + 10,
            y: Math.random() * 60 + 10, // Mostly upper sky
            width: 10,
            height: 10,
            speed: Math.floor(Math.random() * 120 + 10),
            altitude: Math.floor(Math.random() * 1000 + 50),
            threatLevel: threat
          };
          
          setTargets(prev => {
            const kept = prev.filter(() => Math.random() > 0.3).slice(0, 2);
            return [...kept, newTarget];
          });
        } else if (targets.length > 0 && Math.random() > 0.7) {
          setTargets([]); // clear targets sometimes
        }
      }, 1500);
    }
    return () => clearInterval(interval);
  }, [isScanning, isConnected, connectionMode, targets.length]);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 font-sans selection:bg-cyan-500/30">
      {/* Header */}
      <header className="flex items-center justify-between px-6 py-4 border-b border-cyan-900/40 bg-slate-900/50 backdrop-blur-md">
        <div className="flex items-center gap-3">
          <Activity className="text-cyan-500 w-6 h-6 animate-pulse" />
          <h1 className="text-xl font-bold tracking-tight text-cyan-50">SkyWatch Tracker</h1>
        </div>
        <div className="flex items-center gap-6">
          <div className={`flex items-center gap-2 ${isConnected ? 'text-emerald-400' : 'text-slate-500'}`}>
            <Wifi className="w-5 h-5" />
            <span className="text-sm font-medium">{isConnected ? 'Uplink Established' : 'Offline'}</span>
          </div>
          <div className="flex items-center gap-2 text-slate-300">
            <Radio className="w-5 h-5 text-cyan-400 animate-pulse" />
            <span className="text-sm font-medium">Auto-Acoustic ON</span>
          </div>
        </div>
      </header>

      <main className="max-w-screen-2xl mx-auto p-6 grid grid-cols-1 lg:grid-cols-4 gap-6">
        
        {/* Left Panel: Controls */}
        <div className="col-span-1 lg:col-span-1 space-y-6 flex flex-col">
          
          {/* Camera Settings */}
          <div className="bg-slate-900 border border-cyan-900/40 rounded-xl p-5 shadow-[0_0_15px_rgba(6,182,212,0.05)]">
            <h2 className="text-sm font-semibold mb-4 flex items-center gap-2 text-cyan-400 uppercase tracking-wider">
              <Camera className="w-4 h-4" />
              Video Source
            </h2>
            
            <div className="space-y-4">
              <div className="grid grid-cols-3 gap-2">
                <button 
                  disabled={isConnected}
                  onClick={() => setConnectionMode('sim')}
                  className={`text-xs py-2 rounded-lg border transition-colors ${connectionMode === 'sim' ? 'bg-cyan-900/50 border-cyan-500 text-cyan-200' : 'bg-slate-950 border-slate-800 text-slate-400'}`}
                >SIM</button>
                <button 
                  disabled={isConnected}
                  onClick={() => setConnectionMode('webcam')}
                  className={`text-xs py-2 rounded-lg border transition-colors ${connectionMode === 'webcam' ? 'bg-cyan-900/50 border-cyan-500 text-cyan-200' : 'bg-slate-950 border-slate-800 text-slate-400'}`}
                >WEBCAM</button>
                <button 
                  disabled={isConnected}
                  onClick={() => setConnectionMode('mjpeg')}
                  className={`text-xs py-2 rounded-lg border transition-colors ${connectionMode === 'mjpeg' ? 'bg-cyan-900/50 border-cyan-500 text-cyan-200' : 'bg-slate-950 border-slate-800 text-slate-400'}`}
                >MJPEG</button>
              </div>

              {(connectionMode === 'sim' || connectionMode === 'mjpeg') && (
                <div className="relative">
                  <div className="flex gap-1 mb-3 bg-slate-950 p-1 rounded-lg border border-slate-800">
                    {cameras.map((_, i) => (
                      <button
                        key={i}
                        onClick={() => {
                          setActiveCamIndex(i);
                          if (isConnected) {
                            addLog(`Switched feed to CAM ${i + 1}`, 'info');
                          }
                        }}
                        className={`flex-1 py-1 text-xs font-bold rounded transition-colors ${activeCamIndex === i ? 'bg-cyan-600 text-white shadow-[0_0_10px_rgba(8,145,178,0.5)]' : 'text-slate-500 hover:text-slate-300 hover:bg-slate-900'}`}
                      >
                        {i + 1}
                      </button>
                    ))}
                  </div>

                  <div className="flex items-center justify-between mb-1">
                    <label className="text-xs text-slate-400 block">IP Address & Port</label>
                    <button 
                      onClick={() => setShowWifiHelp(!showWifiHelp)}
                      className="text-xs text-cyan-400 hover:text-cyan-300 underline"
                    >
                      Как подключить Wi-Fi камеру?
                    </button>
                  </div>
                  <div className="grid grid-cols-4 gap-2">
                    <input 
                      type="text" 
                      value={activeCam.ip}
                      onChange={(e) => updateCamera(activeCamIndex, 'ip', e.target.value)}
                      disabled={isConnected}
                      placeholder="192.168.1.100"
                      className="col-span-3 w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-sm text-cyan-100 placeholder-slate-600 disabled:opacity-50 focus:border-cyan-500 focus:outline-none"
                    />
                    <input 
                      type="text" 
                      value={activeCam.port}
                      onChange={(e) => updateCamera(activeCamIndex, 'port', e.target.value)}
                      disabled={isConnected}
                      placeholder="80"
                      className="col-span-1 w-full text-center bg-slate-950 border border-slate-700 rounded-lg px-2 py-2 text-sm text-cyan-100 placeholder-slate-600 disabled:opacity-50 focus:border-cyan-500 focus:outline-none"
                    />
                  </div>
                  
                  {showWifiHelp && (
                    <div className="mt-3 p-3 bg-cyan-950/40 border border-cyan-800/50 rounded-lg text-xs text-slate-300 space-y-2">
                      <p className="font-bold text-red-400">Внимание: Веб-сайты не могут автоматически подключаться к Wi-Fi сетям (это запрещено политикой безопасности браузеров).</p>
                      <p className="text-cyan-400 font-medium">Как подключиться напрямую:</p>
                      <ol className="list-decimal pl-4 space-y-1">
                        <li>Зайдите в <strong>настройки Wi-Fi</strong> вашего устройства.</li>
                        <li>Вручную подключитесь к Wi-Fi сети вашей камеры/дрона.</li>
                        <li>Узнайте локальный IP-адрес видеопотока камеры (часто это <code className="bg-black/50 px-1 rounded text-amber-300">192.168.4.1</code> и порт <code className="bg-black/50 px-1 rounded text-amber-300">81</code>).</li>
                        <li>Вернитесь на этот сайт, вставьте IP и порт в поля выше и выберите режим <strong>MJPEG</strong>.</li>
                      </ol>
                      <p className="border-t border-cyan-800/50 pt-2 mt-2">
                        <strong className="text-amber-400">Блокировка видео:</strong> Так как этот сайт работает по защищенному HTTPS, браузер скроет видео с локальной (HTTP) камеры. <br/>
                        <strong>Решение:</strong> Нажмите на иконку "Замочка" слева от адреса сайта ➡️ Настройки сайта (Site settings) ➡️ "Небезопасный контент" (Insecure content) ➡️ <strong>Разрешить (Allow)</strong> ➡️ Обновите страницу.
                      </p>
                    </div>
                  )}
                  {connectionMode === 'mjpeg' && !showWifiHelp && (
                    <p className="text-[10px] text-slate-500 mt-2 leading-tight">
                      Обратите внимание: Для локального HTTP потока камеры нужно разрешить "Insecure Content" (Небезопасный контент) кликнув на иконку замочка рядом с адресом сайта.
                    </p>
                  )}
                </div>
              )}

              <button 
                onClick={toggleConnection}
                className={`w-full py-2.5 mt-2 rounded-lg flex items-center justify-center gap-2 font-medium text-sm transition-all ${
                  isConnected 
                    ? 'bg-red-500/10 text-red-500 hover:bg-red-500/20 border border-red-500/50' 
                    : 'bg-cyan-500/20 text-cyan-400 hover:bg-cyan-500/30 border border-cyan-500/50'
                }`}
              >
                {isConnected ? 'DISCONNECT' : 'INITIALIZE LINK'}
              </button>
            </div>
          </div>

          {/* Tracking Controls */}
          {isConnected && (
            <div className="bg-slate-900 border border-cyan-900/40 rounded-xl p-5 shadow-[0_0_15px_rgba(6,182,212,0.05)] animate-in fade-in slide-in-from-top-4 duration-500">
              <h2 className="text-sm font-semibold mb-4 flex items-center gap-2 text-cyan-400 uppercase tracking-wider">
                <Crosshair className="w-4 h-4" />
                Object Tracking
              </h2>
              
              <button 
                onClick={toggleScanning}
                disabled={modelLoading}
                className={`w-full py-4 rounded-xl flex items-center justify-center gap-2 font-bold transition-all shadow-lg ${
                  modelLoading ? 'bg-slate-800 text-slate-500 cursor-not-allowed' :
                  isScanning 
                    ? 'bg-amber-500/10 text-amber-500 hover:bg-amber-500/20 border border-amber-500/50' 
                    : 'bg-emerald-500 text-slate-950 hover:bg-emerald-400 border border-emerald-400 shadow-[0_0_20px_rgba(16,185,129,0.2)]'
                }`}
              >
                {modelLoading ? (
                  <><Activity className="w-5 h-5 animate-spin" /> LOADING MODEL</>
                ) : isScanning ? (
                  <><Square className="w-5 h-5" fill="currentColor" /> SUSPEND SCAN</>
                ) : (
                  <><Play className="w-5 h-5" fill="currentColor" /> ENGAGE SCANNER</>
                )}
              </button>

              <div className="space-y-4 mt-6 pt-4 border-t border-slate-800">
                <div className="space-y-2">
                  <div className="flex justify-between text-xs">
                    <span className="text-slate-400">Model Framework</span>
                    <span className="font-mono text-cyan-400">TF.js WebGL</span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Target List Snapshot */}
           {isScanning && (
            <div className="bg-slate-900 border border-cyan-900/40 rounded-xl p-5 shadow-[0_0_15px_rgba(6,182,212,0.05)] animate-in fade-in flex-1 overflow-auto max-h-64">
              <h2 className="text-sm font-semibold mb-4 flex items-center gap-2 text-cyan-400 uppercase tracking-wider">
                <Activity className="w-4 h-4" />
                Active Targets ({targets.length})
              </h2>
              {targets.length === 0 ? (
                <p className="text-xs text-slate-500 italic text-center py-4">No targets currently tracked.</p>
              ) : (
                <div className="space-y-3">
                  {targets.map((t, idx) => (
                    <div key={idx} className={`p-3 rounded border text-xs ${
                      t.threatLevel === 'HIGH' ? 'bg-red-500/10 border-red-500/30' : 
                      t.threatLevel === 'MEDIUM' ? 'bg-amber-500/10 border-amber-500/30' : 'bg-slate-950 border-slate-800'
                    }`}>
                      <div className="flex justify-between font-mono mb-1">
                        <span className={t.threatLevel === 'HIGH' ? 'text-red-400' : 'text-cyan-400'}>{t.id}</span>
                        <span className="text-slate-400">{t.type}</span>
                      </div>
                      <div className="flex justify-between text-slate-500 mt-2">
                        <span>ALT: {t.altitude}m</span>
                        <span>SPD: {t.speed} kph</span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Center: Main Camera Feed */}
        <div className="col-span-1 lg:col-span-3 space-y-6 flex flex-col">
          <div className="relative bg-black rounded-2xl overflow-hidden border border-slate-800 shadow-2xl flex-1 min-h-[500px]">
            {/* Background / Placeholder */}
            <div className={`absolute inset-0 transition-opacity duration-1000 ${isConnected && connectionMode === 'sim' ? 'opacity-100' : 'opacity-0'}`}>
               <div className="w-full h-full bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-slate-800 via-slate-900 to-black"></div>
               {/* Noise Overlay */}
               <div className="absolute inset-0 opacity-[0.03] pointer-events-none bg-[url('https://upload.wikimedia.org/wikipedia/commons/7/76/1k_Dissolve_Noise_Texture.png')]"></div>
               <div className="absolute inset-0 bg-blue-950/20 pointer-events-none"></div>
            </div>

            {/* Video Streams */}
            <video 
              ref={videoRef}
              className={`w-full h-full object-cover transition-opacity duration-1000 ${connectionMode === 'webcam' && isConnected ? 'opacity-100' : 'opacity-0 absolute pointer-events-none'}`} 
              playsInline 
              muted 
            />
            {connectionMode === 'mjpeg' && isConnected && (
              <img 
                src={streamUrl} 
                className="w-full h-full object-cover" 
                alt="MJPEG Stream" 
                onError={(e) => {
                  (e.target as HTMLImageElement).src = 'data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxMDAlIiBoZWlnaHQ9IjEwMCUiPjxyZWN0IHdpZHRoPSIxMDAlIiBoZWlnaHQ9IjEwMCUiIGZpbGw9IiMwMDAiLz48dGV4dCB4PSI1MCUiIHk9IjUwJSIgZm9udC1zaXplPSIyMCIgZmlsbD0icmVkIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBkeT0iLjNlbSI+TVBFRyBTdHJlYW0gT2ZmbGluZTwvdGV4dD48L3N2Zz4=';
                }}
              />
            )}

            {!isConnected ? (
              <div className="absolute inset-0 flex flex-col items-center justify-center p-6 text-center z-10 bg-black/80">
                <Video className="w-16 h-16 text-slate-700 mb-4" />
                <h3 className="text-lg font-medium text-slate-300 mb-2">TARGET FEED OFFLINE</h3>
                <p className="text-sm text-slate-500 max-w-sm">Select source (SIM / WEBCAM / MJPEG) and initialize the link.</p>
              </div>
            ) : (
              <>
                {/* Tracking Overlays */}
                {targets.map((target, idx) => (
                  <div 
                    key={idx}
                    className="absolute transition-all duration-75 ease-linear pointer-events-none"
                    style={{ 
                      left: `${target.x}%`, 
                      top: `${target.y}%`, 
                      transform: 'translate(-50%, -50%)',
                      width: target.width ? `${target.width}%` : '80px',
                      height: target.height ? `${target.height}%` : '80px',
                    }}
                  >
                    {/* Bounding Box */}
                    <div className={`w-full h-full border-2 border-dashed ${
                      target.threatLevel === 'HIGH' ? 'border-red-500' :
                      target.threatLevel === 'MEDIUM' ? 'border-amber-500' :
                      'border-cyan-500'
                    }`}>
                      {/* Corner Accents */}
                      <div className="absolute -top-1 -left-1 w-3 h-3 border-t-2 border-l-2 border-current"></div>
                      <div className="absolute -top-1 -right-1 w-3 h-3 border-t-2 border-r-2 border-current"></div>
                      <div className="absolute -bottom-1 -left-1 w-3 h-3 border-b-2 border-l-2 border-current"></div>
                      <div className="absolute -bottom-1 -right-1 w-3 h-3 border-b-2 border-r-2 border-current"></div>
                    </div>
                    {/* Label */}
                    <div className={`absolute top-full left-0 mt-2 whitespace-nowrap text-xs font-mono px-2 py-0.5 rounded backdrop-blur bg-black/50 ${
                      target.threatLevel === 'HIGH' ? 'text-red-400 border border-red-500/50' : 
                      target.threatLevel === 'MEDIUM' ? 'text-amber-400 border border-amber-500/50' : 
                      'text-cyan-400 border border-cyan-500/50'
                    }`}>
                      {target.id} | {target.type} 
                    </div>
                  </div>
                ))}

                {/* Scope Reticle / Center */}
                <div className="absolute inset-0 pointer-events-none flex items-center justify-center">
                  <div className="w-[40vw] h-[40vw] max-w-[400px] max-h-[400px] border border-cyan-500/20 rounded-full flex items-center justify-center">
                    <div className="w-1 h-3 bg-cyan-500/50 absolute top-0"></div>
                    <div className="w-1 h-3 bg-cyan-500/50 absolute bottom-0"></div>
                    <div className="w-3 h-1 bg-cyan-500/50 absolute left-0"></div>
                    <div className="w-3 h-1 bg-cyan-500/50 absolute right-0"></div>
                    <div className="w-1 h-1 bg-cyan-500/80 rounded-full"></div>
                  </div>
                </div>

                {/* OSD (On Screen Display) Text */}
                <div className="absolute inset-0 p-6 flex flex-col justify-between pointer-events-none">
                  <div className="flex justify-between items-start font-mono text-sm drop-shadow-md">
                    <div className={isScanning ? 'text-emerald-400 animate-pulse font-bold bg-black/40 px-2 py-1 rounded' : 'text-slate-400 bg-black/40 px-2 py-1 rounded'}>
                      {isScanning ? 'SCANNING ACTIVE' : 'SCANNING SUSPENDED'}
                    </div>
                    <div className="text-right text-slate-300 bg-black/40 px-2 py-1 rounded">
                      FOV: 90°<br/>
                      SRC: {connectionMode.toUpperCase()}
                    </div>
                  </div>
                  
                  <div className="flex justify-between items-end font-mono text-sm text-cyan-200 drop-shadow-[0_1px_2px_rgba(0,0,0,0.8)]">
                    <div className="bg-black/40 px-2 py-1 rounded">
                      LAT: 32.41<br/>
                      LON: -110.23
                    </div>
                    <div className="text-right bg-black/40 px-2 py-1 rounded">
                      {new Date().toISOString().split('T')[1].substring(0, 8)} ZULU
                    </div>
                  </div>
                </div>
              </>
            )}
          </div>

          {/* Event Logs */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 shadow-inner h-48 flex flex-col">
            <h3 className="text-xs font-bold text-slate-500 tracking-wider mb-2 flex items-center gap-2 shrink-0">
               <Eye className="w-4 h-4" />
               SYSTEM EVENT LOG
            </h3>
            <div className="overflow-y-auto space-y-1.5 flex-1 pr-2 custom-scrollbar">
              {logs.map((log, i) => (
                <div key={i} className={`font-mono text-xs flex gap-3 animate-in fade-in slide-in-from-left-2 ${
                  log.type === 'alert' ? 'text-red-400 bg-red-950/30 p-1 rounded -mx-1 px-2' : 
                  log.type === 'warn' ? 'text-amber-400' : 
                  'text-slate-400'
                }`}>
                  <span className="opacity-50 shrink-0">[{log.time}]</span>
                  <span className={log.type === 'alert' ? 'font-bold' : ''}>{log.msg}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

      </main>

      {/* Helper styles */}
      <style>{`
        .custom-scrollbar::-webkit-scrollbar {
          width: 6px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: rgba(15, 23, 42, 0.5);
          border-radius: 4px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: rgba(30, 41, 59, 1);
          border-radius: 4px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: rgba(51, 65, 85, 1);
        }
      `}</style>
    </div>
  );
}


// Custom lucide icon not perfectly matching, but replacing Radar with a generic SVG if needed since lucide-react might not have full set depending on version. We'll use Activity and Eye mostly.
function Radar(props: any) {
  return (
    <svg
      {...props}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M19.07 4.93A10 10 0 0 0 6.99 3.34" />
      <path d="M4 6h.01" />
      <path d="M2.29 9.62A10 10 0 1 0 21.31 8.35" />
      <path d="M16.24 7.76A6 6 0 1 0 8.23 16.67" />
      <path d="M12 18h.01" />
      <path d="M17.99 11.66A6 6 0 0 1 15.77 16.67" />
      <circle cx="12" cy="12" r="2" />
      <path d="m13.41 10.59 5.66-5.66" />
    </svg>
  )
}
