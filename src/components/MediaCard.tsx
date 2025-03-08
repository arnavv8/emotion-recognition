import React, { useRef, useState, useEffect } from 'react';
import { Upload, Mic, Video, StopCircle, Play, Pause, Settings } from 'lucide-react';
import { MediaRecorderProps, PredictionResult } from '../types';

interface Props {
  type: 'audio' | 'video';
  onFileSelect: (file: File, dataset?: string) => void;
  onRecordingComplete: (blob: Blob, dataset?: string) => void;
  onLivePredict?: (frame: ImageData) => Promise<PredictionResult>;
}

export function MediaCard({ type, onFileSelect, onRecordingComplete, onLivePredict }: Props) {
  const [isRecording, setIsRecording] = useState(false);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [livePrediction, setLivePrediction] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedDataset, setSelectedDataset] = useState<string>('cremad');
  const [showSettings, setShowSettings] = useState(false);

  const mediaRecorder = useRef<MediaRecorder | null>(null);
  const chunks = useRef<Blob[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const audioRef = useRef<HTMLAudioElement>(null);
  const animationFrameRef = useRef<number>();

  useEffect(() => {
    return () => {
      cleanupResources();
    };
  }, [stream]);

  const cleanupResources = () => {
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
    }
    if (audioUrl) {
      URL.revokeObjectURL(audioUrl);
    }
  };

  const startRecording = async () => {
    try {
      setError(null);
      const constraints = {
        audio: true,
        video: type === 'video' ? { facingMode: 'user' } : false,
      };
      
      const mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
      
      setStream(mediaStream);
      const recorder = new MediaRecorder(mediaStream);
      
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunks.current.push(e.data);
        }
      };

      recorder.onerror = (e) => {
        setError(`Recording error: ${e.error.message}`);
        stopRecording();
      };

      recorder.onstop = () => {
        try {
          const blob = new Blob(chunks.current, {
            type: type === 'video' ? 'video/webm' : 'audio/webm',
          });
          onRecordingComplete(blob, selectedDataset);
          
          if (type === 'audio') {
            const url = URL.createObjectURL(blob);
            setAudioUrl(url);
          }
          
          chunks.current = [];
        } catch (err) {
          setError('Failed to process recording');
          console.error('Error processing recording:', err);
        }
      };

      mediaRecorder.current = recorder;
      recorder.start();
      setIsRecording(true);

      if (type === 'video' && videoRef.current) {
        videoRef.current.srcObject = mediaStream;
        startVideoFrameCapture();
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Could not access media device';
      setError(`Media access error: ${errorMessage}`);
      console.error('Media access error:', err);
    }
  };

  const stopRecording = () => {
    try {
      if (mediaRecorder.current && mediaRecorder.current.state !== 'inactive') {
        mediaRecorder.current.stop();
      }
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
      setStream(null);
      setIsRecording(false);
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    } catch (err) {
      setError('Failed to stop recording');
      console.error('Error stopping recording:', err);
    }
  };

  const startVideoFrameCapture = () => {
    if (!canvasRef.current || !videoRef.current || !onLivePredict) return;

    const canvas = canvasRef.current;
    const video = videoRef.current;
    const context = canvas.getContext('2d');

    const captureFrame = async () => {
      if (!context || !video) return;

      try {
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        const imageData = context.getImageData(0, 0, canvas.width, canvas.height);
        
        const prediction = await onLivePredict(imageData);
        setLivePrediction(prediction);
      } catch (err) {
        // Don't show frame capture errors to avoid UI clutter
        console.error('Frame capture error:', err);
      }

      if (isRecording) {
        animationFrameRef.current = requestAnimationFrame(captureFrame);
      }
    };

    captureFrame();
  };

  const toggleAudioPlayback = () => {
    if (!audioRef.current) return;
    
    try {
      if (isPlaying) {
        audioRef.current.pause();
      } else {
        audioRef.current.play();
      }
      setIsPlaying(!isPlaying);
    } catch (err) {
      setError('Failed to control audio playback');
      console.error('Audio playback error:', err);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      try {
        onFileSelect(file, selectedDataset);
      } catch (err) {
        setError('Failed to process selected file');
        console.error('File processing error:', err);
      }
    }
  };

  const toggleSettings = () => {
    setShowSettings(!showSettings);
  };

  return (
    <div className="bg-white rounded-lg shadow-lg p-6 w-full max-w-md">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-2xl font-bold text-gray-800">
          {type === 'audio' ? 'Audio Input' : 'Video Input'}
        </h2>
        <button 
          onClick={toggleSettings}
          className="p-2 rounded-full hover:bg-gray-100"
          title="Settings"
        >
          <Settings size={20} />
        </button>
      </div>
      
      {showSettings && (
        <div className="mb-4 p-4 bg-gray-50 rounded-lg">
          <h3 className="font-medium mb-2">Dataset Selection</h3>
          <div className="flex gap-2">
            <label className="flex items-center">
              <input
                type="radio"
                name={`dataset-${type}`}
                value="ravdess"
                checked={selectedDataset === 'ravdess'}
                onChange={() => setSelectedDataset('ravdess')}
                className="mr-2"
              />
              RAVDESS
            </label>
            <label className="flex items-center">
              <input
                type="radio"
                name={`dataset-${type}`}
                value="cremad"
                checked={selectedDataset === 'cremad'}
                onChange={() => setSelectedDataset('cremad')}
                className="mr-2"
              />
              CREMA-D
            </label>
          </div>
          {type === 'video' && selectedDataset === 'ravdess' && (
            <p className="text-yellow-600 text-sm mt-2">
              Note: Video processing is only available for CREMA-D dataset
            </p>
          )}
        </div>
      )}
      
      {error && (
        <div className="mb-4 p-3 bg-red-100 text-red-700 rounded-lg">
          {error}
        </div>
      )}
      
      <div className="space-y-4">
        <button
          onClick={() => fileInputRef.current?.click()}
          className="w-full flex items-center justify-center gap-2 bg-blue-500 text-white py-3 px-4 rounded-lg hover:bg-blue-600 transition-colors"
          type="button"
        >
          <Upload size={20} />
          Upload {type === 'audio' ? 'Audio' : 'Video'}
        </button>
        
        <input
          type="file"
          ref={fileInputRef}
          accept={type === 'audio' ? 'audio/*' : 'video/*'}
          onChange={handleFileChange}
          className="hidden"
        />

        <button
          onClick={isRecording ? stopRecording : startRecording}
          className={`w-full flex items-center justify-center gap-2 py-3 px-4 rounded-lg transition-colors ${
            isRecording
              ? 'bg-red-500 hover:bg-red-600'
              : 'bg-green-500 hover:bg-green-600'
          } text-white`}
          type="button"
          disabled={type === 'video' && selectedDataset === 'ravdess'}
        >
          {isRecording ? (
            <>
              <StopCircle size={20} />
              Stop Recording
            </>
          ) : (
            <>
              {type === 'audio' ? <Mic size={20} /> : <Video size={20} />}
              Start Recording
            </>
          )}
        </button>

        {type === 'video' && (
          <div className="relative aspect-video bg-gray-100 rounded-lg overflow-hidden">
            <video
              ref={videoRef}
              autoPlay
              muted
              playsInline
              className="w-full h-full object-cover"
            />
            <canvas 
              ref={canvasRef} 
              className="hidden" 
              width="640" 
              height="480" 
            />
            {livePrediction && (
              <div className="absolute top-2 left-2 bg-black bg-opacity-50 text-white px-3 py-1 rounded-full text-sm">
                {livePrediction.emotion} ({(livePrediction.confidence * 100).toFixed(1)}%)
              </div>
            )}
          </div>
        )}

        {type === 'audio' && audioUrl && (
          <div className="flex items-center gap-2">
            <button
              onClick={toggleAudioPlayback}
              className="flex items-center justify-center p-2 rounded-full bg-gray-100 hover:bg-gray-200"
              type="button"
            >
              {isPlaying ? <Pause size={20} /> : <Play size={20} />}
            </button>
            <audio
              ref={audioRef}
              src={audioUrl}
              onEnded={() => setIsPlaying(false)}
              className="w-full"
              controls
            />
          </div>
        )}
      </div>
    </div>
  );
}