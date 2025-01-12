import React, { useState, useRef } from 'react';
import { Smile } from 'lucide-react';
import { MediaCard } from './components/MediaCard';
import { EmotionDisplay } from './components/EmotionDisplay';
import { MediaHistory } from './components/MediaHistory';
import { PredictionResult, StoredMedia } from './types';

function App() {
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [mediaHistory, setMediaHistory] = useState<StoredMedia[]>([]);
  const [error, setError] = useState<string | null>(null);
  const modalRef = useRef<HTMLDialogElement>(null);
  const [selectedMedia, setSelectedMedia] = useState<StoredMedia | null>(null);

  const handleMediaSubmit = async (media: File | Blob, type: 'audio' | 'video') => {
    setIsLoading(true);
    setError(null);
    
    const formData = new FormData();
    formData.append('file', media);
    formData.append('type', type);
    
    try {
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error(`Server returned ${response.status}: ${response.statusText}`);
      }
      
      const result = await response.json();
      if (!result.emotion) {
        throw new Error('Invalid response from server');
      }
      
      setPrediction(result);
      
      const newMedia: StoredMedia = {
        id: Date.now().toString(),
        type,
        blob: media,
        timestamp: new Date(),
        prediction: result,
      };
      
      setMediaHistory(prev => [newMedia, ...prev]);
    } catch (error) {
      console.error('Error:', error);
      setError(error instanceof Error ? error.message : 'Failed to process media');
      setPrediction(null);
    } finally {
      setIsLoading(false);
    }
  };

  const handleLivePredict = async (frame: ImageData): Promise<PredictionResult> => {
    const formData = new FormData();
    const blob = new Blob([frame.data], { type: 'image/png' });
    formData.append('frame', blob);
    
    try {
      const response = await fetch('http://localhost:5000/predict-frame', {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error(`Server returned ${response.status}: ${response.statusText}`);
      }
      
      const result = await response.json();
      if (!result.emotion) {
        throw new Error('Invalid response from server');
      }
      
      return result;
    } catch (error) {
      console.error('Error:', error);
      return {
        emotion: 'neutral',
        confidence: 0,
        videoScore: 0,
      };
    }
  };

  const playMedia = (media: StoredMedia) => {
    setSelectedMedia(media);
    modalRef.current?.showModal();
  };

  const deleteMedia = (id: string) => {
    setMediaHistory(prev => prev.filter(media => media.id !== id));
  };

  return (
    <div className="min-h-screen bg-gray-100">
      <header className="bg-white shadow-md">
        <div className="max-w-7xl mx-auto px-4 py-6 sm:px-6 lg:px-8">
          <div className="flex items-center gap-3">
            <Smile className="w-8 h-8 text-blue-500" />
            <h1 className="text-3xl font-bold text-gray-900">
              Emotion Recognition
            </h1>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-8 sm:px-6 lg:px-8">
        {error && (
          <div className="mb-8 p-4 bg-red-100 border border-red-400 text-red-700 rounded-lg">
            {error}
          </div>
        )}

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
          <MediaCard
            type="audio"
            onFileSelect={(file) => handleMediaSubmit(file, 'audio')}
            onRecordingComplete={(blob) => handleMediaSubmit(blob, 'audio')}
          />
          <MediaCard
            type="video"
            onFileSelect={(file) => handleMediaSubmit(file, 'video')}
            onRecordingComplete={(blob) => handleMediaSubmit(blob, 'video')}
            onLivePredict={handleLivePredict}
          />
        </div>

        <EmotionDisplay prediction={prediction} isLoading={isLoading} />
        
        <MediaHistory
          mediaList={mediaHistory}
          onPlay={playMedia}
          onDelete={deleteMedia}
        />
      </main>

      <dialog
        ref={modalRef}
        className="p-4 rounded-lg shadow-xl backdrop:bg-black backdrop:bg-opacity-50"
        onClick={(e) => {
          if (e.target === modalRef.current) {
            modalRef.current.close();
          }
        }}
      >
        {selectedMedia && (
          <div className="min-w-[320px]">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-xl font-semibold">
                {selectedMedia.type === 'audio' ? 'Audio' : 'Video'} Playback
              </h3>
              <button
                onClick={() => modalRef.current?.close()}
                className="text-gray-500 hover:text-gray-700"
              >
                ×
              </button>
            </div>
            {selectedMedia.type === 'audio' ? (
              <audio
                src={URL.createObjectURL(selectedMedia.blob)}
                controls
                className="w-full"
                autoPlay
              />
            ) : (
              <video
                src={URL.createObjectURL(selectedMedia.blob)}
                controls
                className="w-full"
                autoPlay
              />
            )}
          </div>
        )}
      </dialog>
    </div>
  );
}

export default App;