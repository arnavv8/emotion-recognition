import React, { useState, useRef, useEffect } from 'react';
import { Smile, BarChart } from 'lucide-react';
import { MediaCard } from './components/MediaCard';
import { EmotionDisplay } from './components/EmotionDisplay';
import { MediaHistory } from './components/MediaHistory';
import { AuthForm } from './components/AuthForm';
import { useAuth } from './components/AuthProvider';
import { PredictionResult, StoredMedia, Recording } from './types';
import { uploadRecording, getRecordings, deleteRecording } from './lib/storage';

export default function App() {
  const auth = useAuth();
  const { user } = auth;
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [mediaHistory, setMediaHistory] = useState<StoredMedia[]>([]);
  const [recordings, setRecordings] = useState<Recording[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [modelStatus, setModelStatus] = useState<any>(null);
  const modalRef = useRef<HTMLDialogElement>(null);
  const [selectedMedia, setSelectedMedia] = useState<StoredMedia | null>(null);

  useEffect(() => {
    if (user) {
      loadRecordings();
      checkModelStatus();
    }
  }, [user]);

  const checkModelStatus = async () => {
    try {
      const response = await fetch('http://localhost:5000/status');
      if (response.ok) {
        const data = await response.json();
        setModelStatus(data);
      }
    } catch (error) {
      console.error('Error checking model status:', error);
    }
  };

  const loadRecordings = async () => {
    try {
      const data = await getRecordings();
      setRecordings(data);
    } catch (error) {
      console.error('Error loading recordings:', error);
      setError('Failed to load recordings');
    }
  };

  const handleMediaSubmit = async (media: File | Blob, dataset: string = 'cremad') => {
    if (!user) {
      setError('Please sign in to save recordings');
      return;
    }

    setIsLoading(true);
    setError(null);
    
    const formData = new FormData();
    formData.append('file', media);
    formData.append('type', media instanceof File ? 
      (media.type.startsWith('audio') ? 'audio' : 'video') : 
      (media.type.includes('audio') ? 'audio' : 'video'));
    formData.append('dataset', dataset);
    
    try {
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `Server returned ${response.status}: ${response.statusText}`);
      }
      
      const result = await response.json();
      if (!result.emotion) {
        throw new Error('Invalid response from server');
      }
      
      setPrediction(result);
      
      // Upload to Supabase
      const filename = media instanceof File ? media.name : `${media.type.includes('audio') ? 'audio' : 'video'}_recording_${Date.now()}`;
      await uploadRecording(
        media,
        filename,
        media.type.includes('audio') ? 'audio' : 'video',
        result.emotion,
        result.confidence
      );

      // Refresh recordings list
      await loadRecordings();
      
      const newMedia: StoredMedia = {
        id: Date.now().toString(),
        type: media.type.includes('audio') ? 'audio' : 'video',
        blob: media,
        filename,
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

  const handleDelete = async (id: string) => {
    try {
      await deleteRecording(id);
      await loadRecordings();
      setMediaHistory(prev => prev.filter(media => media.id !== id));
    } catch (error) {
      console.error('Error deleting recording:', error);
      setError('Failed to delete recording');
    }
  };

  const switchDataset = async (dataset: string) => {
    try {
      const response = await fetch('http://localhost:5000/switch-dataset', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ dataset }),
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `Failed to switch dataset`);
      }
      
      await checkModelStatus();
    } catch (error) {
      console.error('Error switching dataset:', error);
      setError(error instanceof Error ? error.message : 'Failed to switch dataset');
    }
  };

  const handleSignOut = () => {
    auth.signOut();
  };

  if (!user) {
    return <AuthForm />;
  }

  return (
    <div className="min-h-screen bg-gray-100">
      <header className="bg-white shadow-md">
        <div className="max-w-7xl mx-auto px-4 py-6 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Smile className="w-8 h-8 text-blue-500" />
              <h1 className="text-3xl font-bold text-gray-900">
                Emotion Recognition
              </h1>
            </div>
            <div className="flex items-center gap-4">
              {modelStatus && (
                <div className="text-sm text-gray-600">
                  Active dataset: <span className="font-medium">{modelStatus.activeDataset?.toUpperCase()}</span>
                </div>
              )}
              <button
                onClick={handleSignOut}
                className="text-gray-600 hover:text-gray-900"
              >
                Sign out
              </button>
            </div>
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
            onFileSelect={(file, dataset) => handleMediaSubmit(file, dataset)}
            onRecordingComplete={(blob, dataset) => handleMediaSubmit(blob, dataset)}
          />
          <MediaCard
            type="video"
            onFileSelect={(file, dataset) => handleMediaSubmit(file, dataset)}
            onRecordingComplete={(blob, dataset) => handleMediaSubmit(blob, dataset)}
            onLivePredict={handleLivePredict}
          />
        </div>

        <EmotionDisplay prediction={prediction} isLoading={isLoading} />
        
        {modelStatus && modelStatus.modelPerformance && Object.keys(modelStatus.modelPerformance).length > 0 && (
          <div className="bg-white rounded-lg shadow-lg p-6 mt-8">
            <div className="flex items-center gap-2 mb-4">
              <BarChart className="w-5 h-5 text-blue-500" />
              <h2 className="text-2xl font-bold text-gray-800">Model Performance</h2>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {Object.entries(modelStatus.modelPerformance).map(([key, metrics]: [string, any]) => (
                <div key={key} className="bg-gray-50 p-4 rounded-lg">
                  <h3 className="font-medium text-gray-800 mb-2">{key.replace('_', ' ').toUpperCase()}</h3>
                  <div className="space-y-1 text-sm">
                    <p>Accuracy: {(metrics.accuracy * 100).toFixed(2)}%</p>
                    {metrics.val_accuracy && (
                      <p>Validation Accuracy: {(metrics.val_accuracy * 100).toFixed(2)}%</p>
                    )}
                    {metrics.val_loss && (
                      <p>Validation Loss: {metrics.val_loss.toFixed(4)}</p>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
        
        <MediaHistory
          mediaList={recordings.map(recording => ({
            id: recording.id,
            type: recording.type,
            blob: new Blob(), // Add the blob property here
            filename: recording.filename,
            timestamp: new Date(recording.created_at),
            prediction: recording.emotion ? {
              emotion: recording.emotion as any,
              confidence: recording.confidence || 0
            } : undefined
          }))}
          onDelete={handleDelete}
        />
      </main>
    </div>
  );
}