export type Emotion = 'happy' | 'sad' | 'angry' | 'neutral' | 'surprised' | 'fearful' | 'disgusted';

export interface PredictionResult {
  emotion: Emotion;
  confidence: number;
  audioScore?: number;
  videoScore?: number;
}

export interface MediaRecorderProps {
  onRecordingComplete: (blob: Blob) => void;
  isRecording: boolean;
  onToggleRecording: () => void;
}

export interface StoredMedia {
  id: string;
  type: 'audio' | 'video';
  blob: Blob;
  timestamp: Date;
  prediction?: PredictionResult;
}