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
  filename: string;
  timestamp: Date;
  prediction?: PredictionResult;
}

export interface Recording {
  id: string;
  user_id: string;
  type: 'audio' | 'video';
  filename: string;
  file_path: string;
  emotion: Emotion | null;
  confidence: number | null;
  created_at: string;
}

export type Database = {
  public: {
    Tables: {
      recordings: {
        Row: Recording;
        Insert: Omit<Recording, 'id' | 'created_at'>;
        Update: Partial<Omit<Recording, 'id' | 'created_at'>>;
      };
    };
  };
};