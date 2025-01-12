import React from 'react';
import { PredictionResult } from '../types';

interface Props {
  prediction: PredictionResult | null;
  isLoading: boolean;
}

export function EmotionDisplay({ prediction, isLoading }: Props) {
  if (isLoading) {
    return (
      <div className="bg-white rounded-lg shadow-lg p-6 w-full max-w-md">
        <div className="animate-pulse space-y-4">
          <div className="h-8 bg-gray-200 rounded w-3/4"></div>
          <div className="h-4 bg-gray-200 rounded w-1/2"></div>
        </div>
      </div>
    );
  }

  if (!prediction) {
    return null;
  }

  const getEmotionColor = (emotion: string) => {
    const colors: Record<string, string> = {
      happy: 'bg-yellow-500',
      sad: 'bg-blue-500',
      angry: 'bg-red-500',
      neutral: 'bg-gray-500',
      surprised: 'bg-purple-500',
      fearful: 'bg-orange-500',
      disgusted: 'bg-green-500',
    };
    return colors[emotion] || 'bg-gray-500';
  };

  return (
    <div className="bg-white rounded-lg shadow-lg p-6 w-full max-w-md">
      <h2 className="text-2xl font-bold mb-4 text-gray-800">Prediction Result</h2>
      
      <div className="space-y-4">
        <div className="flex items-center gap-3">
          <div className={`w-4 h-4 rounded-full ${getEmotionColor(prediction.emotion)}`}></div>
          <span className="text-xl font-semibold capitalize">{prediction.emotion}</span>
        </div>
      </div>
    </div>
  );
}