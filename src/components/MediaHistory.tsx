import React from 'react';
import { Play, Video, Mic, Trash2 } from 'lucide-react';
import { StoredMedia } from '../types';

interface Props {
  mediaList: StoredMedia[];
  onPlay: (media: StoredMedia) => void;
  onDelete: (id: string) => void;
}

export function MediaHistory({ mediaList, onPlay, onDelete }: Props) {
  return (
    <div className="bg-white rounded-lg shadow-lg p-6 mt-8">
      <h2 className="text-2xl font-bold mb-4 text-gray-800">Recording History</h2>
      
      {mediaList.length === 0 ? (
        <p className="text-gray-500 text-center py-4">No recordings yet</p>
      ) : (
        <div className="space-y-4">
          {mediaList.map((media) => (
            <div
              key={media.id}
              className="flex items-center justify-between p-4 bg-gray-50 rounded-lg"
            >
              <div className="flex items-center gap-3">
                {media.type === 'audio' ? (
                  <Mic className="text-blue-500" />
                ) : (
                  <Video className="text-blue-500" />
                )}
                <div>
                  <p className="font-medium">
                    {media.type === 'audio' ? 'Audio' : 'Video'} Recording
                  </p>
                  <p className="text-sm text-gray-500">
                    {new Date(media.timestamp).toLocaleString()}
                  </p>
                </div>
              </div>
              
              <div className="flex items-center gap-2">
                {media.prediction && (
                  <span className="text-sm bg-gray-200 px-2 py-1 rounded">
                    {media.prediction.emotion}
                  </span>
                )}
                <button
                  onClick={() => onPlay(media)}
                  className="p-2 rounded-full hover:bg-gray-200"
                >
                  <Play size={20} />
                </button>
                <button
                  onClick={() => onDelete(media.id)}
                  className="p-2 rounded-full hover:bg-gray-200 text-red-500"
                >
                  <Trash2 size={20} />
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}