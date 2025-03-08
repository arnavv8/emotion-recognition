import React from 'react';
import { Play, Video, Mic, Trash2 } from 'lucide-react';
import { StoredMedia } from '../types';

interface Props {
  mediaList: StoredMedia[];
  onDelete: (id: string) => void;
}

export function MediaHistory({ mediaList, onDelete }: Props) {
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
              className="flex items-center justify-between p-4 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
            >
              <div className="flex items-center gap-3">
                {media.type === 'audio' ? (
                  <Mic className="text-blue-500" />
                ) : (
                  <Video className="text-blue-500" />
                )}
                <div>
                  <p className="font-medium">
                    {media.filename || `${media.type === 'audio' ? 'Audio' : 'Video'} Recording`}
                  </p>
                  <div className="flex gap-2 text-sm text-gray-500">
                    <span>{new Date(media.timestamp).toLocaleString()}</span>
                    {media.prediction && (
                      <span className="px-2 py-0.5 bg-gray-200 rounded-full">
                        {media.prediction.emotion}
                      </span>
                    )}
                  </div>
                </div>
              </div>
              
              <div className="flex items-center gap-2">
                <button
                  onClick={() => onDelete(media.id)}
                  className="p-2 rounded-full hover:bg-gray-200 text-red-500 transition-colors"
                  title="Delete"
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