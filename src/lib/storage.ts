import { supabase } from './supabase';
import { Recording } from '../types';

export async function uploadRecording(
  file: Blob,
  filename: string,
  type: 'audio' | 'video',
  emotion: string | null,
  confidence: number | null
): Promise<Recording> {
  const user = supabase.auth.getUser();
  if (!user) {
    throw new Error('User not authenticated');
  }

  // Upload file to Supabase Storage
  const fileExt = type === 'audio' ? 'webm' : 'mp4';
  const filePath = `${type}s/${Date.now()}_${filename}.${fileExt}`;
  
  const { error: uploadError } = await supabase.storage
    .from('recordings')
    .upload(filePath, file);

  if (uploadError) {
    throw uploadError;
  }

  // Create database record
  const { data, error: dbError } = await supabase
    .from('recordings')
    .insert({
      type,
      filename,
      file_path: filePath,
      emotion,
      confidence,
    })
    .select()
    .single();

  if (dbError) {
    throw dbError;
  }

  return data;
}

export async function getRecordings(): Promise<Recording[]> {
  const { data, error } = await supabase
    .from('recordings')
    .select('*')
    .order('created_at', { ascending: false });

  if (error) {
    throw error;
  }

  return data || [];
}

export async function deleteRecording(id: string): Promise<void> {
  const { data: recording, error: fetchError } = await supabase
    .from('recordings')
    .select('file_path')
    .eq('id', id)
    .single();

  if (fetchError) {
    throw fetchError;
  }

  // Delete file from storage
  const { error: storageError } = await supabase.storage
    .from('recordings')
    .remove([recording.file_path]);

  if (storageError) {
    throw storageError;
  }

  // Delete database record
  const { error: dbError } = await supabase
    .from('recordings')
    .delete()
    .eq('id', id);

  if (dbError) {
    throw dbError;
  }
}