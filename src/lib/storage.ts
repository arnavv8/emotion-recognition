import { supabase } from './supabase';
import { Recording } from '../types';

export async function uploadRecording(
  file: Blob,
  filename: string,
  type: 'audio' | 'video',
  emotion: string | null,
  confidence: number | null
): Promise<Recording> {
  // ✅ Ensure user is authenticated
  const { data: userData, error: authError } = await supabase.auth.getUser();
  if (authError || !userData?.user) {
    throw new Error('User not authenticated');
  }
  
  const userId = userData.user.id;

  // ✅ Ensure unique filename with timestamp
  const fileExt = type === 'audio' ? 'webm' : 'mp4';
  const filePath = `${type}s/${Date.now()}_${filename}.${fileExt}`;

  // ✅ Upload file to Supabase Storage
  const { error: uploadError } = await supabase.storage
    .from('recordings')
    .upload(filePath, file);

  if (uploadError) {
    console.error("Upload failed:", uploadError.message);
    throw uploadError;
  }

  // ✅ Insert record into database
  const { data, error: dbError } = await supabase
    .from('recordings')
    .insert({
      user_id: userId,  // ✅ Track uploader
      type,
      filename,
      file_path: filePath,
      emotion,
      confidence,
    })
    .select()
    .single();

  if (dbError) {
    console.error("Database insert failed:", dbError.message);
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
    console.error("Fetch failed:", error.message);
    throw error;
  }

  return data || [];
}

export async function deleteRecording(id: string): Promise<void> {
  // ✅ Fetch recording details
  const { data: recording, error: fetchError } = await supabase
    .from('recordings')
    .select('file_path')
    .eq('id', id)
    .single();

  if (fetchError || !recording) {
    throw new Error('Recording not found');
  }

  // ✅ Delete file from storage
  if (recording.file_path) {
    const { error: storageError } = await supabase.storage
      .from('recordings')
      .remove([recording.file_path]);

    if (storageError) {
      console.error("Storage deletion failed:", storageError.message);
      throw storageError;
    }
  }

  // ✅ Delete record from database
  const { error: dbError } = await supabase
    .from('recordings')
    .delete()
    .eq('id', id);

  if (dbError) {
    console.error("Database deletion failed:", dbError.message);
    throw dbError;
  }
}
