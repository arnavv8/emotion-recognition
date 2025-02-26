/*
  # Create recordings table

  1. New Tables
    - `recordings`
      - `id` (uuid, primary key)
      - `user_id` (uuid, references auth.users)
      - `type` (text, either 'audio' or 'video')
      - `filename` (text)
      - `file_path` (text)
      - `emotion` (text)
      - `confidence` (numeric)
      - `created_at` (timestamptz)

  2. Security
    - Enable RLS on `recordings` table
    - Add policies for authenticated users to:
      - Read their own recordings
      - Insert new recordings
      - Delete their own recordings
*/

CREATE TABLE IF NOT EXISTS recordings (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid REFERENCES auth.users NOT NULL,
  type text NOT NULL CHECK (type IN ('audio', 'video')),
  filename text NOT NULL,
  file_path text NOT NULL,
  emotion text,
  confidence numeric,
  created_at timestamptz DEFAULT now()
);

ALTER TABLE recordings ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can read own recordings"
  ON recordings
  FOR SELECT
  TO authenticated
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own recordings"
  ON recordings
  FOR INSERT
  TO authenticated
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can delete own recordings"
  ON recordings
  FOR DELETE
  TO authenticated
  USING (auth.uid() = user_id);