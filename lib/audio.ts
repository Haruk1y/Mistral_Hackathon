const toLittleEndian16 = (value: number) => [value & 0xff, (value >> 8) & 0xff];
const toLittleEndian32 = (value: number) => [value & 0xff, (value >> 8) & 0xff, (value >> 16) & 0xff, (value >> 24) & 0xff];

export const generateSineWaveDataUri = (options?: { durationSec?: number; frequency?: number; sampleRate?: number }) => {
  const durationSec = options?.durationSec ?? 1.3;
  const frequency = options?.frequency ?? 392;
  const sampleRate = options?.sampleRate ?? 22050;
  const numSamples = Math.floor(sampleRate * durationSec);

  const samples = new Int16Array(numSamples);
  for (let i = 0; i < numSamples; i += 1) {
    const t = i / sampleRate;
    const envelope = Math.exp(-t * 3);
    samples[i] = Math.floor(Math.sin(2 * Math.PI * frequency * t) * envelope * 32767 * 0.22);
  }

  const dataSize = samples.length * 2;
  const fileSize = 44 + dataSize;

  const header = [
    ...Array.from(Buffer.from("RIFF")),
    ...toLittleEndian32(fileSize - 8),
    ...Array.from(Buffer.from("WAVE")),
    ...Array.from(Buffer.from("fmt ")),
    ...toLittleEndian32(16),
    ...toLittleEndian16(1),
    ...toLittleEndian16(1),
    ...toLittleEndian32(sampleRate),
    ...toLittleEndian32(sampleRate * 2),
    ...toLittleEndian16(2),
    ...toLittleEndian16(16),
    ...Array.from(Buffer.from("data")),
    ...toLittleEndian32(dataSize)
  ];

  const pcm = new Uint8Array(dataSize);
  for (let i = 0; i < samples.length; i += 1) {
    const sample = samples[i];
    pcm[i * 2] = sample & 0xff;
    pcm[i * 2 + 1] = (sample >> 8) & 0xff;
  }

  const wav = new Uint8Array(fileSize);
  wav.set(header, 0);
  wav.set(pcm, 44);

  const base64 = Buffer.from(wav).toString("base64");
  return `data:audio/wav;base64,${base64}`;
};
