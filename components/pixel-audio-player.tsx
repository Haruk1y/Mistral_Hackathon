"use client";

import { useEffect, useMemo, useRef, useState } from "react";

const formatTime = (seconds: number) => {
  if (!Number.isFinite(seconds) || seconds < 0) return "0:00";
  const total = Math.floor(seconds);
  const minutes = Math.floor(total / 60);
  const remain = total % 60;
  return `${minutes}:${remain.toString().padStart(2, "0")}`;
};

export const PixelAudioPlayer = ({
  src,
  locale,
  className = ""
}: {
  src: string;
  locale: "ja" | "en";
  className?: string;
}) => {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [playing, setPlaying] = useState(false);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);

  useEffect(() => {
    setPlaying(false);
    setDuration(0);
    setCurrentTime(0);

    const audio = audioRef.current;
    if (!audio) return;
    audio.pause();
    audio.currentTime = 0;
  }, [src]);

  const togglePlayback = async () => {
    const audio = audioRef.current;
    if (!audio) return;

    if (audio.paused) {
      try {
        await audio.play();
      } catch {
        setPlaying(false);
      }
      return;
    }

    audio.pause();
  };

  const onSeek = (nextValue: string) => {
    const audio = audioRef.current;
    if (!audio) return;
    const nextTime = Number(nextValue);
    if (!Number.isFinite(nextTime)) return;
    audio.currentTime = nextTime;
    setCurrentTime(nextTime);
  };

  const canSeek = duration > 0;
  const playLabel = locale === "ja" ? "再生" : "Play";
  const pauseLabel = locale === "ja" ? "一時停止" : "Pause";
  const currentLabel = useMemo(() => formatTime(currentTime), [currentTime]);
  const durationLabel = useMemo(() => formatTime(duration), [duration]);

  return (
    <div className={`audio-player ${className}`}>
      <audio
        ref={audioRef}
        src={src}
        preload="metadata"
        onLoadedMetadata={(event) => {
          setDuration(event.currentTarget.duration || 0);
        }}
        onTimeUpdate={(event) => {
          setCurrentTime(event.currentTarget.currentTime || 0);
        }}
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
        onEnded={() => setPlaying(false)}
      />
      <button
        type="button"
        className="audio-player-toggle"
        onClick={() => void togglePlayback()}
        aria-label={playing ? pauseLabel : playLabel}
      >
        {playing ? "PAUSE" : "PLAY"}
      </button>
      <input
        className="audio-player-seek"
        type="range"
        min={0}
        max={canSeek ? duration : 1}
        step={0.01}
        value={canSeek ? currentTime : 0}
        onChange={(event) => onSeek(event.target.value)}
        disabled={!canSeek}
        aria-label={locale === "ja" ? "再生位置" : "Playback Position"}
      />
      <span className="audio-player-time">
        {currentLabel} / {durationLabel}
      </span>
    </div>
  );
};
