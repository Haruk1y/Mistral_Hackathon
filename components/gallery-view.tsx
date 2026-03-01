"use client";

import Link from "next/link";
import { useLocale } from "@/components/locale-context";
import { PixelAudioPlayer } from "@/components/pixel-audio-player";
import { PixelPanel, PixelTag } from "@/components/pixel-ui";
import { useGame } from "@/components/game-context";
import { CATALOG_PARTS } from "@/lib/catalog";

const getPartName = (partId: string) => CATALOG_PARTS.find((part) => part.id === partId)?.name ?? partId;

export const GalleryView = () => {
  const { tracks, state } = useGame();
  const { locale, text } = useLocale();
  const weatherClass = `scene-weather-${state?.weather ?? "sunny"}`;

  return (
    <div className={`gallery-layout ${weatherClass}`}>
      <PixelPanel title={text("galleryTitle")} className="gallery-main">
        {tracks.length === 0 ? (
          <p>{text("galleryEmpty")}</p>
        ) : (
          <ul className="track-list">
            {tracks.map((track) => (
              <li key={track.id} className="track-item">
                <div>
                  <h4>
                    {text("galleryTrackPrefix")} #{track.id.slice(0, 6)}
                  </h4>
                  <p>
                    {text("galleryScore")} {track.score} / {text("galleryRank")} {track.rank}
                  </p>
                  <div className="stack-row wrap">
                    {Object.entries(track.usedPartsBySlot).map(([slot, partId]) => (
                      <PixelTag key={`${track.id}-${slot}`}>
                        {slot.toUpperCase()}: {getPartName(partId)}
                      </PixelTag>
                    ))}
                  </div>
                </div>
                <PixelAudioPlayer src={track.audioUrl} locale={locale} />
              </li>
            ))}
          </ul>
        )}
      </PixelPanel>

      <PixelPanel title={text("galleryProgressTitle")} className="gallery-side">
        <p>
          {text("galleryDelivered")}: {tracks.length}
        </p>
        <p>
          {text("galleryTotalRequests")}: {state?.commissionOrder.length ?? 0}
        </p>
        <Link href="/game/street" className="pixel-button-link">
          {text("galleryBackToStreet")}
        </Link>
      </PixelPanel>
    </div>
  );
};
