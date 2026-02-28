"use client";

import Link from "next/link";
import { useMemo } from "react";
import { useLocale } from "@/components/locale-context";
import { getCustomerName } from "@/lib/game-engine";
import { commissionStatusLabel } from "@/lib/i18n";
import {
  STREET_CROWD_CHARACTER_BY_ID,
  STREET_CROWD_CHARACTER_IDS_BY_CUSTOMER,
  STREET_CROWD_CHARACTERS,
  STREET_CROWD_POSE_KEYS,
  type StreetCrowdPoseKey
} from "@/lib/street-crowd-catalog";
import type { Commission, Weather } from "@/lib/types";

const hashString = (input: string): number => {
  let hash = 2166136261;
  for (let i = 0; i < input.length; i += 1) {
    hash ^= input.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
};

const mulberry32 = (seed: number) => {
  let t = seed >>> 0;
  return () => {
    t += 0x6d2b79f5;
    let x = Math.imul(t ^ (t >>> 15), t | 1);
    x ^= x + Math.imul(x ^ (x >>> 7), x | 61);
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
  };
};

const clamp = (value: number, min: number, max: number) => Math.max(min, Math.min(max, value));

export const StreetCommissionLayer = ({
  day,
  weather,
  commissions,
  onSelect
}: {
  day: number;
  weather: Weather;
  commissions: Commission[];
  onSelect?: (commissionId: string) => void;
}) => {
  const { locale } = useLocale();

  const actors = useMemo(() => {
    if (commissions.length === 0) return [];

    const allCastIds = STREET_CROWD_CHARACTERS.map((character) => character.id);
    if (allCastIds.length === 0) return [];

    const activeCommissions = commissions.filter((commission) => commission.status !== "delivered");
    const source = activeCommissions.length > 0 ? activeCommissions : commissions;
    const target = source.slice(0, 6);
    const random = mulberry32(
      hashString(`street-actors:${day}:${weather}:${target.map((item) => item.id).join(",")}`)
    );
    const usedCastIds = new Set<string>();

    return target.map((commission, index) => {
      const preferredIds = STREET_CROWD_CHARACTER_IDS_BY_CUSTOMER[commission.customerId] ?? allCastIds;
      const primaryPool = preferredIds.length > 0 ? preferredIds : allCastIds;

      let castId = primaryPool[Math.floor(random() * primaryPool.length)];
      let guard = 0;
      while (usedCastIds.has(castId) && guard < primaryPool.length) {
        const next = (primaryPool.indexOf(castId) + 1) % primaryPool.length;
        castId = primaryPool[next];
        guard += 1;
      }

      if (usedCastIds.has(castId)) {
        const fallbackPool = allCastIds.filter((id) => !usedCastIds.has(id));
        if (fallbackPool.length > 0) {
          castId = fallbackPool[Math.floor(random() * fallbackPool.length)];
        }
      }
      usedCastIds.add(castId);

      const character = STREET_CROWD_CHARACTER_BY_ID[castId] ?? STREET_CROWD_CHARACTERS[0];
      const pose: StreetCrowdPoseKey =
        STREET_CROWD_POSE_KEYS[Math.floor(random() * STREET_CROWD_POSE_KEYS.length)] ?? "front";
      const spriteFile = character.sprites[pose] ?? STREET_CROWD_CHARACTERS[0].sprites.front;

      const count = Math.max(1, target.length);
      const lane = (index + 1) / (count + 1);
      const left = clamp(10 + lane * 80 + (random() - 0.5) * 10, 8, 92);
      const top = clamp(63 + (random() - 0.5) * 10, 56, 76);

      return {
        castId: character.id,
        pose,
        commission,
        src: `/assets/characters/sprites/street-crowd-v2/${spriteFile}`,
        left,
        top,
        opacity: 0.84 + random() * 0.16
      };
    });
  }, [commissions, day, weather]);

  if (actors.length === 0) return null;

  return (
    <div className="street-commission-layer" aria-label="Street commissions">
      {actors.map((actor) => (
        <Link
          key={actor.commission.id}
          href={{
            pathname: `/game/compose/${actor.commission.id}`,
            query: {
              cast: actor.castId
            }
          }}
          className="street-commission-actor"
          style={{
            left: `${actor.left}%`,
            top: `${actor.top}%`,
            zIndex: Math.round(actor.top),
            opacity: actor.opacity,
            transform: "translate(-50%, -100%)"
          }}
          title={`${getCustomerName(actor.commission.customerId)} - ${commissionStatusLabel(
            locale,
            actor.commission.status
          )}`}
          onClick={() => onSelect?.(actor.commission.id)}
        >
          <img src={actor.src} alt="" className="street-commission-sprite" />
          <span className="street-commission-bubble">♪</span>
        </Link>
      ))}
    </div>
  );
};
