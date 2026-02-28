import { CATALOG_CUSTOMERS } from "@/lib/catalog";
import type { Customer } from "@/lib/types";

export type StreetCrowdPoseKey = "front" | "diag_left" | "diag_right";

export type StreetCrowdCharacter = {
  id: string;
  customerId: Customer["id"];
  name: string;
  portraitAsset: string;
  sprites: Record<StreetCrowdPoseKey, string>;
};

const TOTAL_CAST = 20;
const CUSTOMER_ORDER: Customer["id"][] = ["anna", "ben", "cara", "dan"];
const POSE_KEYS: StreetCrowdPoseKey[] = ["front", "diag_left", "diag_right"];

const pad2 = (value: number) => String(value).padStart(2, "0");

const customerForCast = (castNumber: number): Customer["id"] =>
  CUSTOMER_ORDER[(castNumber - 1) % CUSTOMER_ORDER.length];

const customerNameMap = new Map(CATALOG_CUSTOMERS.map((customer) => [customer.id, customer.name] as const));

export const STREET_CROWD_CHARACTERS: StreetCrowdCharacter[] = Array.from(
  { length: TOTAL_CAST },
  (_, index) => {
    const castNumber = index + 1;
    const castId = `crowd_${pad2(castNumber)}`;
    const customerId = customerForCast(castNumber);

    return {
      id: castId,
      customerId,
      name: customerNameMap.get(customerId) ?? customerId.toUpperCase(),
      portraitAsset: "/assets/placeholders/portrait.svg",
      sprites: {
        front: "/assets/placeholders/sprite.svg",
        diag_left: "/assets/placeholders/sprite.svg",
        diag_right: "/assets/placeholders/sprite.svg"
      }
    };
  }
);

export const STREET_CROWD_CHARACTER_BY_ID: Record<string, StreetCrowdCharacter> = Object.fromEntries(
  STREET_CROWD_CHARACTERS.map((character) => [character.id, character] as const)
);

const PRIMARY_CAST_BY_CUSTOMER: Record<Customer["id"], string[]> = {
  anna: ["crowd_06"],
  ben: ["crowd_09"],
  cara: ["crowd_14"],
  dan: ["crowd_19"]
};

export const STREET_CROWD_CHARACTER_IDS_BY_CUSTOMER: Record<Customer["id"], string[]> =
  PRIMARY_CAST_BY_CUSTOMER;

export const normalizeStreetCrowdPose = (value: string | null | undefined): StreetCrowdPoseKey => {
  if (value === "diag_left" || value === "diag_right" || value === "front") {
    return value;
  }
  return "front";
};

export const STREET_CROWD_POSE_KEYS = POSE_KEYS;
