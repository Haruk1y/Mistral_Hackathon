import { CATALOG_CUSTOMERS } from "@/lib/catalog";
import type { Customer } from "@/lib/types";

export type StreetCrowdPoseKey = "front" | "diag_left" | "diag_right";

export type StreetCrowdCharacter = {
  id: string;
  customerId: Customer["id"];
  name: string;
  profile: string;
  portraitAsset: string;
  sprites: Record<StreetCrowdPoseKey, string>;
};

const TOTAL_CAST = 20;
const CUSTOMER_ORDER: Customer["id"][] = ["anna", "ben", "cara", "dan"];
const POSE_KEYS: StreetCrowdPoseKey[] = ["front", "diag_left", "diag_right"];
const EXCLUDED_CAST_IDS = new Set<string>([
  "crowd_02",
  "crowd_04",
  "crowd_08",
  "crowd_10",
  "crowd_14",
  "crowd_16",
  "crowd_18",
  "crowd_19",
  "crowd_20"
]);

const pad2 = (value: number) => String(value).padStart(2, "0");

const customerForCast = (castNumber: number): Customer["id"] =>
  CUSTOMER_ORDER[(castNumber - 1) % CUSTOMER_ORDER.length];

const CAST_CUSTOMER_BY_ID: Record<string, Customer["id"]> = {
  crowd_01: "anna",
  crowd_03: "ben",
  crowd_05: "cara",
  crowd_06: "anna",
  crowd_07: "cara",
  crowd_09: "ben",
  crowd_11: "cara",
  crowd_12: "dan",
  crowd_13: "ben",
  crowd_15: "anna",
  crowd_17: "dan"
};

const customerNameMap = new Map(CATALOG_CUSTOMERS.map((customer) => [customer.id, customer.name] as const));

const CAST_PERSONAS: Record<string, { name: string; profile: string }> = {
  crowd_01: {
    name: "ラルフ",
    profile: "白いバンダナの荷運び。朝の市場で一番乗りし、足取りに合う軽快なビートを好む。"
  },
  crowd_02: {
    name: "ノーラ",
    profile: "灰色フードの薬草売り。雨上がりの路地で静かに店を開き、湿った空気に溶ける音を求める。"
  },
  crowd_03: {
    name: "レオ",
    profile: "革エプロンの見習い職人。手を動かすときの集中を切らさない素直なリズムが好き。"
  },
  crowd_04: {
    name: "マーサ",
    profile: "フード姿の旅商人。遠い街の土産話に合う素朴で温かな旋律を集めている。"
  },
  crowd_05: {
    name: "ヴィクター",
    profile: "銀髪の元執事。襟を正したくなる端正な和声と、落ち着いたテンポを好む。"
  },
  crowd_06: {
    name: "こはる",
    profile: "赤いおだんご髪の配達見習い。朝市に合う軽やかな曲が好き。"
  },
  crowd_07: {
    name: "ガロ",
    profile: "太い眉の鍛冶屋。火花のように芯のある低音と、粘り強いグルーヴを求める。"
  },
  crowd_08: {
    name: "エドガー",
    profile: "丸眼鏡の古書修復士。紙の匂いと一緒に聴ける柔らかなアコースティックを愛する。"
  },
  crowd_09: {
    name: "湊",
    profile: "丸メガネの書店手伝い。ページをめくる手が進む穏やかな曲を好む。"
  },
  crowd_10: {
    name: "フローレンス",
    profile: "銀髪の食堂女将。夕方の客足がゆるむ頃に似合う、やさしいスウィングを集める。"
  },
  crowd_11: {
    name: "ハルト",
    profile: "槍持ちの見回り役。背筋が伸びるような明瞭なリズムと、無駄のない構成が好き。"
  },
  crowd_12: {
    name: "オズワルド",
    profile: "笑顔のパン職人。焼きたての香りに合う、あたたかく丸いサウンドを求める。"
  },
  crowd_13: {
    name: "ユリウス",
    profile: "紺衣の航路案内人。潮の満ち引きを思わせるゆったりした抑揚を好む。"
  },
  crowd_14: {
    name: "澄江",
    profile: "眼鏡の仕立て屋。夕暮れに似合う上品で落ち着いた旋律を求める。"
  },
  crowd_15: {
    name: "リタ",
    profile: "赤髪の花売り。店先を明るくする、抜けの良いメロディが得意分野。"
  },
  crowd_16: {
    name: "トマス",
    profile: "笑い皺の深い時計職人。針の刻みに合う正確な拍と、控えめな装飾を好む。"
  },
  crowd_17: {
    name: "イアン",
    profile: "若い旅の地図描き。遠回りしたくなるような、余韻の長い進行を集めている。"
  },
  crowd_18: {
    name: "オリーヴ",
    profile: "巻き髪の薬師。深呼吸したくなる中域の温かさと穏やかな持続音を求める。"
  },
  crowd_19: {
    name: "ギデオン",
    profile: "白髭の旅人。古い街の思い出を呼び起こす温かな音を探している。"
  },
  crowd_20: {
    name: "宗玄",
    profile: "白髭の旅人。古い街の思い出を呼び起こす温かな音を探している。"
  }
};

export const STREET_CROWD_CHARACTERS: StreetCrowdCharacter[] = Array.from({ length: TOTAL_CAST }, (_, index) => {
  const castNumber = index + 1;
  const castId = `crowd_${pad2(castNumber)}`;
  const customerId = CAST_CUSTOMER_BY_ID[castId] ?? customerForCast(castNumber);
  const persona = CAST_PERSONAS[castId];

  return {
    id: castId,
    customerId,
    name: persona?.name ?? customerNameMap.get(customerId) ?? customerId.toUpperCase(),
    profile: persona?.profile ?? "街角で依頼を探している。",
    portraitAsset: `/assets/characters/portraits/street-crowd-v2/por_${castId}_face_160.png`,
    sprites: {
      front: `crowd_${pad2(castNumber)}_front_64x96.png`,
      diag_left: `crowd_${pad2(castNumber)}_diag_left_64x96.png`,
      diag_right: `crowd_${pad2(castNumber)}_diag_right_64x96.png`
    }
  };
}).filter((character) => !EXCLUDED_CAST_IDS.has(character.id));

export const STREET_CROWD_CHARACTER_BY_ID: Record<string, StreetCrowdCharacter> = Object.fromEntries(
  STREET_CROWD_CHARACTERS.map((character) => [character.id, character] as const)
);

export const STREET_CROWD_CHARACTER_IDS_BY_CUSTOMER: Record<Customer["id"], string[]> =
  CUSTOMER_ORDER.reduce<Record<Customer["id"], string[]>>(
    (acc, customerId) => ({
      ...acc,
      [customerId]: STREET_CROWD_CHARACTERS.filter((character) => character.customerId === customerId).map(
        (character) => character.id
      )
    }),
    {
      anna: [],
      ben: [],
      cara: [],
      dan: []
    }
  );

export const normalizeStreetCrowdPose = (value: string | null | undefined): StreetCrowdPoseKey => {
  if (value === "diag_left" || value === "diag_right" || value === "front") {
    return value;
  }
  return "front";
};

export const STREET_CROWD_POSE_KEYS = POSE_KEYS;
