import type { CommissionStatus, MusicJobStatus, SlotKey, Weather } from "@/lib/types";

export type Locale = "ja" | "en";

export const LOCALE_STORAGE_KEY = "otokotoba-locale";
export const DEFAULT_LOCALE: Locale = "ja";
export const SUPPORTED_LOCALES: Locale[] = ["ja", "en"];

const EN_MESSAGES = {
  navBrand: "Atelier kotone",
  navStreet: "Town",
  navShop: "Shop",
  navKotone: "Kotone",
  navGallery: "Gallery",
  navMoney: "Money",
  navDay: "Day",
  navWeather: "Weather",
  langJapanese: "日本語",
  langEnglish: "English",
  streetWaitingTitle: "Waiting at the Corner",
  streetLead: "Read the atmosphere of the town and deliver each abstract request as otowords.",
  streetGoToShop: "Go to Shop",
  streetOpenGallery: "Open Gallery",
  streetResetSave: "Reset Save",
  streetStartNextDay: "Start Next Day",
  commissionPanelTitle: "Town Requests",
  commissionSelect: "Select",
  commissionInterpret: "Interpret",
  commissionCompose: "Compose",
  composeTitle: "Compose",
  composeNotFound: "Commission not found.",
  composeCustomerTitle: "Customer",
  composeRunInterpreter: "Run Interpreter",
  composeHintPrefix: "Hint",
  composePanelTitle: "Composition Panel",
  composePartsPanelTitle: "Owned Kotone",
  composePartsSlotPrefix: "Active Slot",
  composePartsHint: "Click a slot above, then choose one kotone from this list.",
  composeSelectPlaceholder: "-- select --",
  composeRecommendedPrefix: "Recommended",
  composeNeedInterpreter: "Run Interpreter to see recommendations.",
  composeNoPartsInSlot: "No owned kotone in this slot yet.",
  composePreviewPhrase: "Preview Phrase",
  composePreviewing: "Previewing...",
  composeAndPlay: "Compose & Play",
  composeReset: "Reset",
  composeResultTitle: "Result",
  composeScoreLabel: "Score",
  composeRankLabel: "Rank",
  composeRewardLabel: "Reward",
  composeMusicJobLabel: "Music Job",
  shopTitle: "Music Supplies",
  shopGuide: "Browse categories and collect parts you can afford.",
  shopPlayerMoney: "Player Money",
  shopItemsTitle: "Items",
  shopOwned: "Owned",
  shopPurchase: "Purchase",
  shopSummaryTitle: "Catalog Summary",
  shopTotalParts: "Total Parts",
  shopOwnedParts: "Owned Parts",
  shopMissingParts: "Missing Parts",
  kotoneDebugTitle: "Kotone Index (Debug)",
  kotoneDebugGuide: "All kotone currently loaded from catalog.ts.",
  kotoneDebugTotal: "Total",
  galleryTitle: "Track Gallery",
  galleryEmpty: "No delivered tracks yet. Take requests in Town and compose.",
  galleryTrackPrefix: "Track",
  galleryScore: "Score",
  galleryRank: "Rank",
  galleryProgressTitle: "Delivery Progress",
  galleryDelivered: "Delivered",
  galleryTotalRequests: "Total Requests",
  galleryBackToStreet: "Back to Town"
} as const;

const JA_MESSAGES: Record<keyof typeof EN_MESSAGES, string> = {
  navBrand: "ことねのアトリエ",
  navStreet: "街角",
  navShop: "ショップ",
  navKotone: "Kotone",
  navGallery: "ギャラリー",
  navMoney: "所持金",
  navDay: "日数",
  navWeather: "天気",
  langJapanese: "日本語",
  langEnglish: "English",
  streetWaitingTitle: "街角で依頼待ち",
  streetLead: "街角の空気を読みながら、客の抽象的な要望をおとことばに分解して届ける。",
  streetGoToShop: "ショップへ",
  streetOpenGallery: "ギャラリーを開く",
  streetResetSave: "セーブ初期化",
  streetStartNextDay: "次の日へ",
  commissionPanelTitle: "街角の依頼",
  commissionSelect: "選択",
  commissionInterpret: "解釈",
  commissionCompose: "作曲",
  composeTitle: "作曲",
  composeNotFound: "依頼が見つかりません。",
  composeCustomerTitle: "依頼主",
  composeRunInterpreter: "解釈を実行",
  composeHintPrefix: "ヒント",
  composePanelTitle: "作曲パネル",
  composePartsPanelTitle: "所持Kotone",
  composePartsSlotPrefix: "選択中スロット",
  composePartsHint: "上のスロットを選び、この一覧からセットするKotoneを選択してください。",
  composeSelectPlaceholder: "-- 選択 --",
  composeRecommendedPrefix: "推奨",
  composeNeedInterpreter: "解釈を実行すると推奨が表示されます。",
  composeNoPartsInSlot: "このスロットに対応する所持Kotoneがありません。",
  composePreviewPhrase: "試聴フレーズ",
  composePreviewing: "試聴中...",
  composeAndPlay: "作曲して再生",
  composeReset: "リセット",
  composeResultTitle: "結果",
  composeScoreLabel: "スコア",
  composeRankLabel: "ランク",
  composeRewardLabel: "報酬",
  composeMusicJobLabel: "生成ジョブ",
  shopTitle: "音材店",
  shopGuide: "カテゴリを横断して、今の所持金で買えるパーツを収集できます。",
  shopPlayerMoney: "所持金",
  shopItemsTitle: "アイテム",
  shopOwned: "所持済み",
  shopPurchase: "購入",
  shopSummaryTitle: "図鑑サマリー",
  shopTotalParts: "総パーツ数",
  shopOwnedParts: "所持パーツ数",
  shopMissingParts: "未所持パーツ数",
  kotoneDebugTitle: "Kotone一覧（デバッグ）",
  kotoneDebugGuide: "catalog.ts から読み込まれている全Kotoneです。",
  kotoneDebugTotal: "総数",
  galleryTitle: "納品曲ギャラリー",
  galleryEmpty: "まだ納品された曲がありません。街角で依頼を受けて作曲してください。",
  galleryTrackPrefix: "トラック",
  galleryScore: "スコア",
  galleryRank: "ランク",
  galleryProgressTitle: "納品進捗",
  galleryDelivered: "納品済み",
  galleryTotalRequests: "依頼総数",
  galleryBackToStreet: "街角へ戻る"
};

export type MessageKey = keyof typeof EN_MESSAGES;

const WEATHER_LABELS_BY_LOCALE: Record<Locale, Record<Weather, string>> = {
  en: {
    sunny: "SUNNY",
    cloudy: "CLOUDY",
    rainy: "RAINY"
  },
  ja: {
    sunny: "晴れ",
    cloudy: "くもり",
    rainy: "雨"
  }
};

const SLOT_LABELS_BY_LOCALE: Record<Locale, Record<SlotKey, string>> = {
  en: {
    style: "STYLE",
    instrument: "INSTRUMENT",
    mood: "MOOD",
    gimmick: "GIMMICK"
  },
  ja: {
    style: "スタイル",
    instrument: "楽器",
    mood: "ムード",
    gimmick: "ギミック"
  }
};

const COMMISSION_STATUS_LABELS_BY_LOCALE: Record<Locale, Record<CommissionStatus, string>> = {
  en: {
    queued: "QUEUED",
    mixing: "MIXING",
    generating: "GENERATING",
    delivered: "DELIVERED"
  },
  ja: {
    queued: "待機中",
    mixing: "調整中",
    generating: "生成中",
    delivered: "納品済み"
  }
};

const JOB_STATUS_LABELS_BY_LOCALE: Record<Locale, Record<MusicJobStatus, string>> = {
  en: {
    queued: "QUEUED",
    running: "RUNNING",
    done: "DONE",
    failed: "FAILED"
  },
  ja: {
    queued: "待機中",
    running: "実行中",
    done: "完了",
    failed: "失敗"
  }
};

export const isLocale = (value: string): value is Locale => SUPPORTED_LOCALES.includes(value as Locale);

export const t = (locale: Locale, key: MessageKey): string =>
  locale === "ja" ? JA_MESSAGES[key] : EN_MESSAGES[key];

export const weatherLabel = (locale: Locale, weather: Weather): string => WEATHER_LABELS_BY_LOCALE[locale][weather];

export const slotLabel = (locale: Locale, slot: SlotKey): string => SLOT_LABELS_BY_LOCALE[locale][slot];

export const commissionStatusLabel = (locale: Locale, status: CommissionStatus): string =>
  COMMISSION_STATUS_LABELS_BY_LOCALE[locale][status];

export const jobStatusLabel = (locale: Locale, status: MusicJobStatus): string => JOB_STATUS_LABELS_BY_LOCALE[locale][status];
