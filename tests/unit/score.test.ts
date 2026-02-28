import { REQUEST_TEMPLATES } from "@/lib/catalog";
import { calculateRank, ensureSlotCategoryIntegrity, scoreComposition } from "@/lib/score";
import type { SlotKey } from "@/lib/types";

const validSelection: Record<SlotKey, string> = {
  style: "style_80s_citypop",
  instrument: "inst_piano_upright",
  mood: "mood_rain_ambience",
  gimmick: "gimmick_beat_mute"
};

describe("scoreComposition", () => {
  it("returns score within range", () => {
    const result = scoreComposition(validSelection, REQUEST_TEMPLATES[0].targetProfile);
    expect(result.total).toBeGreaterThanOrEqual(0);
    expect(result.total).toBeLessThanOrEqual(100);
  });

  it("maps rank thresholds correctly", () => {
    expect(calculateRank(95)).toBe("S");
    expect(calculateRank(80)).toBe("A");
    expect(calculateRank(61)).toBe("B");
    expect(calculateRank(40)).toBe("C");
    expect(calculateRank(39)).toBe("D");
  });

  it("validates slot category integrity", () => {
    const errors = ensureSlotCategoryIntegrity({
      ...validSelection,
      style: "inst_piano_upright"
    });

    expect(errors.length).toBeGreaterThan(0);
  });
});
