import { runInterpreter } from "@/lib/interpreter";
import { interpreterResponseSchema } from "@/lib/schemas";

describe("runInterpreter", () => {
  it("returns schema-valid response", async () => {
    delete process.env.HF_API_TOKEN;

    const response = await runInterpreter({
      requestText: "quiet evening after rain",
      weather: "rainy",
      inventoryPartIds: [
        "style_80s_citypop",
        "style_90s_hiphop",
        "inst_piano_upright",
        "inst_soft_strings",
        "mood_rain_ambience",
        "mood_sun_glow",
        "gimmick_beat_mute"
      ]
    });

    const parsed = interpreterResponseSchema.safeParse(response);
    expect(parsed.success).toBe(true);
  });
});
