import { resolve } from "node:path";
import { generateRequestText } from "@/lib/request-generator";
import { runInterpreter } from "@/lib/interpreter";
import { getCoveredCombinations, isCombinationCovered, pickRequestSampleFromTestDataset } from "@/lib/ft-test-dataset";

const fixturePath = resolve(process.cwd(), "tests/fixtures/ft_request_param_test.mini.jsonl");

const originalEnv = {
  REQUEST_GENERATION_BACKEND: process.env.REQUEST_GENERATION_BACKEND,
  INTERPRETER_BACKEND: process.env.INTERPRETER_BACKEND,
  KOTONE_REFERENCE_DATASET_PATHS: process.env.KOTONE_REFERENCE_DATASET_PATHS,
  KOTONE_REFERENCE_DATASET_PATH: process.env.KOTONE_REFERENCE_DATASET_PATH,
  HF_TOKEN: process.env.HF_TOKEN,
  HF_API_TOKEN: process.env.HF_API_TOKEN,
  MISTRAL_FINE_TUNED_MODEL_ID: process.env.MISTRAL_FINE_TUNED_MODEL_ID
};

beforeEach(() => {
  delete process.env.KOTONE_REFERENCE_DATASET_PATHS;
  process.env.KOTONE_REFERENCE_DATASET_PATH = fixturePath;
  delete process.env.HF_TOKEN;
  delete process.env.HF_API_TOKEN;
  delete process.env.MISTRAL_FINE_TUNED_MODEL_ID;
});

afterAll(() => {
  process.env.REQUEST_GENERATION_BACKEND = originalEnv.REQUEST_GENERATION_BACKEND;
  process.env.INTERPRETER_BACKEND = originalEnv.INTERPRETER_BACKEND;
  process.env.KOTONE_REFERENCE_DATASET_PATHS = originalEnv.KOTONE_REFERENCE_DATASET_PATHS;
  process.env.KOTONE_REFERENCE_DATASET_PATH = originalEnv.KOTONE_REFERENCE_DATASET_PATH;
  process.env.HF_TOKEN = originalEnv.HF_TOKEN;
  process.env.HF_API_TOKEN = originalEnv.HF_API_TOKEN;
  process.env.MISTRAL_FINE_TUNED_MODEL_ID = originalEnv.MISTRAL_FINE_TUNED_MODEL_ID;
});

describe("dataset-backed runtime fallback", () => {
  it("samples request_text and returns hidden params in dataset mode", async () => {
    process.env.REQUEST_GENERATION_BACKEND = "dataset";

    const generated = await generateRequestText({
      templateText: "template sentence",
      weather: "rainy",
      customerId: "customer_a",
      customerName: "A",
      customerPersonality: "calm"
    });

    expect([
      "Could you make a warm lane melody for evening rain?",
      "I need a bright market tune with a playful pulse."
    ]).toContain(generated.requestText);
    expect(generated.targetHiddenParams?.vector).toBeDefined();
    expect(generated.targetHiddenParams?.vector.warmth).toBeGreaterThanOrEqual(0);
    expect(generated.targetHiddenParams?.vector.warmth).toBeLessThanOrEqual(100);
  });

  it("predicts hidden params from rule prompt combo via dataset lookup", async () => {
    process.env.INTERPRETER_BACKEND = "dataset";

    const response = await runInterpreter({
      requestText:
        "RULE_PROMPT:\nCompose nostalgic retro pixel-town background music.\nReturn instrumental music suitable for a game scene.\nThis is a rule-based prompt generated from selected Kotone parts.\nSelected Kotone combination:\n- style: 80s City Pop (style_80s_citypop)\n- instrument: Upright Piano (inst_piano_upright)\n- mood: Rain Ambience (mood_rain_ambience)\n- gimmick: Beat Mute (gimmick_beat_mute)\nStyle: warm, cozy, handcrafted, street evening, non-vocal, emotional but simple.",
      weather: "rainy",
      inventoryPartIds: ["style_80s_citypop", "inst_piano_upright", "mood_rain_ambience", "gimmick_beat_mute"]
    });

    expect(response.targetHiddenParams.vector.energy).toBe(20);
    expect(response.targetHiddenParams.vector.warmth).toBe(90);
    expect(response.targetHiddenParams.vector.nostalgia).toBe(90);
    expect(response.rationale.join(" ")).toContain("dataset_lookup_strategy=rule_combo_exact");
  });

  it("supports deterministic sampling with seed hint", async () => {
    const first = await pickRequestSampleFromTestDataset("seed-a");
    const second = await pickRequestSampleFromTestDataset("seed-a");
    const third = await pickRequestSampleFromTestDataset("seed-b");

    expect(first?.requestText).toBe(second?.requestText);
    expect(first?.requestText).not.toBeUndefined();
    expect(third?.requestText).not.toBeUndefined();
  });

  it("checks exact combo coverage from dataset", async () => {
    const combos = await getCoveredCombinations();
    expect(combos.length).toBeGreaterThan(0);

    const covered = await isCombinationCovered({
      style: "style_80s_citypop",
      instrument: "inst_piano_upright",
      mood: "mood_rain_ambience",
      gimmick: "gimmick_beat_mute"
    });
    const uncovered = await isCombinationCovered({
      style: "style_80s_citypop",
      instrument: "inst_piano_upright",
      mood: "mood_sun_glow",
      gimmick: "gimmick_beat_mute"
    });

    expect(covered).toBe(true);
    expect(uncovered).toBe(false);
  });
});
