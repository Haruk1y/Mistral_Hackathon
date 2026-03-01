# Atelier Kotone

![Atelier Kotone Slide 1](./public/assets/Atelier%20Kotone/page1_1.png)

Atelier Kotone is a music-creation game where you compose songs by combining modular sound elements called Kotone.

Players take on commissions from townspeople, each with abstract and emotional requests. By selecting and blending Kotone — such as style, instrument, mood, and special effects — you craft a piece of music that matches the client’s vision.

Earn money from satisfied customers, unlock new Kotone, expand your creative palette, and build your personal music gallery.

Atelier Kotone turns music creation into an accessible, playful experience — where imagination becomes sound.

## Technology Highlights: Mistral + ElevenLabs + W&B

This project is centered on three external AI components that shape both the product experience and model iteration workflow:

| Technology | What it does in Atelier Kotone | Main implementation points |
|---|---|---|
| **Mistral (Mistral Large 3 + Ministral 3B)** | Generates teacher data, powers distillation, and runs hidden-parameter interpretation with a **fine-tuned Ministral 3B model** during gameplay | `scripts/ft/generate_teacher_pairs.mjs`, `scripts/hf/train_sft_request_to_hidden_lm.py`, `lib/interpreter.ts` |
| **ElevenLabs API** | Generates final music from player-selected Kotone combinations in the runtime loop | `lib/music-provider.ts`, `lib/music-jobs.ts` |
| **Weights & Biases (W&B + Weave + MCP)** | Tracks evaluation/training, stores failure cases, and drives self-improvement planning across cycles | `scripts/wandb/weave_eval_runner.py`, `scripts/wandb/fetch_mcp_eval_context.mjs`, `scripts/loop/run_self_improvement_cycle.mjs` |

## Core Concept

![Atelier Kotone Slide 2](./public/assets/Atelier%20Kotone/page2_1.png)
![Atelier Kotone Slide 3](./public/assets/Atelier%20Kotone/page3_1.png)

Music prompting is still hard for beginners: people often know the feeling they want, but not how to express it in prompt language.

Atelier Kotone solves this by turning music intent into modular creative choices.  
Instead of writing complex prompts directly, players compose with Kotone parts:

- Style
- Instrument
- Mood
- Gimmick

This makes AI music creation approachable without requiring technical expertise.

![Atelier Kotone Slide 4](./public/assets/Atelier%20Kotone/page4_1.png)
![Atelier Kotone Slide 5](./public/assets/Atelier%20Kotone/page5_1.png)

## How It Plays

1. Accept a commission from a townsperson.
2. Read an emotional, abstract request.
3. Combine Kotone parts to shape the song.
4. Deliver the result and get rewarded.
5. Unlock more Kotone and expand your options.
6. Build your personal gallery of finished music.

## Hidden Target Design

Each commission has hidden target parameters behind the scenes.  
Players never see these values directly. The challenge is to infer the intent from the request and find the Kotone combination that best matches it.

This creates a playful loop of experimentation, feedback, and improvement.

## Technical Game Loop
![Atelier Kotone Slide 6](./public/assets/Atelier%20Kotone/page6_1.png)

At runtime, the game follows this flow:

1. A commission request is generated/interpreted by **Ministral 3B**.
2. The same model defines hidden target feature parameters for that request.
3. The player selects Kotone parts and builds a composition prompt.
4. The prompt is sent to **ElevenLabs** to generate music.
5. **Ministral 3B** estimates the resulting feature parameters from the selected Kotone/prompt.
6. The game compares target vs. estimated parameters and returns final score + feedback.

Hidden feature dimensions:

- Energy
- Warmth
- Brightness
- Acousticness
- Complexity
- Nostalgia

## AI Layer (Distillation Strategy)

- Teacher data is generated with **Mistral 3 Large**, then distilled into **Ministral 3B** to target both quality and low latency.
- We **fine-tuned** the distilled Ministral 3B model on our task-specific dataset so it can estimate hidden music parameters for gameplay evaluation.
- The reason for distillation is practical gameplay: lower inference cost and faster response while keeping enough semantic quality.
- In this prototype, some outputs are pre-generated due to current hosting constraints around local/real-time inference.
- The intended full experience is real-time generation of random commissions and hidden targets with the distilled model.
- Final music output is generated with the **ElevenLabs API** based on player-selected Kotone.

## Training Pipeline (Distillation + Self-Improve)
![Atelier Kotone Slide 7](./public/assets/Atelier%20Kotone/page7_1.png)

1. Generate high-quality teacher data with **Mistral Large 3**.
2. Fine-tune/distill into **Ministral 3B**.
3. Run evaluation in **Weights & Biases**.
4. Use **W&B MCP** to inspect loss/failures and propose hyperparameter updates.
5. Generate additional data for weak cases and retrain.

This loop is repeated to balance quality, speed, and cost for game-ready inference.

## Run Locally

### Prerequisites

- Node.js 20+ (Node.js 18.18+ also works with Next.js 15)
- npm

### 1) Install

```bash
npm install
```

### 2) Create local env file

```bash
cp .env.example .env.local
```

### 3) Choose runtime mode

#### Demo mode (no external API keys)

Use dataset/rule-based interpretation and fallback audio so the game can run without cloud keys.

```bash
INTERPRETER_BACKEND=dataset ELEVENLABS_ALLOW_FALLBACK_AUDIO=true npm run dev
```

#### Full mode (with real model/music APIs)

Set at least:

- `HF_TOKEN` (or `HF_API_TOKEN`)
- `ELEVENLABS_API_KEY`

Recommended when running the full training/eval loop:

- `MISTRAL_API_KEY` (teacher data generation)
- `WANDB_API_KEY` (W&B/Weave/MCP logging and analysis)

Then run:

```bash
npm run dev
```

### 4) Open the app

Open `http://localhost:3000`  
The root route redirects to `/game/street`.

### Optional checks

```bash
# unit tests
npm test

# evaluation pipeline
npm run eval:run
npm run eval:aggregate
```

## Vision

Just as we learn languages like English or Japanese to express ourselves, prompting will become a core literacy for the AI era.

Atelier Kotone is designed as a playful first step into that future through creativity, curiosity, and music.
