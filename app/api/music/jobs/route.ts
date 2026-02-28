import { NextResponse } from "next/server";
import { createMusicRequestSchema } from "@/lib/schemas";
import { createJob } from "@/lib/music-jobs";

export const dynamic = "force-dynamic";

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const parsed = createMusicRequestSchema.safeParse(body);

    if (!parsed.success) {
      return NextResponse.json({ error: parsed.error.flatten() }, { status: 400 });
    }

    const result = createJob(parsed.data);
    return NextResponse.json(result);
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Music job create error" },
      { status: 500 }
    );
  }
}
