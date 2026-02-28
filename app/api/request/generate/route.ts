import { NextResponse } from "next/server";
import { generateRequestText } from "@/lib/request-generator";
import { requestGenerationRequestSchema } from "@/lib/schemas";

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const parsed = requestGenerationRequestSchema.safeParse(body);

    if (!parsed.success) {
      return NextResponse.json({ error: parsed.error.flatten() }, { status: 400 });
    }

    const result = await generateRequestText(parsed.data);
    return NextResponse.json(result);
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Request generation error" },
      { status: 500 }
    );
  }
}
