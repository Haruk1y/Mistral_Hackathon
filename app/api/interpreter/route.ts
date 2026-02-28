import { NextResponse } from "next/server";
import { runInterpreter } from "@/lib/interpreter";
import { interpreterRequestSchema } from "@/lib/schemas";

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const parsed = interpreterRequestSchema.safeParse(body);

    if (!parsed.success) {
      return NextResponse.json({ error: parsed.error.flatten() }, { status: 400 });
    }

    const result = await runInterpreter(parsed.data);
    return NextResponse.json(result);
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Interpreter error" },
      { status: 500 }
    );
  }
}
