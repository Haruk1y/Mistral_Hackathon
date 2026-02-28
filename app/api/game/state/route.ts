import { NextResponse } from "next/server";
import { getServerState, setServerState } from "@/lib/server-state";

export const dynamic = "force-dynamic";

export async function GET() {
  return NextResponse.json(getServerState());
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    setServerState(body);
    return NextResponse.json({ ok: true as const });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Invalid game state" },
      { status: 400 }
    );
  }
}
