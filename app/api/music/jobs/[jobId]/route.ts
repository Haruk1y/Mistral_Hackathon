import { NextResponse } from "next/server";
import { getJobStatus } from "@/lib/music-jobs";

export const dynamic = "force-dynamic";

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ jobId: string }> }
) {
  const { jobId } = await params;
  const result = getJobStatus(jobId);

  return NextResponse.json(result);
}
