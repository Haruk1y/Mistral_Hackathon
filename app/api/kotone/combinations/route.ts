import { NextResponse } from "next/server";
import { getCoveredCombinations } from "@/lib/ft-test-dataset";

export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const inventoryCsv = searchParams.get("inventoryPartIds") || "";
    const inventoryPartIds = inventoryCsv
      .split(",")
      .map((item) => item.trim())
      .filter((item) => item.length > 0);

    const combos = await getCoveredCombinations(inventoryPartIds.length > 0 ? inventoryPartIds : undefined);
    return NextResponse.json({
      count: combos.length,
      combinations: combos
    });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "failed_to_load_kotone_combinations" },
      { status: 500 }
    );
  }
}
