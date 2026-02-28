import { CompositionWorkbench } from "@/components/composition-workbench";

export default async function ComposePage({
  params,
  searchParams
}: {
  params: Promise<{ commissionId: string }>;
  searchParams: Promise<{ cast?: string }>;
}) {
  const { commissionId } = await params;
  const { cast } = await searchParams;

  return <CompositionWorkbench commissionId={commissionId} streetCastId={cast} />;
}
