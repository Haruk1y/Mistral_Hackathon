type ComposePageProps = {
  params: Promise<{
    commissionId: string;
  }>;
};

export default async function ComposePage({ params }: ComposePageProps) {
  const { commissionId } = await params;

  return (
    <div className="card">
      <h2>Compose</h2>
      <p>Commission ID: {commissionId}</p>
      <p>言葉スロットで音楽の方向性を選ぶ最小UIです。</p>
      <img className="placeholder" src="/assets/placeholders/compose.svg" alt="Compose placeholder" />
    </div>
  );
}
