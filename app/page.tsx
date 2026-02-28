import Link from "next/link";

export default function HomePage() {
  return (
    <main className="page">
      <section className="card">
        <h1>ことねのアトリエ (Atelier kotone)</h1>
        <p>
          言葉を選んで音楽の方向性を組み立てる、軽量プロトタイプです。
          まずはゲーム導線だけを最小構成で実装しています。
        </p>
        <Link className="button" href="/game/street">
          Enter Prototype
        </Link>
      </section>
    </main>
  );
}
