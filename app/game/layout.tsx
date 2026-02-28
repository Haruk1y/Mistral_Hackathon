import type { ReactNode } from "react";
import Link from "next/link";

export default function GameLayout({ children }: { children: ReactNode }) {
  return (
    <main className="game-shell">
      <header className="game-nav">
        <div className="brand">ことねのアトリエ</div>
        <nav className="links">
          <Link href="/game/street">Street</Link>
          <Link href="/game/compose/demo">Compose</Link>
          <Link href="/game/shop">Shop</Link>
          <Link href="/game/gallery">Gallery</Link>
        </nav>
      </header>
      <section className="game-content">{children}</section>
    </main>
  );
}
