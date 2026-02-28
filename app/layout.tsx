import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ことねのアトリエ | Atelier kotone",
  description: "Retro pixel-art composition game prototype with placeholder assets."
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="ja">
      <body>{children}</body>
    </html>
  );
}
