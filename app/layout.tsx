import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ことねのアトリエ | Atelier kotone",
  description: "A lightweight prototype for a word-to-music creation game."
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="ja">
      <body>{children}</body>
    </html>
  );
}
