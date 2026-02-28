"use client";

import { PixelButton } from "@/components/pixel-ui";
import { useLocale } from "@/components/locale-context";

export const LanguageToggle = () => {
  const { locale, setLocale, text } = useLocale();

  return (
    <div className="lang-toggle" aria-label="language toggle">
      <PixelButton type="button" className={locale === "ja" ? "is-active" : ""} onClick={() => setLocale("ja")}>
        {text("langJapanese")}
      </PixelButton>
      <PixelButton type="button" className={locale === "en" ? "is-active" : ""} onClick={() => setLocale("en")}>
        {text("langEnglish")}
      </PixelButton>
    </div>
  );
};
