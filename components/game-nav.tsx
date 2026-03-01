"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { WEATHER_ICON } from "@/lib/catalog";
import { useDemoControls } from "@/components/demo-controls-context";
import { useGame } from "@/components/game-context";
import { useLocale } from "@/components/locale-context";
import { LanguageToggle } from "@/components/language-toggle";
import { weatherLabel } from "@/lib/i18n";

const NAV_ITEMS = [
  { href: "/game/street", key: "navStreet" as const },
  { href: "/game/shop", key: "navShop" as const },
  { href: "/game/kotone", key: "navKotone" as const },
  { href: "/game/gallery", key: "navGallery" as const }
];

export const GameNav = () => {
  const pathname = usePathname();
  const { state } = useGame();
  const { locale, text } = useLocale();
  const { showDemoControls } = useDemoControls();
  const navItems = showDemoControls ? NAV_ITEMS : NAV_ITEMS.filter((item) => item.href !== "/game/kotone");

  return (
    <header className="game-nav">
      <div className="game-nav-brand">
        <span>{text("navBrand")}</span>
      </div>
      <nav className="game-nav-links">
        {navItems.map((item) => (
          <Link
            key={item.href}
            href={item.href}
            className={pathname.startsWith(item.href) ? "active" : ""}
            prefetch={false}
          >
            {text(item.key)}
          </Link>
        ))}
      </nav>
      <div className="game-nav-status">
        <LanguageToggle />
        <span className="game-nav-stat game-nav-money">
          {text("navMoney")}: {state?.money ?? "-"}G
        </span>
        <span className="game-nav-stat game-nav-day">
          {text("navDay")}: {state?.day ?? "-"}
        </span>
        <span className="game-nav-stat game-nav-weather">
          {text("navWeather")}:{" "}
          {state?.weather ? `${WEATHER_ICON[state.weather]} ${weatherLabel(locale, state.weather)}` : "-"}
        </span>
      </div>
    </header>
  );
};
