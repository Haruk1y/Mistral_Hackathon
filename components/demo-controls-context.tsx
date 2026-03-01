"use client";

import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from "react";

const DEMO_CONTROLS_STORAGE_KEY = "otokotoba-demo-controls-visible";

type DemoControlsContextValue = {
  showDemoControls: boolean;
};

const DemoControlsContext = createContext<DemoControlsContextValue | null>(null);

const isEditableTarget = (target: EventTarget | null) => {
  if (!(target instanceof HTMLElement)) return false;
  if (target.isContentEditable) return true;

  const tag = target.tagName.toLowerCase();
  return tag === "input" || tag === "textarea" || tag === "select";
};

export const DemoControlsProvider = ({ children }: { children: ReactNode }) => {
  const [showDemoControls, setShowDemoControls] = useState(true);

  useEffect(() => {
    try {
      const raw = window.localStorage.getItem(DEMO_CONTROLS_STORAGE_KEY);
      if (raw === "true") setShowDemoControls(true);
      if (raw === "false") setShowDemoControls(false);
    } catch {
      // ignore local storage errors
    }
  }, []);

  useEffect(() => {
    try {
      window.localStorage.setItem(DEMO_CONTROLS_STORAGE_KEY, showDemoControls ? "true" : "false");
    } catch {
      // ignore local storage errors
    }
  }, [showDemoControls]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.repeat) return;
      if (isEditableTarget(event.target)) return;
      if (event.key.toLowerCase() !== "d") return;
      if (!event.shiftKey || event.metaKey || event.ctrlKey || event.altKey) return;

      event.preventDefault();
      setShowDemoControls((prev) => !prev);
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  const value = useMemo(() => ({ showDemoControls }), [showDemoControls]);
  return <DemoControlsContext.Provider value={value}>{children}</DemoControlsContext.Provider>;
};

export const useDemoControls = () => {
  const context = useContext(DemoControlsContext);
  if (!context) {
    throw new Error("useDemoControls must be used within DemoControlsProvider");
  }

  return context;
};
