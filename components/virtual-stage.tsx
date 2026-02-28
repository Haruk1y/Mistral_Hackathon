"use client";

import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { ASSET_MANIFEST } from "@/lib/assets-manifest";

const VIRTUAL_WIDTH = ASSET_MANIFEST.virtualResolution.w;
const VIRTUAL_HEIGHT = ASSET_MANIFEST.virtualResolution.h;

export const VirtualStage = ({ children }: { children: ReactNode }) => {
  const rootRef = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ width: 0, height: 0 });

  useEffect(() => {
    const root = rootRef.current;
    if (!root) return;

    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      setSize({ width: entry.contentRect.width, height: entry.contentRect.height });
    });

    observer.observe(root);

    return () => observer.disconnect();
  }, []);

  const scale = useMemo(() => {
    if (!size.width || !size.height) return 1;

    const raw = Math.min(size.width / VIRTUAL_WIDTH, size.height / VIRTUAL_HEIGHT);
    if (!Number.isFinite(raw) || raw <= 0) return 1;

    return raw;
  }, [size.height, size.width]);

  const stageStyle = useMemo(
    () => ({
      width: `${VIRTUAL_WIDTH}px`,
      height: `${VIRTUAL_HEIGHT}px`,
      transform: `scale(${scale})`
    }),
    [scale]
  );

  return (
    <div ref={rootRef} className="stage-root">
      <div className="stage" style={stageStyle}>
        {children}
      </div>
    </div>
  );
};

export const StageLayer = ({ className = "", children }: { className?: string; children: ReactNode }) => {
  return <div className={`stage-layer ${className}`}>{children}</div>;
};
