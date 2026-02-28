"use client";

import type { ButtonHTMLAttributes, HTMLAttributes, ReactNode } from "react";

export const PixelPanel = ({
  title,
  className = "",
  children,
  ...props
}: HTMLAttributes<HTMLDivElement> & { title?: string; children: ReactNode }) => {
  return (
    <section className={`pixel-panel ${className}`} {...props}>
      {title ? <header className="pixel-panel-title">{title}</header> : null}
      <div className="pixel-panel-body">{children}</div>
    </section>
  );
};

export const PixelButton = ({ className = "", children, ...props }: ButtonHTMLAttributes<HTMLButtonElement>) => {
  return (
    <button className={`pixel-button ${className}`} {...props}>
      {children}
    </button>
  );
};

export const PixelTag = ({ children }: { children: ReactNode }) => {
  return <span className="pixel-tag">{children}</span>;
};

export const PixelStat = ({ label, value }: { label: string; value: string | number }) => {
  return (
    <div className="pixel-stat">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
};
