"use client";

import { useState, useRef, useEffect } from "react";
import React from "react";

// ---------------------------------------------------------------------------
// Liquid Glass Carousel V2 - With momentum and precise alignment
// Glass frames the actual dataset name, content syncs below
// ---------------------------------------------------------------------------

interface DatasetCarouselProps {
  datasets: string[];
  currentIndex: number;
  onIndexChange: (index: number) => void;
}

export function DatasetCarousel({
  datasets,
  currentIndex,
  onIndexChange,
}: DatasetCarouselProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [position, setPosition] = useState(0);
  const [velocity, setVelocity] = useState(0);
  
  const dragStartX = useRef(0);
  const lastDragX = useRef(0);
  const lastDragTime = useRef(0);
  const animationFrame = useRef<number>();

  const ITEM_WIDTH = 160;
  const ITEM_GAP = 20;
  const TOTAL_WIDTH = ITEM_WIDTH + ITEM_GAP;

  // Calculate target position for an index
  const getTargetPosition = (index: number) => {
    return -index * TOTAL_WIDTH;
  };

  // Get current index from position
  const getCurrentIndex = (pos: number) => {
    const idx = Math.round(-pos / TOTAL_WIDTH);
    return Math.max(0, Math.min(datasets.length - 1, idx));
  };

  // Initialize position
  useEffect(() => {
    setPosition(getTargetPosition(currentIndex));
  }, []);

  // Momentum animation
  useEffect(() => {
    if (isDragging || velocity === 0) return;

    const animate = () => {
      setVelocity(v => {
        const newV = v * 0.95; // Friction
        if (Math.abs(newV) < 0.5) {
          // Velocity too low, snap to nearest
          const targetIdx = getCurrentIndex(position);
          if (targetIdx !== currentIndex) {
            onIndexChange(targetIdx);
          }
          snapToTarget(targetIdx);
          return 0;
        }
        return newV;
      });

      setPosition(p => p + velocity);

      // Check if we crossed into a new dataset
      const newIdx = getCurrentIndex(position);
      if (newIdx !== currentIndex) {
        onIndexChange(newIdx);
      }

      if (Math.abs(velocity) >= 0.5) {
        animationFrame.current = requestAnimationFrame(animate);
      }
    };

    animationFrame.current = requestAnimationFrame(animate);

    return () => {
      if (animationFrame.current) {
        cancelAnimationFrame(animationFrame.current);
      }
    };
  }, [velocity, isDragging, position]);

  // Snap to target index
  const snapToTarget = (targetIdx: number) => {
    const targetPos = getTargetPosition(targetIdx);
    
    const snap = () => {
      setPosition(p => {
        const diff = targetPos - p;
        if (Math.abs(diff) < 0.5) {
          return targetPos;
        }
        return p + diff * 0.15;
      });

      if (Math.abs(position - targetPos) >= 0.5) {
        requestAnimationFrame(snap);
      }
    };

    requestAnimationFrame(snap);
  };

  // Drag handlers
  const handleStart = (clientX: number) => {
    setIsDragging(true);
    setVelocity(0);
    dragStartX.current = clientX;
    lastDragX.current = clientX;
    lastDragTime.current = Date.now();

    if (animationFrame.current) {
      cancelAnimationFrame(animationFrame.current);
    }
  };

  const handleMove = (clientX: number) => {
    if (!isDragging) return;

    const now = Date.now();
    const deltaTime = now - lastDragTime.current;
    const deltaX = clientX - lastDragX.current;

    if (deltaTime > 0) {
      // Calculate velocity (pixels per ms)
      setVelocity(deltaX / deltaTime * 16); // Convert to pixels per frame (60fps)
    }

    const totalDelta = clientX - dragStartX.current;
    const basePos = getTargetPosition(currentIndex);
    setPosition(basePos + totalDelta);

    lastDragX.current = clientX;
    lastDragTime.current = now;

    // Update current index as we drag
    const newIdx = getCurrentIndex(basePos + totalDelta);
    if (newIdx !== currentIndex) {
      onIndexChange(newIdx);
    }
  };

  const handleEnd = () => {
    if (!isDragging) return;
    setIsDragging(false);

    // If velocity is too low, snap immediately
    if (Math.abs(velocity) < 2) {
      const targetIdx = getCurrentIndex(position);
      if (targetIdx !== currentIndex) {
        onIndexChange(targetIdx);
      }
      snapToTarget(targetIdx);
      setVelocity(0);
    }
    // Otherwise let momentum take over
  };

  // Mouse events
  const onMouseDown = (e: React.MouseEvent) => {
    e.preventDefault();
    handleStart(e.clientX);
  };

  const onMouseMove = (e: React.MouseEvent) => {
    handleMove(e.clientX);
  };

  const onMouseUp = () => {
    handleEnd();
  };

  const onMouseLeave = () => {
    if (isDragging) {
      handleEnd();
    }
  };

  // Touch events
  const onTouchStart = (e: React.TouchEvent) => {
    handleStart(e.touches[0].clientX);
  };

  const onTouchMove = (e: React.TouchEvent) => {
    handleMove(e.touches[0].clientX);
  };

  const onTouchEnd = () => {
    handleEnd();
  };

  return (
    <div className="relative w-full py-10 overflow-hidden bg-gradient-to-b from-transparent via-paper-2/30 to-transparent">
      {/* Liquid glass selector - frames the dataset name */}
      <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 pointer-events-none z-20">
        <div 
          className="relative rounded-2xl"
          style={{
            width: ITEM_WIDTH,
            height: 60,
            background: "linear-gradient(135deg, rgba(255,255,255,0.2) 0%, rgba(255,255,255,0.1) 100%)",
            backdropFilter: "blur(24px) saturate(180%)",
            WebkitBackdropFilter: "blur(24px) saturate(180%)",
            boxShadow: `
              0 0 0 1.5px rgba(255,255,255,0.25),
              0 10px 40px rgba(0,0,0,0.12),
              inset 0 1px 0 rgba(255,255,255,0.5),
              inset 0 -1px 0 rgba(0,0,0,0.1)
            `,
            border: "1px solid rgba(255,255,255,0.3)",
          }}
        >
          <div 
            className="absolute inset-0 rounded-2xl"
            style={{
              background: "radial-gradient(circle at 50% 0%, rgba(255,255,255,0.5), transparent 70%)",
              opacity: 0.8,
            }}
          />
          
          <div className="absolute inset-0 rounded-2xl overflow-hidden">
            <div 
              style={{
                position: "absolute",
                inset: 0,
                background: "linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent)",
                animation: "shimmer 2.5s ease-in-out infinite",
              }}
            />
          </div>
        </div>
      </div>

      {/* Draggable strip */}
      <div
        className="relative h-20 select-none"
        style={{ cursor: isDragging ? "grabbing" : "grab" }}
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onMouseLeave={onMouseLeave}
        onTouchStart={onTouchStart}
        onTouchMove={onTouchMove}
        onTouchEnd={onTouchEnd}
      >
        <div
          className="absolute left-1/2 top-1/2 -translate-y-1/2 flex"
          style={{
            transform: `translate(-50%, -50%) translateX(${position}px)`,
            gap: `${ITEM_GAP}px`,
            willChange: "transform",
          }}
        >
          {datasets.map((dataset, idx) => {
            const distance = Math.abs(idx - currentIndex);
            const isCenter = idx === currentIndex;
            
            const opacity = isCenter ? 1.0 : distance === 1 ? 0.5 : 0.25;
            const scale = isCenter ? 1.0 : distance === 1 ? 0.85 : 0.75;

            return (
              <div
                key={idx}
                className="flex-shrink-0"
                style={{
                  width: ITEM_WIDTH,
                  opacity,
                  transform: `scale(${scale})`,
                  transition: "opacity 0.2s, transform 0.2s",
                }}
              >
                <div
                  className={`
                    h-20 rounded-2xl flex items-center justify-center px-4
                    font-semibold text-base transition-all
                    ${isCenter 
                      ? "text-accent" 
                      : "text-muted"
                    }
                  `}
                >
                  <span className="truncate text-center">{dataset}</span>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Position indicator */}
      <div className="text-center mt-2">
        <p className="text-xs text-faint font-medium">
          {currentIndex + 1} / {datasets.length}
        </p>
      </div>

      <style jsx>{`
        @keyframes shimmer {
          0%, 100% { transform: translateX(-100%); opacity: 0; }
          50% { transform: translateX(100%); opacity: 1; }
        }
      `}</style>
    </div>
  );
}
