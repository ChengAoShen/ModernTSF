"use client";

import { useState, useRef, useEffect } from "react";
import React from "react";

// ---------------------------------------------------------------------------
// Liquid Glass Carousel - Mac-style dataset navigation
// Draggable infinite loop with automatic content switching
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
  const containerRef = useRef<HTMLDivElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [startX, setStartX] = useState(0);
  const [scrollOffset, setScrollOffset] = useState(0);
  const [dragVelocity, setDragVelocity] = useState(0);
  const lastDragTime = useRef(Date.now());
  const lastDragX = useRef(0);

  // Item width and spacing
  const ITEM_WIDTH = 120;
  const ITEM_SPACING = 20;
  const TOTAL_ITEM_WIDTH = ITEM_WIDTH + ITEM_SPACING;

  // Create infinite loop by triplicating the dataset
  const infiniteDatasets = [...datasets, ...datasets, ...datasets];
  const centerOffset = datasets.length * TOTAL_ITEM_WIDTH;

  // Normalize offset to keep within one cycle
  const normalizeOffset = (offset: number) => {
    const cycleWidth = datasets.length * TOTAL_ITEM_WIDTH;
    let normalized = offset % cycleWidth;
    if (normalized < -cycleWidth / 2) normalized += cycleWidth;
    if (normalized > cycleWidth / 2) normalized -= cycleWidth;
    return normalized;
  };

  // Calculate which dataset is in the center
  const getCurrentDatasetIndex = (offset: number) => {
    const normalizedOffset = normalizeOffset(offset);
    const index = Math.round(-normalizedOffset / TOTAL_ITEM_WIDTH);
    return ((index % datasets.length) + datasets.length) % datasets.length;
  };

  // Handle drag start
  const handleDragStart = (clientX: number) => {
    setIsDragging(true);
    setStartX(clientX);
    lastDragTime.current = Date.now();
    lastDragX.current = clientX;
    setDragVelocity(0);
  };

  // Handle drag move
  const handleDragMove = (clientX: number) => {
    if (!isDragging) return;

    const now = Date.now();
    const timeDelta = now - lastDragTime.current;
    const distance = clientX - lastDragX.current;

    if (timeDelta > 0) {
      setDragVelocity(distance / timeDelta * 16); // pixels per frame at 60fps
    }

    const delta = clientX - startX;
    const newOffset = normalizeOffset(scrollOffset + delta);
    setScrollOffset(newOffset);
    setStartX(clientX);

    lastDragTime.current = now;
    lastDragX.current = clientX;

    // Update current index as we drag
    const newIndex = getCurrentDatasetIndex(newOffset);
    if (newIndex !== currentIndex) {
      onIndexChange(newIndex);
    }
  };

  // Handle drag end with momentum
  const handleDragEnd = () => {
    setIsDragging(false);

    // Apply momentum scrolling
    if (Math.abs(dragVelocity) > 0.5) {
      applyMomentum(dragVelocity);
    } else {
      // Snap to nearest item
      snapToNearest();
    }
  };

  // Apply momentum scrolling
  const applyMomentum = (initialVelocity: number) => {
    let velocity = initialVelocity;
    const friction = 0.92; // Friction coefficient
    
    const animate = () => {
      velocity *= friction;
      
      if (Math.abs(velocity) < 0.3) {
        snapToNearest();
        return;
      }

      const newOffset = normalizeOffset(scrollOffset + velocity);
      setScrollOffset(newOffset);

      const newIndex = getCurrentDatasetIndex(newOffset);
      if (newIndex !== currentIndex) {
        onIndexChange(newIndex);
      }

      requestAnimationFrame(animate);
    };

    requestAnimationFrame(animate);
  };

  // Snap to nearest dataset
  const snapToNearest = () => {
    const targetIndex = currentIndex;
    const targetOffset = -targetIndex * TOTAL_ITEM_WIDTH;
    
    const animate = () => {
      const current = scrollOffset;
      const diff = targetOffset - current;
      
      if (Math.abs(diff) < 0.5) {
        setScrollOffset(targetOffset);
        return;
      }

      const newOffset = current + diff * 0.15; // Smooth easing
      setScrollOffset(newOffset);
      requestAnimationFrame(animate);
    };

    requestAnimationFrame(animate);
  };

  // Mouse events
  const onMouseDown = (e: React.MouseEvent) => {
    e.preventDefault();
    handleDragStart(e.clientX);
  };

  const onMouseMove = (e: React.MouseEvent) => {
    handleDragMove(e.clientX);
  };

  const onMouseUp = () => {
    handleDragEnd();
  };

  const onMouseLeave = () => {
    if (isDragging) {
      handleDragEnd();
    }
  };

  // Touch events
  const onTouchStart = (e: React.TouchEvent) => {
    handleDragStart(e.touches[0].clientX);
  };

  const onTouchMove = (e: React.TouchEvent) => {
    handleDragMove(e.touches[0].clientX);
  };

  const onTouchEnd = () => {
    handleDragEnd();
  };

  // Click on item to navigate
  const handleItemClick = (index: number) => {
    if (!isDragging) {
      const actualIndex = index % datasets.length;
      onIndexChange(actualIndex);
      
      // Animate to target
      const targetOffset = -actualIndex * TOTAL_ITEM_WIDTH;
      const animate = () => {
        const current = scrollOffset;
        const diff = targetOffset - current;
        
        if (Math.abs(diff) < 0.5) {
          setScrollOffset(targetOffset);
          return;
        }

        const newOffset = current + diff * 0.12;
        setScrollOffset(newOffset);
        requestAnimationFrame(animate);
      };
      animate();
    }
  };

  return (
    <div className="relative w-full py-8 overflow-hidden">
      {/* Liquid glass selector (center highlight) */}
      <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 pointer-events-none z-10">
        <div 
          className="relative rounded-2xl transition-all duration-300"
          style={{
            width: ITEM_WIDTH,
            height: 56,
            background: "linear-gradient(135deg, rgba(255,255,255,0.15) 0%, rgba(255,255,255,0.05) 100%)",
            backdropFilter: "blur(20px) saturate(180%)",
            WebkitBackdropFilter: "blur(20px) saturate(180%)",
            boxShadow: `
              0 0 0 1px rgba(255,255,255,0.1),
              0 8px 32px rgba(0,0,0,0.12),
              inset 0 1px 0 rgba(255,255,255,0.3),
              inset 0 -1px 0 rgba(0,0,0,0.1)
            `,
            border: "1px solid rgba(255,255,255,0.18)",
          }}
        >
          {/* Inner glow */}
          <div 
            className="absolute inset-0 rounded-2xl opacity-60"
            style={{
              background: "radial-gradient(circle at 50% 0%, rgba(255,255,255,0.3), transparent 70%)",
            }}
          />
          
          {/* Shimmer effect */}
          <div 
            className="absolute inset-0 rounded-2xl overflow-hidden"
          >
            <div 
              className="absolute inset-0 animate-shimmer"
              style={{
                background: "linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent)",
                transform: "translateX(-100%)",
                animation: "shimmer 3s infinite",
              }}
            />
          </div>
        </div>
      </div>

      {/* Scrollable dataset strip */}
      <div
        ref={containerRef}
        className="relative h-14 cursor-grab active:cursor-grabbing select-none"
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onMouseLeave={onMouseLeave}
        onTouchStart={onTouchStart}
        onTouchMove={onTouchMove}
        onTouchEnd={onTouchEnd}
      >
        <div
          className="absolute left-1/2 top-1/2 -translate-y-1/2 flex items-center transition-transform"
          style={{
            transform: `translate(-50%, -50%) translateX(${scrollOffset}px)`,
            transitionDuration: isDragging ? "0ms" : "150ms",
            transitionTimingFunction: "cubic-bezier(0.4, 0, 0.2, 1)",
          }}
        >
          {infiniteDatasets.map((dataset, idx) => {
            const actualIndex = idx % datasets.length;
            const isCenter = idx === datasets.length + currentIndex;
            const distance = Math.abs(idx - (datasets.length + currentIndex));
            const opacity = Math.max(0.3, 1 - distance * 0.2);
            const scale = isCenter ? 1 : Math.max(0.75, 1 - distance * 0.1);

            return (
              <div
                key={`${dataset}-${idx}`}
                className="flex-shrink-0 transition-all duration-200"
                style={{
                  width: ITEM_WIDTH,
                  marginRight: ITEM_SPACING,
                  opacity,
                  transform: `scale(${scale})`,
                }}
                onClick={() => handleItemClick(actualIndex)}
              >
                <div
                  className={`
                    h-14 rounded-xl flex items-center justify-center
                    font-medium text-sm transition-all duration-200
                    ${isCenter 
                      ? "text-accent-fg" 
                      : "text-muted hover:text-ink"
                    }
                  `}
                  style={{
                    background: isCenter 
                      ? "transparent"
                      : "rgba(0,0,0,0.02)",
                  }}
                >
                  {dataset}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Dataset counter */}
      <div className="text-center mt-2">
        <p className="text-xs text-faint">
          Dataset {currentIndex + 1} of {datasets.length}
        </p>
      </div>

      {/* Add shimmer animation */}
      <style jsx>{`
        @keyframes shimmer {
          0% {
            transform: translateX(-100%);
          }
          100% {
            transform: translateX(100%);
          }
        }
        .animate-shimmer {
          animation: shimmer 3s infinite;
        }
      `}</style>
    </div>
  );
}
