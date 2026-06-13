"use client";

import { useState, useRef, useEffect } from "react";
import React from "react";

// ---------------------------------------------------------------------------
// Fixed Liquid Glass Carousel - Precise alignment, smooth interaction
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
  const [dragStartX, setDragStartX] = useState(0);
  const [dragOffset, setDragOffset] = useState(0);
  const animationRef = useRef<number>();

  // Fixed dimensions
  const ITEM_WIDTH = 140;
  const ITEM_GAP = 16;
  const TOTAL_WIDTH = ITEM_WIDTH + ITEM_GAP;

  // Calculate target position for current index
  const getTargetPosition = (index: number) => {
    return -index * TOTAL_WIDTH;
  };

  const [position, setPosition] = useState(getTargetPosition(currentIndex));

  // Update position when index changes externally
  useEffect(() => {
    if (!isDragging) {
      animateToPosition(getTargetPosition(currentIndex));
    }
  }, [currentIndex, isDragging]);

  // Smooth animation to target position
  const animateToPosition = (targetPos: number) => {
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
    }

    const animate = () => {
      setPosition((current) => {
        const diff = targetPos - current;
        if (Math.abs(diff) < 0.5) {
          return targetPos;
        }
        return current + diff * 0.2; // Smooth easing
      });

      if (Math.abs(position - targetPos) >= 0.5) {
        animationRef.current = requestAnimationFrame(animate);
      }
    };

    animationRef.current = requestAnimationFrame(animate);
  };

  // Find closest index based on position
  const getClosestIndex = (pos: number) => {
    const index = Math.round(-pos / TOTAL_WIDTH);
    return Math.max(0, Math.min(datasets.length - 1, index));
  };

  // Mouse/Touch handlers
  const handleStart = (clientX: number) => {
    setIsDragging(true);
    setDragStartX(clientX);
    setDragOffset(0);
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
    }
  };

  const handleMove = (clientX: number) => {
    if (!isDragging) return;
    
    const offset = clientX - dragStartX;
    setDragOffset(offset);
    
    const newPos = getTargetPosition(currentIndex) + offset;
    setPosition(newPos);
  };

  const handleEnd = () => {
    if (!isDragging) return;
    
    setIsDragging(false);
    
    // Determine new index based on final position
    const newIndex = getClosestIndex(position);
    
    if (newIndex !== currentIndex) {
      onIndexChange(newIndex);
    } else {
      // Snap back to current
      animateToPosition(getTargetPosition(currentIndex));
    }
    
    setDragOffset(0);
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

  // Click to navigate
  const handleItemClick = (index: number) => {
    if (!isDragging && Math.abs(dragOffset) < 5) {
      onIndexChange(index);
    }
  };

  return (
    <div className="relative w-full py-12 overflow-hidden">
      {/* Liquid glass selector (fixed at center) */}
      <div 
        className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 pointer-events-none z-10"
        style={{
          width: ITEM_WIDTH,
          height: 64,
        }}
      >
        <div 
          className="w-full h-full rounded-2xl"
          style={{
            background: "linear-gradient(135deg, rgba(255,255,255,0.18) 0%, rgba(255,255,255,0.08) 100%)",
            backdropFilter: "blur(24px) saturate(180%)",
            WebkitBackdropFilter: "blur(24px) saturate(180%)",
            boxShadow: `
              0 0 0 1px rgba(255,255,255,0.2),
              0 8px 32px rgba(0,0,0,0.15),
              inset 0 1px 0 rgba(255,255,255,0.4),
              inset 0 -1px 0 rgba(0,0,0,0.1)
            `,
            border: "1px solid rgba(255,255,255,0.25)",
          }}
        >
          {/* Inner glow */}
          <div 
            className="absolute inset-0 rounded-2xl"
            style={{
              background: "radial-gradient(circle at 50% 0%, rgba(255,255,255,0.4), transparent 65%)",
              opacity: 0.7,
            }}
          />
          
          {/* Shimmer */}
          <div className="absolute inset-0 rounded-2xl overflow-hidden">
            <div 
              className="absolute inset-0"
              style={{
                background: "linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent)",
                animation: "shimmer 2.5s infinite",
              }}
            />
          </div>
        </div>
      </div>

      {/* Scrollable items */}
      <div
        ref={containerRef}
        className="relative h-16 select-none"
        style={{
          cursor: isDragging ? "grabbing" : "grab",
        }}
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
            transition: isDragging ? "none" : "transform 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
            gap: `${ITEM_GAP}px`,
          }}
        >
          {datasets.map((dataset, idx) => {
            const isCenter = idx === currentIndex;
            const distance = Math.abs(idx - currentIndex);
            
            // Fixed opacity and scale based on distance
            const opacity = distance === 0 ? 1.0 : distance === 1 ? 0.6 : 0.35;
            const scale = distance === 0 ? 1.0 : distance === 1 ? 0.88 : 0.78;

            return (
              <div
                key={idx}
                className="flex-shrink-0"
                style={{
                  width: ITEM_WIDTH,
                  opacity,
                  transform: `scale(${scale})`,
                  transition: isDragging ? "none" : "opacity 0.3s, transform 0.3s",
                }}
                onClick={() => handleItemClick(idx)}
              >
                <div
                  className={`
                    h-16 rounded-xl flex items-center justify-center
                    font-medium text-sm transition-colors cursor-pointer
                    ${isCenter 
                      ? "text-accent" 
                      : "text-muted hover:text-ink"
                    }
                  `}
                  style={{
                    background: isCenter ? "transparent" : "rgba(0,0,0,0.03)",
                  }}
                >
                  <span className="truncate px-2">{dataset}</span>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Counter */}
      <div className="text-center mt-3">
        <p className="text-xs text-faint font-medium">
          Dataset {currentIndex + 1} of {datasets.length}
        </p>
      </div>

      <style jsx>{`
        @keyframes shimmer {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(100%); }
        }
      `}</style>
    </div>
  );
}
