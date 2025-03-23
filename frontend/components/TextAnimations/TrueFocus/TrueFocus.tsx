import React, { useState, useEffect } from 'react';

const BlurAnimation = () => {
  const [isBlurred, setIsBlurred] = useState(true);
  const [magnifierPosition, setMagnifierPosition] = useState({ x: 40, y: 40 });
  const [targetPosition, setTargetPosition] = useState({ x: 100, y: 60 });
  const borderColor = "#39FF14"; // Neon green
  const glowColor = "rgba(57, 255, 20, 0.7)"; // Semi-transparent neon green
  
  useEffect(() => {
    const interval = setInterval(() => {
      setIsBlurred(prev => !prev);
    }, 2000); // Toggle every 2 seconds
    
    return () => clearInterval(interval);
  }, []);
  
  // Random magnifying glass movement
  useEffect(() => {
    const moveInterval = setInterval(() => {
      // Generate a new random target position
      const newTargetX = Math.random() * 160 + 40;
      const newTargetY = Math.random() * 60 + 30;
      setTargetPosition({ x: newTargetX, y: newTargetY });
    }, 1000); // Change direction more frequently (every 1 second)
    
    return () => clearInterval(moveInterval);
  }, []);
  
  // Smooth movement towards target position
  useEffect(() => {
    const animationFrame = requestAnimationFrame(function animate() {
      // Calculate the direction to move
      const dx = targetPosition.x - magnifierPosition.x;
      const dy = targetPosition.y - magnifierPosition.y;
      
      // Calculate distance to target
      const distance = Math.sqrt(dx * dx + dy * dy);
      
      // If we're close enough to the target, stop moving
      if (distance < 0.5) {
        return;
      }
      
      // Move 5% of the remaining distance each frame (faster movement)
      const newX = magnifierPosition.x + dx * 0.05;
      const newY = magnifierPosition.y + dy * 0.05;
      
      setMagnifierPosition({ x: newX, y: newY });
      
      // Continue animation
      requestAnimationFrame(animate);
    });
    
    return () => cancelAnimationFrame(animationFrame);
  }, [magnifierPosition, targetPosition]);
  
  return (
    <div className="flex items-center justify-center w-full h-64">
      <div className="relative flex items-center justify-center w-64 h-32 rounded">
        {/* Blurred REAL text */}
        <div 
          className={`absolute text-7xl font-bold transition-all duration-1000 ${isBlurred ? 'opacity-100 blur-sm' : 'opacity-0 blur-none'}`}
          style={{ filter: isBlurred ? 'blur(4px)' : 'blur(0px)' }}
        >
          R E A L
        </div>
        
        {/* Clear FAKE text */}
        <div 
          className={`absolute text-7xl font-bold transition-all duration-1000 ${isBlurred ? 'opacity-0 blur-none' : 'opacity-100 blur-none'}`}
        >
          FAKE
        </div>
        
        {/* Magnifying glass */}
        <div 
          className="absolute pointer-events-none"
          style={{
            left: magnifierPosition.x,
            top: magnifierPosition.y,
            transform: 'translate(-50%, -50%)',
            zIndex: 10
          }}
        >
          <div className="relative">
            {/* Magnifier circle - larger with white border */}
            <div 
              className="rounded-full overflow-hidden flex items-center justify-center"
              style={{ 
                width: '70px', 
                height: '70px',
                border: '4px solid white',
                boxShadow: '0 0 10px rgba(255, 255, 255, 0.7)',
                backgroundColor: 'transparent'
              }}
            >
              {/* Glass lens effect with magnifying look */}
              <div 
                className="absolute w-full h-full rounded-full"
                style={{
                  background: 'radial-gradient(circle, rgba(255,255,255,0.25) 0%, rgba(0,0,0,0.15) 70%, rgba(0,0,0,0.3) 100%)',
                  transform: 'scale(0.9)'
                }}
              ></div>
              
              {/* Highlight on glass */}
              <div className="absolute rounded-full bg-white opacity-40" 
                   style={{ width: '25px', height: '10px', top: '15px', left: '15px', transform: 'rotate(20deg)' }}></div>
            </div>
            
            {/* Thicker handle with white color */}
            <div 
              style={{
                position: 'absolute',
                width: '10px',
                height: '35px',
                backgroundColor: 'white',
                bottom: '-30px',
                right: '15px',
                transform: 'rotate(160deg)',
                borderRadius: '4px',
                boxShadow: '0 0 5px rgba(255, 255, 255, 0.7)'
              }}
            ></div>
          </div>
        </div>
        
        {/* Corner borders container */}
        <div className="absolute inset-0 w-full h-full pointer-events-none">
          {/* Top-left corner */}
          <div className="absolute transition-all duration-1000" style={{
            top: isBlurred ? '-4px' : '4px',
            left: isBlurred ? '-4px' : '4px',
          }}>
            <div style={{
              position: 'absolute',
              width: '20px',
              height: '3px',
              backgroundColor: borderColor,
              boxShadow: `0 0 8px 2px ${glowColor}`,
            }}></div>
            <div style={{
              position: 'absolute',
              width: '3px',
              height: '20px',
              backgroundColor: borderColor,
              boxShadow: `0 0 8px 2px ${glowColor}`,
            }}></div>
          </div>
          
          {/* Top-right corner */}
          <div className="absolute transition-all duration-1000" style={{
            top: isBlurred ? '-4px' : '4px',
            right: isBlurred ? '-4px' : '4px',
          }}>
            <div style={{
              position: 'absolute',
              width: '20px',
              height: '3px',
              right: 0,
              backgroundColor: borderColor,
              boxShadow: `0 0 8px 2px ${glowColor}`,
            }}></div>
            <div style={{
              position: 'absolute',
              width: '3px',
              height: '20px',
              right: 0,
              backgroundColor: borderColor,
              boxShadow: `0 0 8px 2px ${glowColor}`,
            }}></div>
          </div>
          
          {/* Bottom-left corner */}
          <div className="absolute transition-all duration-1000" style={{
            bottom: isBlurred ? '-4px' : '4px',
            left: isBlurred ? '-4px' : '4px',
          }}>
            <div style={{
              position: 'absolute',
              width: '20px',
              height: '3px',
              bottom: 0,
              backgroundColor: borderColor,
              boxShadow: `0 0 8px 2px ${glowColor}`,
            }}></div>
            <div style={{
              position: 'absolute',
              width: '3px',
              height: '20px',
              bottom: 0,
              backgroundColor: borderColor,
              boxShadow: `0 0 8px 2px ${glowColor}`,
            }}></div>
          </div>
          
          {/* Bottom-right corner */}
          <div className="absolute transition-all duration-1000" style={{
            bottom: isBlurred ? '-4px' : '4px',
            right: isBlurred ? '-4px' : '4px',
          }}>
            <div style={{
              position: 'absolute',
              width: '20px',
              height: '3px',
              bottom: 0,
              right: 0,
              backgroundColor: borderColor,
              boxShadow: `0 0 8px 2px ${glowColor}`,
            }}></div>
            <div style={{
              position: 'absolute',
              width: '3px',
              height: '20px',
              bottom: 0,
              right: 0,
              backgroundColor: borderColor,
              boxShadow: `0 0 8px 2px ${glowColor}`,
            }}></div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default BlurAnimation;