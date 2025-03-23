"use client";

import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';

export default function Loading() {
  // Increased glass radius
  const glassRadius = 45;
  
  // Generate random emoji pattern on load
  const [emojiPattern, setEmojiPattern] = useState<{ emoji: string; x: number; y: number; size: number; rotation: number; opacity: number; }[]>([]);
  
  useEffect(() => {
    const personEmojis = ['👤',];
    const robotEmojis = ['🤖', '👾', '💻', '⚙️'];
    
    const patternArray = [];
    
    // Increased number of emojis to 120 (from 40)
    for (let i = 0; i < 120; i++) {
      // 80% chance for person emoji, 20% chance for robot emoji
      const emoji = Math.random() > 0.2 
        ? personEmojis[Math.floor(Math.random() * personEmojis.length)]
        : robotEmojis[Math.floor(Math.random() * robotEmojis.length)];
      
      patternArray.push({
        emoji,
        x: Math.random() * 100, // % position
        y: Math.random() * 100, // % position
        size: Math.random() * 1.8 + 1, // Increased size between 1em and 2.8em
        rotation: Math.random() * 40 - 20, // Random rotation -20 to 20 degrees
        opacity: Math.random() * 0.4 + 0.15, // Increased opacity between 0.15 and 0.55
      });
    }
    
    setEmojiPattern(patternArray);
  }, []);
  
  return (
    <div className="flex items-center justify-center min-h-screen bg-grey relative overflow-hidden">
      {/* Background pattern of emojis */}
      {emojiPattern.map((item, index) => (
        <div 
          key={index}
          className="absolute text-black"
          style={{
            left: `${item.x}%`,
            top: `${item.y}%`,
            fontSize: `${item.size}em`,
            transform: `rotate(${item.rotation}deg)`,
            opacity: item.opacity,
            zIndex: 1
          }}
        >
          {item.emoji}
        </div>
      ))}
      
      <div className="relative w-full max-w-md px-4 z-10">
        {/* White backdrop behind the FAKE text to improve readability */}
        <div className="absolute bg-grey rounded-xl opacity-90" 
             style={{
               width: "100%",
               height: "120px",
               top: "10px",
               left: "0",
               zIndex: 5
             }}></div>
             
        {/* Container for the word FAKE */}
        <div className="relative text-center mb-16 z-10">
          {/* Base word in light gray */}
          <h1 className="text-8xl font-extrabold tracking-widest text-gray-300">
            FAKE
          </h1>

          {/* Animated magnifying glass positioned over the word */}
          <motion.div 
            className="absolute"
            initial={{ left: "50px", top: "10px" }}
            animate={{ 
              left: ["30px", "200px", "30px"]
            }}
            transition={{ 
              repeat: Infinity, 
              duration: 3,
              ease: "easeInOut"
            }}
            style={{
              top: "0px",
              width: `${glassRadius * 2}px`,
              height: `${glassRadius * 2}px`,
              zIndex: 20
            }}
          >
            {/* Magnified content under the glass */}
            <div className="absolute rounded-full overflow-hidden" 
                 style={{
                   width: `${glassRadius * 2 - 8}px`, 
                   height: `${glassRadius * 2 - 8}px`, 
                   top: "4px", 
                   left: "4px",
                   zIndex: 10
                 }}>
              {/* REAL version of text perfectly positioned to show through glass */}
              <motion.div
                initial={{ left: "-30px" }}
                animate={{ 
                  left: ["-30px", "-200px", "-30px"]
                }}
                transition={{ 
                  repeat: Infinity, 
                  duration: 3,
                  ease: "easeInOut"
                }}
                style={{
                  position: "absolute",
                  top: "-0px",
                  width: "400px"
                }}
              >
                <h1 className="text-8xl font-extrabold tracking-widest text-black">
                  REAL
                </h1>
              </motion.div>
            </div>

            {/* Magnifier glass frame - now with transparent background */}
            <div 
              className="rounded-full absolute top-0 left-0 w-full h-full"
              style={{ 
                border: '4px solid black',
                boxShadow: '0 0 10px rgba(0, 0, 0, 0.3)',
                zIndex: 15,
                background: 'transparent', // Changed from white to transparent
                backdropFilter: 'blur(1px)' // Added slight blur for glass effect
              }}
            >
              {/* Glass lens effect - now more transparent */}
              <div 
                className="absolute w-full h-full rounded-full"
                style={{
                  background: 'radial-gradient(circle, rgba(255,255,255,0.15) 0%, rgba(200,200,200,0.08) 70%, rgba(150,150,150,0.1) 100%)',
                  transform: 'scale(0.9)'
                }}
              ></div>
              
              {/* Highlight on glass - reduced opacity */}
              <div className="absolute rounded-full bg-white opacity-20" 
                   style={{ 
                     width: '35px',
                     height: '14px',
                     top: '20px', 
                     left: '20px', 
                     transform: 'rotate(20deg)' 
                   }}></div>
            </div>
            
            {/* Handle */}
            <div 
              style={{
                position: 'absolute',
                width: '12px',
                height: '40px',
                backgroundColor: 'black',
                bottom: '-35px',
                right: '14px',
                transform: 'rotate(45deg)',
                borderRadius: '6px',
                boxShadow: '0 0 5px rgba(0, 0, 0, 0.3)',
                zIndex: 15
              }}
            ></div>
          </motion.div>
        </div>
        
        {/* Loading text with white background for better readability */}
        <div className="text-center relative z-10">
          <div className="bg-black rounded-lg py-2 px-4 inline-block opacity-90">
            <p className="text-white font-medium animate-pulse">
              Redirecting......
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}