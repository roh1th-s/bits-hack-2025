"use client";

import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';

const RandomMovingGlass = () => {
  const [position, setPosition] = useState({ x: 50, y: 50 });
  
  // Function to generate random positions within viewport
  const generateRandomPosition = () => {
    // Get viewport dimensions (with some padding)
    const viewportWidth = typeof window !== 'undefined' ? window.innerWidth - 100 : 500;
    const viewportHeight = typeof window !== 'undefined' ? window.innerHeight - 100 : 500;
    
    return {
      x: Math.random() * viewportWidth,
      y: Math.random() * viewportHeight
    };
  };
  
  // Update position periodically
  useEffect(() => {
    // Set initial random position
    setPosition(generateRandomPosition());
    
    // Change position every 3-6 seconds
    const intervalId = setInterval(() => {
      setPosition(generateRandomPosition());
    }, Math.random() * 3000 + 3000);
    
    return () => clearInterval(intervalId);
  }, []);
  
  return (
    <motion.div
      className="pointer-events-none fixed z-50"
      animate={{
        x: position.x,
        y: position.y,
        rotate: [0, 10, -5, 15, 0],
        scale: [1, 1.05, 0.98, 1.02, 1]
      }}
      transition={{
        type: "spring",
        stiffness: 50,
        damping: 15,
        rotate: {
          duration: 10,
          repeat: Infinity,
          ease: "linear"
        },
        scale: {
          duration: 8,
          repeat: Infinity,
          ease: "easeInOut"
        }
      }}
      style={{
        width: "150px",
        height: "150px"
      }}
    >
      {/* Glass Circle */}
      <div className="w-full h-full relative">
        {/* Main glass effect */}
        <div 
          className="absolute inset-0 rounded-full"
          style={{ 
            border: '2px solid #000',
            boxShadow: '0 0 15px rgba(0, 0, 0, 0.2)',
            background: 'rgba(255, 255, 255, 0.15)',
            backdropFilter: 'blur(8px)'
          }}
        >
          {/* Inner reflections */}
          <div className="absolute w-1/3 h-1/5 bg-white opacity-40 rounded-full" 
              style={{ 
                top: '15%', 
                left: '10%', 
                transform: 'rotate(-20deg)' 
              }}></div>
              
          <div className="absolute w-1/4 h-1/6 bg-white opacity-30 rounded-full" 
              style={{ 
                top: '40%', 
                right: '20%',
                transform: 'rotate(30deg)' 
              }}></div>
              
          {/* Subtle circular gradient inside */}
          <div className="absolute inset-0 rounded-full" 
              style={{
                background: 'radial-gradient(circle, rgba(255,255,255,0.1) 0%, rgba(200,200,255,0.05) 60%, rgba(150,150,200,0.08) 100%)',
                transform: 'scale(0.9)'
              }}></div>
        </div>
        
        {/* Moving highlight to simulate light refraction */}
        <motion.div 
          className="absolute bg-white opacity-10 rounded-full w-1/2 h-1/2"
          animate={{
            left: ['10%', '60%', '10%'],
            top: ['10%', '60%', '10%'],
          }}
          transition={{
            duration: 8,
            repeat: Infinity,
            ease: "easeInOut"
          }}
        ></motion.div>
      </div>
    </motion.div>
  );
};

export default RandomMovingGlass;