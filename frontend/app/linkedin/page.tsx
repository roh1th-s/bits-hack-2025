"use client";
import React, { useState, FormEvent, useRef, useEffect } from 'react';
import { Search, ArrowLeft, Clock, Trash2, ExternalLink } from 'lucide-react';
import Link from 'next/link';
import { motion } from 'framer-motion';

interface UrlSearchProps {
  className?: string;
}

interface UrlHistory {
  url: string;
  timestamp: Date;
}

export default function UrlSearchPage({ className }: UrlSearchProps): JSX.Element {
  const [url, setUrl] = useState<string>('');
  const [error, setError] = useState<string | null>(null);
  const [isSearching, setIsSearching] = useState<boolean>(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const [inputFocused, setInputFocused] = useState<boolean>(false);
  
  // Add URL history state
  const [urlHistory, setUrlHistory] = useState<UrlHistory[]>([]);
  const [showHistory, setShowHistory] = useState<boolean>(false);

  // Focus the input field when component mounts
  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.focus();
    }
  }, []);

  // Load history from localStorage on component mount
  useEffect(() => {
    const savedHistory = localStorage.getItem('urlSearchHistory');
    if (savedHistory) {
      try {
        // Parse stored JSON and convert timestamp strings back to Date objects
        const parsedHistory = JSON.parse(savedHistory).map((item: any) => ({
          url: item.url,
          timestamp: new Date(item.timestamp)
        }));
        setUrlHistory(parsedHistory);
      } catch (error) {
        console.error('Failed to parse history:', error);
      }
    }
  }, []);

  // Save URL to history
  const saveToHistory = (urlToSave: string): void => {
    const newHistory = [
      { url: urlToSave, timestamp: new Date() },
      ...urlHistory.filter(item => item.url !== urlToSave).slice(0, 9) // Keep max 10 items, remove duplicates
    ];
    setUrlHistory(newHistory);
    localStorage.setItem('urlSearchHistory', JSON.stringify(newHistory));
  };

  // Delete URL from history
  const deleteFromHistory = (urlToDelete: string): void => {
    const newHistory = urlHistory.filter(item => item.url !== urlToDelete);
    setUrlHistory(newHistory);
    localStorage.setItem('urlSearchHistory', JSON.stringify(newHistory));
  };

  // Clear all history
  const clearHistory = (): void => {
    setUrlHistory([]);
    localStorage.removeItem('urlSearchHistory');
  };

  const handleSubmit = (e: FormEvent<HTMLFormElement>): void => {
    e.preventDefault();
    
    // Basic URL validation
    if (!url) {
      setError("Please provide valid link");
      return;
    }
    
    // Check if URL is valid format
    try {
      const urlObj = new URL(url);
      if (!urlObj.hostname) {
        throw new Error("Invalid URL");
      }
      
      // Begin search animation
      setError(null);
      setIsSearching(true);
      
      // Save valid URL to history
      saveToHistory(url);
      
      // Simulate search process
      setTimeout(() => {
        setIsSearching(false);
        // Here you would typically redirect or process the URL
        // For demo purposes, we'll just keep the user on this page
      }, 1500);
      
    } catch (error) {
      setError("Please provide valid link");
    }
  };

  // Helper function to format timestamp
  const formatTimestamp = (date: Date): string => {
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMins / 60);
    const diffDays = Math.floor(diffHours / 24);

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins} ${diffMins === 1 ? 'minute' : 'minutes'} ago`;
    if (diffHours < 24) return `${diffHours} ${diffHours === 1 ? 'hour' : 'hours'} ago`;
    if (diffDays < 7) return `${diffDays} ${diffDays === 1 ? 'day' : 'days'} ago`;
    
    return date.toLocaleDateString(undefined, { 
      year: 'numeric', 
      month: 'short', 
      day: 'numeric' 
    });
  };

  // Side magnifying glass component with faster animation
  const MagnifyingGlass = ({ leftSide = true }) => {
    return (
      <motion.div
        initial={{ y: 0, rotate: leftSide ? -10 : 10 }}
        animate={{ 
          y: [0, -8, 0, -5, 0],
          rotate: leftSide ? [-10, -5, -12, -7, -10] : [10, 5, 12, 7, 10],
          scale: [1, 1.08, 1, 1.05, 1]
        }}
        transition={{ 
          repeat: Infinity, 
          duration: leftSide ? 2 : 1.8,
          ease: "easeInOut"
        }}
        style={{
          position: "absolute",
          top: leftSide ? "-12px" : "-8px",
          left: leftSide ? "-40px" : "auto",
          right: leftSide ? "auto" : "-40px",
          width: "40px",
          height: "40px",
          zIndex: 10
        }}
      >
        {/* Glass */}
        <div className="rounded-full w-10 h-10 relative">
          {/* Glass frame */}
          <div 
            className="rounded-full absolute top-0 left-0 w-full h-full"
            style={{ 
              border: '2px solid #333',
              boxShadow: '0 0 8px rgba(0, 0, 0, 0.2)',
              background: 'rgba(255, 255, 255, 0.05)',
              backdropFilter: 'blur(1px)'
            }}
          >
            {/* Glass lens effect */}
            <div 
              className="absolute w-full h-full rounded-full"
              style={{
                background: 'radial-gradient(circle, rgba(255,255,255,0.1) 0%, rgba(200,200,200,0.05) 70%, rgba(150,150,150,0.08) 100%)',
                transform: 'scale(0.9)'
              }}
            ></div>
            
            {/* Highlight on glass */}
            <div className="absolute rounded-full bg-white opacity-30" 
                style={{ 
                  width: '15px',
                  height: '6px',
                  top: '6px', 
                  left: '6px', 
                  transform: 'rotate(20deg)' 
                }}></div>
          </div>
          
          {/* Handle */}
          <div 
            style={{
              position: 'absolute',
              width: '8px',
              height: '20px',
              backgroundColor: '#333',
              bottom: leftSide ? '-18px' : '-16px',
              right: leftSide ? '2px' : '6px',
              transform: `rotate(${leftSide ? 45 : 135}deg)`,
              borderRadius: '4px',
              boxShadow: '0 0 3px rgba(0, 0, 0, 0.2)'
            }}
          ></div>
        </div>
      </motion.div>
    );
  };


  // History Box Component
  const HistoryBox = () => {
    return (
      <motion.div 
        className="relative mt-12 mb-6 rounded-lg overflow-hidden"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: 20 }}
        transition={{ duration: 0.3 }}
      >
        {/* Animated border */}
        <div className="absolute inset-0 rounded-lg overflow-hidden">
          <div className="absolute inset-0 bg-gradient-to-r from-primary via-primary/50 to-primary rounded-lg">
            <motion.div 
              className="absolute inset-0 bg-gradient-to-r from-primary/30 via-primary/70 to-primary/30"
              animate={{
                backgroundPosition: ['0% 0%', '100% 100%'],
              }}
              transition={{
                duration: 3,
                repeat: Infinity,
                repeatType: 'reverse',
                ease: 'linear'
              }}
            />
          </div>
        </div>
        
        {/* Content with padding for border */}
        <div className="relative m-[2px] bg-card rounded-lg p-4">
          <div className="flex justify-between items-center mb-4">
            <div className="flex items-center">
              <Clock size={16} className="text-primary mr-2" />
              <h3 className="font-medium">Search History</h3>
            </div>
            
            <div className="flex gap-2">
              {urlHistory.length > 0 && (
                <button 
                  onClick={clearHistory}
                  className="text-xs text-muted-foreground hover:text-foreground flex items-center"
                >
                  <Trash2 size={14} className="mr-1" />
                  Clear All
                </button>
              )}
            </div>
          </div>
          
          {urlHistory.length === 0 ? (
            <div className="text-center py-6 text-sm text-muted-foreground">
              <p>No search history yet</p>
              <p className="mt-1 text-xs">Your recent searches will appear here</p>
            </div>
          ) : (
            <ul className="space-y-2 max-h-60 overflow-y-auto">
              {urlHistory.map((item, index) => (
                <motion.li
                  key={`${item.url}-${index}`}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.05 }}
                  className="flex items-center justify-between p-2 rounded-md hover:bg-accent group"
                >
                  <div className="flex-1 truncate">
                    <div className="flex items-center">
                      <button 
                        onClick={() => setUrl(item.url)}
                        className="truncate text-sm hover:text-primary flex-1 text-left"
                      >
                        {item.url}
                      </button>
                      <a 
                        href={item.url} 
                        target="_blank" 
                        rel="noopener noreferrer"
                        className="ml-2 text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity"
                      >
                        <ExternalLink size={14} />
                      </a>
                    </div>
                    <p className="text-xs text-muted-foreground mt-1">
                      {formatTimestamp(item.timestamp)}
                    </p>
                  </div>
                  <button
                    onClick={() => deleteFromHistory(item.url)}
                    className="opacity-0 group-hover:opacity-100 transition-opacity text-muted-foreground hover:text-destructive ml-2"
                    aria-label="Delete from history"
                  >
                    <Trash2 size={14} />
                  </button>
                </motion.li>
              ))}
            </ul>
          )}
        </div>
      </motion.div>
    );
  };

  // Toggle history visibility button
  const HistoryToggle = () => (
    <motion.button
      onClick={() => setShowHistory(!showHistory)}
      className={`mt-6 mx-auto flex items-center text-sm text-muted-foreground hover:text-primary transition-colors`}
      whileHover={{ scale: 1.05 }}
      whileTap={{ scale: 0.95 }}
    >
      <Clock size={16} className="mr-1" />
      {showHistory ? 'Hide Search History' : 'Show Search History'}
    </motion.button>
  );

  return (
    <div className={`min-h-screen bg-background p-4 ${className || ''}`}>
      {/* Back button */}
      <div className="absolute top-4 right-4">
        <Link href="/detect">
          <button
            type="button"
            className="flex items-center text-sm text-muted-foreground"
            aria-label="Back to Detect Page"
          >
            <ArrowLeft size={16} className="mr-1" />
            Back
          </button>
        </Link>
      </div>
      
      <div className="max-w-2xl mx-auto mt-20">
        <h1 className="text-2xl font-bold text-center mb-8">Search URL</h1>
        
        <form onSubmit={handleSubmit} className="w-full">
          <div className="relative">
            <div className={`
              absolute inset-0 bg-primary/10 rounded-lg transition-all duration-300 ease-in-out
              ${inputFocused ? 'scale-105 opacity-100' : 'scale-100 opacity-0'}
            `}></div>
            
            {/* Animated magnifying glasses on sides */}
            <MagnifyingGlass leftSide={true} />
            <MagnifyingGlass leftSide={false} />
            
            <div className="relative flex items-center">
              <input
                ref={inputRef}
                type="text"
                value={url}
                onChange={(e) => {
                  setUrl(e.target.value);
                  if (error) setError(null);
                }}
                onFocus={() => setInputFocused(true)}
                onBlur={() => setInputFocused(false)}
                placeholder="Please provide link"
                className={`
                  w-full p-4 pr-12 rounded-lg border border-input
                  bg-background text-foreground shadow-sm
                  focus:ring-2 focus:ring-primary focus:outline-none
                  transition-all duration-300
                  ${error ? 'border-red-500 focus:ring-red-500' : ''}
                  ${isSearching ? 'bg-primary/5' : ''}
                `}
              />
              
              
              <button
                type="submit"
                disabled={isSearching}
                className={`
                  absolute right-3 p-2 rounded-full
                  transition-all duration-300
                  ${isSearching 
                    ? 'bg-primary text-primary-foreground animate-pulse' 
                    : 'bg-secondary hover:bg-primary text-secondary-foreground hover:text-primary-foreground'}
                `}
                aria-label="Search URL"
              >
                <Search size={20} />
              </button>
            </div>
            
            {/* Incognito design horizontal line */}
            <div className="relative mt-2 w-full">
              <div className="absolute w-full h-px bg-gray-200"></div>
              <div className="absolute w-full flex justify-center">
                <div className="flex items-center -mt-3 bg-background px-4">
                  {/* Incognito hat icon */}
                  <svg width="20" height="16" viewBox="0 0 20 16" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M19 6C19 8.21 17.21 10 15 10C12.79 10 11 8.21 11 6C11 3.79 12.79 2 15 2C17.21 2 19 3.79 19 6Z" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                    <path d="M9 6C9 8.21 7.21 10 5 10C2.79 10 1 8.21 1 6C1 3.79 2.79 2 5 2C7.21 2 9 3.79 9 6Z" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                    <path d="M1 6H19" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                    <path d="M10 15C10 12.79 11.79 11 14 11H16C18.21 11 20 12.79 20.2 15" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" strokeDasharray="1 3"/>
                    <path d="M0 15C0 12.79 1.79 11 4 11H6C8.21 11 10 12.79 10 15" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" strokeDasharray="1 3"/>
                  </svg>
                  <span className="ml-2 text-xs text-muted-foreground">Incognito Search</span>
                </div>
              </div>
            </div>
          </div>
          
          {/* Error message */}
          {error && (
            <div className="mt-6 text-red-500 text-sm animate-fade-down">
              {error}
            </div>
          )}
          
          {/* Search status */}
          {isSearching && (
            <div className="mt-6 flex justify-center">
              <div className="flex items-center space-x-2 text-sm text-muted-foreground animate-fade-up">
                <div className="w-4 h-4 rounded-full border-2 border-primary border-t-transparent animate-spin"></div>
                <span>Searching...</span>
              </div>
            </div>
          )}
        </form>
        
        <div className="mt-8 text-center text-sm text-muted-foreground">
          <p>Enter a complete URL including http:// or https://</p>
        </div>
        
        {/* History toggle button */}
        <div className="flex justify-center">
          <HistoryToggle />
        </div>
        
        {/* Show history box when toggle is enabled */}
        {showHistory && <HistoryBox />}
      </div>
    </div>
  );
}