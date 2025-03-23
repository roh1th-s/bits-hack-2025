"use client";
import React, { useState, FormEvent, useRef, useEffect, ChangeEvent } from 'react';
import { Search, ArrowLeft, Upload, Clock } from 'lucide-react';
import Link from 'next/link';
import { motion } from 'framer-motion';

interface UrlSearchProps {
  className?: string;
}

export default function UrlSearchPage({ className }: UrlSearchProps): JSX.Element {
  const [file, setFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [inputFocused, setInputFocused] = useState<boolean>(false);
  // Add state for upload history
  const [uploadHistory, setUploadHistory] = useState<Array<{name: string, date: string}>>([
    { name: "project_backup.zip", date: "2025-03-22 14:32" },
    { name: "source_code_v1.zip", date: "2025-03-20 09:15" },
    { name: "assets_collection.zip", date: "2025-03-18 16:45" }
  ]);

  // Focus the input field when component mounts
  useEffect(() => {
    if (fileInputRef.current) {
      fileInputRef.current.focus();
    }
  }, []);

  const handleSubmit = (e: FormEvent<HTMLFormElement>): void => {
    e.preventDefault();
    
    // Basic file validation
    if (!file) {
      setError("Please select a ZIP file");
      return;
    }
    
    // Check if file is a ZIP
    if (!file.name.toLowerCase().endsWith('.zip')) {
      setError("Please provide a valid ZIP file");
      return;
    }
    
    // Begin upload animation
    setError(null);
    setIsUploading(true);
    
    // Simulate upload process
    setTimeout(() => {
      setIsUploading(false);
      // Add new file to history when upload completes
      const now = new Date();
      const formattedDate = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')} ${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}`;
      setUploadHistory(prev => [{name: file.name, date: formattedDate}, ...prev]);
      // Here you would typically process the ZIP file
      // For demo purposes, we'll just keep the user on this page
    }, 1500);
  };

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0]);
      if (error) setError(null);
    }
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
        <h1 className="text-2xl font-bold text-center mb-8">Upload ZIP File</h1>
        
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
              <label 
                htmlFor="file-upload" 
                className={`
                  w-full p-4 pr-12 rounded-lg border border-input
                  bg-background text-foreground shadow-sm
                  focus:ring-2 focus:ring-primary focus:outline-none
                  transition-all duration-300 cursor-pointer flex items-center
                  ${error ? 'border-red-500 focus:ring-red-500' : ''}
                  ${isUploading ? 'bg-primary/5' : ''}
                `}
              >
                {file ? file.name : "Select a ZIP file"}
                <input
                  ref={fileInputRef}
                  id="file-upload"
                  type="file"
                  accept=".zip"
                  onChange={handleFileChange}
                  onFocus={() => setInputFocused(true)}
                  onBlur={() => setInputFocused(false)}
                  className="sr-only" // Hide the actual input
                />
              </label>
            
              
              <button
                type="submit"
                disabled={isUploading}
                className={`
                  absolute right-3 p-2 rounded-full
                  transition-all duration-300
                  ${isUploading 
                    ? 'bg-primary text-primary-foreground animate-pulse' 
                    : 'bg-secondary hover:bg-primary text-secondary-foreground hover:text-primary-foreground'}
                `}
                aria-label="Upload ZIP File"
              >
                <Upload size={20} />
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
          
          {/* Upload status */}
          {isUploading && (
            <div className="mt-6 flex justify-center">
              <div className="flex items-center space-x-2 text-sm text-muted-foreground animate-fade-up">
                <div className="w-4 h-4 rounded-full border-2 border-primary border-t-transparent animate-spin"></div>
                <span>Uploading...</span>
              </div>
            </div>
          )}
        </form>
        
        <div className="mt-8 text-center text-sm text-muted-foreground">
          <p>Please upload a ZIP file (max 50MB)</p>
        </div>

        {/* Upload History Box with Fancy Borders */}
        <div className="mt-10 mx-auto max-w-md">
          <div 
            className="p-5 bg-background rounded-lg"
            style={{
              border: '1px solid #e2e8f0',
              boxShadow: '0 4px 6px rgba(0, 0, 0, 0.05), 0 1px 3px rgba(0, 0, 0, 0.1)',
              position: 'relative',
              backgroundImage: 'linear-gradient(to right, rgba(99, 102, 241, 0.05), rgba(99, 102, 241, 0.02))'
            }}
          >
            {/* Fancy corner borders */}
            <div style={{ position: 'absolute', top: '-2px', left: '-2px', width: '15px', height: '15px', borderTop: '3px solid #6366f1', borderLeft: '3px solid #6366f1', borderRadius: '4px 0 0 0' }}></div>
            <div style={{ position: 'absolute', top: '-2px', right: '-2px', width: '15px', height: '15px', borderTop: '3px solid #6366f1', borderRight: '3px solid #6366f1', borderRadius: '0 4px 0 0' }}></div>
            <div style={{ position: 'absolute', bottom: '-2px', left: '-2px', width: '15px', height: '15px', borderBottom: '3px solid #6366f1', borderLeft: '3px solid #6366f1', borderRadius: '0 0 0 4px' }}></div>
            <div style={{ position: 'absolute', bottom: '-2px', right: '-2px', width: '15px', height: '15px', borderBottom: '3px solid #6366f1', borderRight: '3px solid #6366f1', borderRadius: '0 0 4px 0' }}></div>
            
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-medium flex items-center">
                <Clock size={18} className="mr-2 text-primary" />
                Recent Uploads
              </h3>
              <span className="text-xs text-muted-foreground bg-primary/10 px-2 py-1 rounded-full">
                {uploadHistory.length} files
              </span>
            </div>
            
            <div className="space-y-3 max-h-48 overflow-y-auto pr-1">
              {uploadHistory.map((item, index) => (
                <div 
                  key={index} 
                  className="flex items-center justify-between p-2 rounded-md border-l-2 border-primary/70 bg-primary/5 hover:bg-primary/10 transition-colors"
                >
                  <div className="flex items-center">
                    <svg 
                      width="18" 
                      height="18" 
                      viewBox="0 0 24 24" 
                      fill="none" 
                      xmlns="http://www.w3.org/2000/svg"
                      className="mr-2 text-primary"
                    >
                      <path d="M21 14V19C21 19.5304 20.7893 20.0391 20.4142 20.4142C20.0391 20.7893 19.5304 21 19 21H5C4.46957 21 3.96086 20.7893 3.58579 20.4142C3.21071 20.0391 3 19.5304 3 19V5C3 4.46957 3.21071 3.96086 3.58579 3.58579C3.96086 3.21071 4.46957 3 5 3H10" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                      <path d="M12 12L21 3" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                      <path d="M16 3H21V8" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                    </svg>
                    <span className="truncate max-w-xs text-sm">{item.name}</span>
                  </div>
                  <span className="text-xs text-muted-foreground">{item.date}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}