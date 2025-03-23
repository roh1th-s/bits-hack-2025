"use client"

import type React from "react"
import { useState, useRef } from "react"
import { motion, AnimatePresence } from "framer-motion"
import ReactJson from "react-json-view"
import { Navbar } from "@/components/navbar"
import { Footer } from "@/components/footer"
import { ChevronRight, Linkedin } from "lucide-react";
import { Instagram, Send, MessageCircle } from 'lucide-react';
import Link from "next/link";

import {
  Search,
  Upload,
  FileImage,
  Shield,
  AlertTriangle,
  CheckCircle,
  Info,
  X,
  Loader2,
  Gauge,
  FileText,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion"
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import { cn } from "@/lib/utils"
interface RedirectButtonProps {
  className?: string;
}
async function RedirectButton({ className = "" }: RedirectButtonProps) {
  await new Promise((resolve) => setTimeout(resolve, 1000));

  return (
    <div className="flex flex-col sm:flex-row gap-4 justify-center items-center pb-8">
      {/* WhatsApp Button */}
      <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
        <Link href="/whatsapp">
          <button
            type="button"
            className={`
                bg-green-500 hover:bg-green-600 
                text-white font-medium 
                py-3 px-6 rounded-md 
                shadow-md hover:shadow-lg
                transition-all duration-300
                flex items-center justify-center gap-2
                bg-gradient-to-r from-green-500 via-green-400 to-green-600
                ${className}
              `}
            aria-label="Connect with WhatsApp"
          >
            <MessageCircle/> Connect with WhatsApp
            <ChevronRight className="h-5 w-5" />
          </button>
        </Link>
      </motion.div>

      {/* LinkedIn Button */}
      <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
        <Link href="/linkedin">
          <button
            type="button"
            className={`
                bg-blue-600 hover:bg-blue-700 
                text-white font-medium 
                py-3 px-6 rounded-md 
                shadow-md hover:shadow-lg
                transition-all duration-300
                flex items-center justify-center gap-2
                bg-gradient-to-r from-blue-600 via-blue-500 to-blue-700
                ${className}
              `}
            aria-label="Connect with LinkedIn"
          >
            <Linkedin/> Connect with LinkedIn
            <ChevronRight className="h-5 w-5" />
          </button>
        </Link>
      </motion.div>

      {/* Instagram Button */}
      <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
        <Link href="/instagram">
          <button
            type="button"
            className={`
                bg-pink-500 hover:bg-pink-600 
                text-white font-medium 
                py-3 px-6 rounded-md 
                shadow-md hover:shadow-lg
                transition-all duration-300
                flex items-center justify-center gap-2
                bg-gradient-to-r from-pink-500 via-pink-400 to-purple-500
                ${className}
              `}
            aria-label="Connect with Instagram"
          >
            <Instagram/> Connect with Instagram
            <ChevronRight className="h-5 w-5" />
          </button>
        </Link>
      </motion.div>
    </div>
  );
}

export default function Page() {
  return (
    <main className="min-h-screen flex flex-col bg-gradient-to-b from-background to-background/95">
      <Navbar />

      <div className="flex-1 container mx-auto px-4 py-24">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="max-w-5xl mx-auto"
        >
          <div className="flex items-center space-x-4 mb-6">
            <div className="p-3 rounded-full bg-primary/10">
              <Search className="w-8 h-8 text-primary" />
            </div>
            <div>
              <h1 className="text-4xl font-space font-bold">Fake Profile Detection</h1>
              <p className="text-muted-foreground">
                Analyze Profiles for identification of AI-generated content
              </p>
            </div>
          </div>

          <RedirectButton />

          <motion.div
            className="grid grid-cols-1 md:grid-cols-3 gap-6"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3, staggerChildren: 0.1 }}
          >
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
              whileHover={{ y: -5, boxShadow: "0 10px 30px rgba(0,0,0,0.1)" }}
            >
              <Card className="h-[200px]">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Shield className="w-5 h-5 text-primary" />
                    How It Works
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-muted-foreground">
                    Our deepfake detection system uses advanced AI to analyze images for signs of manipulation. The
                    system examines facial features, lighting, and texture patterns to identify inconsistencies.
                  </p>
                </CardContent>
              </Card>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 }}
              whileHover={{ y: -5, boxShadow: "0 10px 30px rgba(0,0,0,0.1)" }}
            >
              <Card className="h-[200px]">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <FileImage className="w-5 h-5 text-primary" />
                    Supported Formats
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="flex flex-wrap gap-2">
                    <Badge variant="outline">JPG</Badge>
                    <Badge variant="outline">JPEG</Badge>
                    <Badge variant="outline">PNG</Badge>
                    <Badge variant="outline">WEBP</Badge>
                    <Badge variant="outline">MP4</Badge>
                    <Badge variant="outline">MOV</Badge>
                    <Badge variant="outline">AVI</Badge>
                  </div>
                </CardContent>
              </Card>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.6 }}
              whileHover={{ y: -5, boxShadow: "0 10px 30px rgba(0,0,0,0.1)" }}
            >
              <Card className="h-[200px]">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Info className="w-5 h-5 text-primary" />
                    Privacy
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-muted-foreground">
                    Your uploads are processed securely and not stored permanently. We respect your privacy and do not
                    share your data with third parties.
                  </p>
                </CardContent>
              </Card>
            </motion.div>
          </motion.div>
        </motion.div>
      </div>

      <Footer />
    </main>
  );
}
