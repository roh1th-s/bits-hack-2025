"use client";

import { motion } from "framer-motion";
import { useInView } from "react-intersection-observer";
import { AlertOctagon, ShieldCheck, Search, BellRing, CheckCircle } from "lucide-react";

const features = [
  {
    icon: AlertOctagon,
    title: "Flag Fake Accounts",
    description: "Quickly report suspicious accounts with ease.",
  },
  {
    icon: ShieldCheck,
    title: "User Verification",
    description: "Ensure only verified users interact on the platform.",
  },
  {
    icon: Search,
    title: "Advanced Detection",
    description: "Leverage AI to detect fraudulent account patterns.",
  },
  {
    icon: BellRing,
    title: "Real-time Alerts",
    description: "Receive instant alerts on fake account activities.",
  },
  {
    icon: CheckCircle,
    title: "Secure Community",
    description: "Help maintain a safe and trusted environment.",
  },
];

export function FeaturesSection() {
  const [ref, inView] = useInView({
    triggerOnce: true,
    threshold: 0.1,
  });

  return (
    <section ref={ref} className="py-24 bg-secondary/50">
      <div className="container mx-auto px-4">
        <motion.h2
          initial={{ opacity: 0, y: 20 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.6 }}
          className="text-3xl md:text-4xl font-space font-bold text-center mb-16"
        >
          Features for a Safer Community
        </motion.h2>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {features.map((feature, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              animate={inView ? { opacity: 1, y: 0 } : {}}
              transition={{ duration: 0.6, delay: index * 0.1 }}
              className="p-6 rounded-lg bg-card hover:bg-card/80 transition-colors"
            >
              <feature.icon className="w-10 h-10 mb-4 text-primary" />
              <h3 className="text-xl font-semibold mb-2">{feature.title}</h3>
              <p className="text-muted-foreground">{feature.description}</p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}