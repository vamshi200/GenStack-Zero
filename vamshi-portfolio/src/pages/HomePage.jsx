import { motion } from 'framer-motion'
import { ArrowRight, ExternalLink, UserRound } from 'lucide-react'
import AnimatedLogo from '../components/branding/AnimatedLogo'
import { siteConfig } from '../data/site'

function HomePage() {
  return (
    <main className="relative flex min-h-screen items-center justify-center overflow-hidden px-6 py-16 text-neutral-100">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,_rgba(229,9,20,0.16),_transparent_42%)]" />

      <motion.section
        initial={{ opacity: 0, y: 24 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.75, ease: [0.16, 1, 0.3, 1] }}
        className="relative w-full max-w-4xl rounded-[2rem] border border-white/10 bg-black/55 p-8 shadow-[0_30px_80px_rgba(0,0,0,0.45)] backdrop-blur md:p-12"
      >
        <div className="absolute inset-x-10 top-0 h-px bg-gradient-to-r from-transparent via-red-500/70 to-transparent" />

        <div className="flex flex-col items-center text-center">
          <AnimatedLogo />
          <p className="mt-4 text-xs font-semibold uppercase tracking-[0.45em] text-red-500">
            {siteConfig.name}
          </p>
          <h1 className="mt-6 max-w-3xl text-4xl font-semibold tracking-tight text-white md:text-6xl">
            {siteConfig.title}
          </h1>
          <p className="mt-5 max-w-2xl text-sm leading-7 text-neutral-300 md:text-base">
            React, Vite, Tailwind CSS, Framer Motion, and Lucide React are now
            configured as the foundation for an original, Netflix-inspired
            personal portfolio.
          </p>

          <div className="mt-10 flex flex-col gap-3 sm:flex-row">
            <a
              href={siteConfig.github}
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center justify-center gap-2 rounded-full border border-white/15 bg-white/5 px-5 py-3 text-sm font-medium text-white transition hover:border-red-500/50 hover:bg-red-600 hover:text-white"
            >
              <ExternalLink size={18} />
              GitHub
            </a>
            <a
              href={siteConfig.linkedin}
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center justify-center gap-2 rounded-full border border-white/15 bg-white/5 px-5 py-3 text-sm font-medium text-white transition hover:border-white/30 hover:bg-white hover:text-black"
            >
              <UserRound size={18} />
              LinkedIn
            </a>
          </div>

          <div className="mt-10 inline-flex items-center gap-2 rounded-full border border-red-500/20 bg-red-500/10 px-4 py-2 text-xs uppercase tracking-[0.3em] text-red-200">
            Setup only
            <ArrowRight size={14} />
          </div>
        </div>
      </motion.section>
    </main>
  )
}

export default HomePage
