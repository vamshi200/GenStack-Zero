import { motion } from 'framer-motion'

function AnimatedLogo() {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.82, rotateX: -20 }}
      animate={{ opacity: 1, scale: 1, rotateX: 0 }}
      transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
      className="relative flex h-24 w-20 items-center justify-center"
      aria-label="Vamshi logo"
      role="img"
    >
      <motion.span
        animate={{
          textShadow: [
            '0 0 16px rgba(229, 9, 20, 0.35)',
            '0 0 34px rgba(229, 9, 20, 0.9)',
            '0 0 16px rgba(229, 9, 20, 0.35)',
          ],
        }}
        transition={{ duration: 2.8, repeat: Infinity, ease: 'easeInOut' }}
        className="text-7xl font-black tracking-[-0.2em] text-red-600"
      >
        V
      </motion.span>
    </motion.div>
  )
}

export default AnimatedLogo
