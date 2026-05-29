/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        background: 'hsl(var(--background))',
        foreground: 'hsl(var(--foreground))',
        card: 'hsl(var(--card))',
        'card-foreground': 'hsl(var(--card-foreground))',
        muted: 'hsl(var(--muted))',
        'muted-foreground': 'hsl(var(--muted-foreground))',
        border: 'hsl(var(--border))',
        accent: 'hsl(var(--accent))',
        primary: 'hsl(var(--primary))',
        'primary-foreground': 'hsl(var(--primary-foreground))',
        destructive: 'hsl(var(--destructive))',
        success: 'hsl(var(--success))',
      },
      keyframes: {
        'progress-indeterminate': {
          '0%': { transform: 'translateX(-100%)' },
          '100%': { transform: 'translateX(400%)' },
        },
        // DNA double-helix loader: each rung scales horizontally on a sine-
        // like cycle; staggered animation-delays across rungs create the
        // illusion of a rotating helix.
        'dna-rung': {
          '0%, 100%': { transform: 'scaleX(1)' },
          '25%': { transform: 'scaleX(0.18)' },
          '50%': { transform: 'scaleX(1)' },
          '75%': { transform: 'scaleX(0.18)' },
        },
      },
      animation: {
        'progress-indeterminate': 'progress-indeterminate 1.4s ease-in-out infinite',
        'dna-rung': 'dna-rung 1.6s ease-in-out infinite',
      },
    },
  },
  plugins: [],
}
