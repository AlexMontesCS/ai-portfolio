/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{vue,js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'snake-green': '#10b981',
        'snake-dark': '#065f46',
        'food-red': '#ef4444',
        'grid-gray': '#374151',
        'neural-blue': '#3b82f6',
        'neural-purple': '#8b5cf6'
      },
      animation: {
        'pulse-slow': 'pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'bounce-slow': 'bounce 2s infinite',
        'fade-in': 'fadeIn 0.5s ease-in-out',
        'slide-up': 'slideUp 0.3s ease-out'
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' }
        },
        slideUp: {
          '0%': { transform: 'translateY(10px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' }
        }
      },
      fontFamily: {
        'mono': ['Fira Code', 'Monaco', 'Consolas', 'monospace'],
      },
      gridTemplateColumns: {
        'snake': 'repeat(15, minmax(0, 1fr))',
      },
      spacing: {
        '18': '4.5rem',
        '88': '22rem',
      }
    },
  },
  plugins: [],
}
