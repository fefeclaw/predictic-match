import type { Metadata } from 'next'

export const metadata: Metadata = {
  title: 'Predictic Match - Football Predictions',
  description: 'Système de prédiction de matchs de football par IA avec ML, Polymarket et bookmaker odds',
  keywords: 'football, prediction, AI, ML, betting, soccer, analytics',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="fr">
      <body className="antialiased">{children}</body>
    </html>
  )
}
