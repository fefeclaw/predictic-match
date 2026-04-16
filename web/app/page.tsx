export default function Home() {
  return (
    <main className="min-h-screen bg-gradient-to-br from-green-900 via-green-800 to-green-900 p-8">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <header className="text-center mb-12">
          <h1 className="text-5xl font-bold text-white mb-4">
            ⚽ Predictic Match
          </h1>
          <p className="text-xl text-green-100">
            Système de prédiction de matchs de football par IA
          </p>
          <div className="mt-4 flex justify-center gap-4">
            <a
              href="https://github.com/fefeclaw/predictic-match"
              target="_blank"
              className="bg-white text-green-800 px-6 py-2 rounded-full font-semibold hover:bg-green-100 transition"
            >
              📦 Voir sur GitHub
            </a>
            <a
              href="#predictions"
              className="bg-green-600 text-white px-6 py-2 rounded-full font-semibold hover:bg-green-500 transition"
            >
              🔮 Voir les prédictions
            </a>
          </div>
        </header>

        {/* Features */}
        <section className="grid md:grid-cols-3 gap-6 mb-12">
          <div className="bg-white/10 backdrop-blur rounded-xl p-6 border border-white/20">
            <div className="text-3xl mb-3">📊</div>
            <h3 className="text-white font-semibold mb-2">3 Couches de Données</h3>
            <p className="text-green-100 text-sm">
              Bookmaker odds + Polymarket + Modèle ML pour une précision maximale
            </p>
          </div>
          <div className="bg-white/10 backdrop-blur rounded-xl p-6 border border-white/20">
            <div className="text-3xl mb-3">🤖</div>
            <h3 className="text-white font-semibold mb-2">100% Autonome</h3>
            <p className="text-green-100 text-sm">
              Aucun dépendance à Claude API - entièrement gratuit et open source
            </p>
          </div>
          <div className="bg-white/10 backdrop-blur rounded-xl p-6 border border-white/20">
            <div className="text-3xl mb-3">📈</div>
            <h3 className="text-white font-semibold mb-2">ML Avancé</h3>
            <p className="text-green-100 text-sm">
              XGBoost + Ensemble avec ELO ratings, xG proxy, et analyse de fatigue
            </p>
          </div>
        </section>

        {/* Predictions Section */}
        <section id="predictions" className="bg-white rounded-xl p-8 shadow-2xl">
          <h2 className="text-3xl font-bold text-gray-800 mb-6">
            🔮 Dernières Prédictions
          </h2>
          <div className="bg-gray-100 rounded-lg p-8 text-center">
            <p className="text-gray-600 mb-4">
              Les prédictions sont générées automatiquement chaque jour
            </p>
            <div className="animate-pulse">
              <div className="h-4 bg-gray-300 rounded w-3/4 mx-auto mb-2"></div>
              <div className="h-4 bg-gray-300 rounded w-1/2 mx-auto"></div>
            </div>
            <p className="text-sm text-gray-500 mt-4">
              📊 Prochaine mise à jour : dans 2 heures
            </p>
          </div>
        </section>

        {/* Stats */}
        <section className="mt-12 grid md:grid-cols-4 gap-4">
          <div className="bg-white/10 backdrop-blur rounded-xl p-4 text-center border border-white/20">
            <div className="text-3xl font-bold text-white">23</div>
            <div className="text-green-100 text-sm">Ligues Supportées</div>
          </div>
          <div className="bg-white/10 backdrop-blur rounded-xl p-4 text-center border border-white/20">
            <div className="text-3xl font-bold text-white">&gt;50%</div>
            <div className="text-green-100 text-sm">Accuracy Cible</div>
          </div>
          <div className="bg-white/10 backdrop-blur rounded-xl p-4 text-center border border-white/20">
            <div className="text-3xl font-bold text-white">3</div>
            <div className="text-green-100 text-sm">Sources de Données</div>
          </div>
          <div className="bg-white/10 backdrop-blur rounded-xl p-4 text-center border border-white/20">
            <div className="text-3xl font-bold text-white">24/7</div>
            <div className="text-green-100 text-sm">Mise à Jour Auto</div>
          </div>
        </section>

        {/* Footer */}
        <footer className="mt-12 text-center text-green-200 text-sm">
          <p>⚠️ Ce système est à but informatif uniquement. Pariez responsablement.</p>
          <p className="mt-2">
            Créé avec ❤️ par <a href="https://github.com/fefeclaw" className="underline hover:text-white">@fefeclaw</a>
          </p>
        </footer>
      </div>
    </main>
  )
}
