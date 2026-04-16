# GitHub Workflows - Predictic Match

## 📋 Workflows Disponibles

### 1. Daily Predictions (`predictions.yml`)
**Déclencheur :** Tous les jours à 06:00 UTC + déclenchement manuel

**Fonctionnalités :**
- Exécute le pipeline de prédiction complet
- Génère les prédictions pour les ligues configurées
- Crée les visualisations
- Commit et push les résultats
- Upload les artifacts (30 jours)

**Déclenchement manuel :**
```
Actions → Daily Predictions → Run workflow
- League: E0 (Premier League), SP1 (La Liga), etc.
- Season: 2425, 2324, etc.
```

---

### 2. CI - Tests & Validation (`ci.yml`)
**Déclencheur :** Push/PR sur main et develop

**Fonctionnalités :**
- Tests sur Python 3.9, 3.10, 3.11
- Linting avec flake8
- Tests unitaires avec pytest
- Couverture de code avec Codecov
- Build de l'application web Next.js

**Statut :** ✅ Requis pour les PR

---

### 3. Deploy to Vercel (`deploy-vercel.yml`)
**Déclencheur :** Push sur main (dossier web/) + manuel

**Fonctionnalités :**
- Dé dé dé dé dé dé déploie automatiquement sur Vercel
- Utilise le token Vercel depuis les secrets
- Build et déploiement en production

**Variables requises :**
- `VERCEL_TOKEN` dans les secrets du dépôt

---

## 🔧 Configuration des Secrets

Aller dans : **Settings → Secrets and variables → Actions**

### Secrets Requis :

| Secret | Description | Comment obtenir |
|--------|-------------|-----------------|
| `VERCEL_TOKEN` | Token de déploiement Vercel | `vercel tokens create` |
| `GITHUB_TOKEN` | Auto (fourni par GitHub) | N/A |

---

## 📊 Monitoring

### Voir les exécutions :
1. **Actions** tab sur GitHub
2. Sélectionner le workflow
3. Voir l'historique des runs

### Logs détaillés :
- Cliquer sur un run
- Voir les logs de chaque étape
- Télécharger les artifacts

---

## 🚨 Résolution de Problèmes

### Échec du workflow predictions :
```bash
# Vérifier les logs d'erreur
# Souvent : données indisponibles ou dépendances manquantes
```

### Échec du déploiement Vercel :
```bash
# Vérifier que VERCEL_TOKEN est valide
# vercel tokens ls
```

### Tests échoués :
```bash
# Lancer les tests en local
pytest tests/ -v
```

---

## 📝 Bonnes Pratiques

1. **Ne jamais committer de tokens** dans le code
2. **Utiliser les secrets GitHub** pour toutes les credentials
3. **Limiter les artifacts** à 30 jours maximum
4. **Tester en local** avant de pusher
5. **Utiliser le caching** pour accélérer les builds

---

## 🔗 Liens Utiles

- [GitHub Actions Docs](https://docs.github.com/en/actions)
- [Vercel GitHub Integration](https://vercel.com/docs/git)
- [pytest Documentation](https://docs.pytest.org/)
