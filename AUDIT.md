# Audit du dépôt camera-client — 23 octobre 2026

## Verdict et périmètre

Audit de l'ensemble des 16 fichiers suivis au départ, sur la branche `main`,
base `anonymized-revision`. Le dépôt était propre. Revue du client Gradio, moteur IA,
reconnaisseurs, serveur Pi, configuration, notifications, dépendances,
service systemd, documentation et tests. L'audit et ses corrections ont été livrés
dans le commit `anonymized-revision`, publié sur `main` après validation CI Ubuntu et Windows.

**Complément du 24 octobre :** le serveur a depuis été déployé et testé sur le
vrai Pi Zero W, avec corrections supplémentaires. Voir la
[qualification matérielle](audit/camera-2026-10-24.md). Les tableaux ci-dessous
décrivent les preuves de l'audit initial du 23 ; leurs limites matérielles sont
complétées par ce second rapport. Le blocage sécurité Pillow reste ouvert.

Les défauts applicatifs décrits comme corrigés ci-dessous ont été traités.
La qualification de sécurité des dépendances reste **en échec** à cause de
Pillow. La qualification matérielle complète Pi/CUDA reste à effectuer.
Les photos, modèles, certificats, galeries existantes, configuration locale
et environnement Python global n'ont pas été modifiés.

## Constats et corrections

| Priorité | Constat confirmé et impact | Correction | Preuve / limite |
| --- | --- | --- | --- |
| P1 | `/stream` était ajouté directement à FastAPI sans contrôle du login Gradio ; le flux restait accessible malgré le login UI | Contrôle des cookies et de la session Gradio dans `streaming.py`, réponse 401 et absence de cache | Tests HTTP réels : anonyme, cookie forgé, login, lecture, logout |
| P1 | Un nom de personne/animal pouvait contenir `../`, un chemin absolu ou Windows ; écritures hors galerie | Validation commune des noms, confinement après résolution, rejet des liens sortants et noms réservés | Tests de chemins, accents/espaces, lien symbolique et appel réel `capture_face` |
| P1 | `ui.share=true` sans login n'était qu'un message de log | Refus de démarrer sans les deux champs de login ; login partiel également refusé | Tests de configuration |
| P1 | Des erreurs d'import du serveur rétablissaient silencieusement des valeurs par défaut, dont un token vide | Suppression de ce repli, dépendances Pi explicites | Le serveur échoue au démarrage si sa configuration ne peut pas être chargée |
| P1 | Le démarrage Windows pouvait tuer le propriétaire du port et tous les tunnels Gradio nommés | Vérification non destructive du port, erreur exploitable par l'utilisateur | Test avec un véritable socket occupé, conservé ouvert |
| P1 | `SMTP.starttls()` utilisait le contexte implicite de la bibliothèque | Contexte TLS explicite vérifiant certificat et nom du serveur | Test du contexte avant authentification ; aucun e-mail réel envoyé |
| P2 | Les variables d'environnement étaient typées d'après la présence des clés YAML ; un `false` absent du YAML devenait une chaîne vraie | Conversion selon les types des dataclasses, validation des booléens, listes, ports, seuils, résolutions, classes et clés | Tests YAML et variables d'environnement, rejet des types invalides et valeurs non finies |
| P2 | Un JSON valide mais non objet, `data` non objet, FPS négatif/booléen ou débit nul pouvaient casser le contrôle caméra | Validation des structures, des entiers, FPS 1–30, débit 100 kbit/s–25 Mbit/s et résolutions | Intégration sur un véritable serveur WebSocket local |
| P2 | Le client affichait « Sent » dès l'envoi, même si le Pi rejetait la commande | Lecture de l'état initial et de l'acquittement, comparaison avec la demande, fermeture dans `finally` | Test de rejet explicite côté client ; intégration protocole côté serveur |
| P1 | Les repères faciaux étaient conservés dans l'image 640×640 alors que les boîtes étaient remises à l'échelle ; décalage d'ancrage et ordre BGR également incorrects | RGB pour le détecteur, centres d'ancrage conformes à InsightFace, repères remis à l'échelle, repli sans repères | Tests numériques à échelles différentes ; précision d'identification non mesurée |
| P2 | Activer le tracking sur SAM3 appelait une méthode `.track` inexistante | SAM et SAM3 utilisent leur voie de segmentation lorsque le tracking est activé | Test de régression SAM3 |
| P2 | Un chargement de modèle raté changeait le nom actif avant le remplacement effectif | Publication du modèle, nom et type seulement après construction réussie | Test de conservation du modèle précédent |
| P2 | La présence d'un moteur TensorRT forçait son usage sur CPU | Sélection automatique d'un moteur uniquement avec CUDA ; VLM 4 bits refusé tôt sur CPU | Test du chargement CPU ; inférence CPU réelle avec `yolo26n.pt` |
| P2 | L'export d'un modèle de segmentation pouvait recharger le modèle de détection, et entrait en concurrence avec l'inférence | Export sous verrou et conservation de la tâche ; erreurs de rechargement propagées | Chemin relu ; export TensorRT matériel non exécuté |
| P2 | Changer le nom du tracker avec `persist=True` conservait les anciens objets Ultralytics | Reconstruction des trackers déjà initialisés sans doubler les callbacks | Test réel local : `BOTSORT` devient `BYTETracker` |
| P2 | Les caches d'identité pouvaient survivre à la disparition de tous les tracks ou à un changement de modèle | Purge des caches et historiques | Tests de purge ; traitement des changements sous verrou |
| P2 | Les recadrages animaux acceptaient des coordonnées négatives ou hors image | Bornage aux limites et retour vide pour une boîte inversée | Test de recadrage |
| P2 | Les galeries et réglages e-mail étaient écrits directement dans leurs fichiers ; les erreurs d'image pouvaient être annoncées comme succès | Écriture temporaire puis remplacement atomique, contrôle de `imwrite`, fermeture des NPZ | Simulation d'échec disque : fichier précédent conservé ; tests sans photos personnelles |
| P2 | La clé e-mail était créée à l'import, les chemins dépendaient du répertoire courant, un mot de passe indéchiffrable restait utilisable | Chemins ancrés au dépôt, création différée avec permissions restreintes, mot de passe vidé en cas d'échec, sauvegardes sérialisées | Tests de chiffrement et conservation du mot de passe sur champ vide |
| P2 | Les alertes annonçaient un e-mail envoyé avant la fin du thread SMTP | Envoi synchrone dans le thread d'alerte existant, journal de succès après retour | Tests SMTP simulés, propagation de l'échec |
| P1 | `KillMode=process-group` n'est pas une valeur systemd valide ; arrêt susceptible de laisser le flux enfant en vie | `control-group`, Python 3 explicite ; arrêt du watchdog avant libération du verrou | Référence systemd ; qualification sur le Pi encore requise |
| P2 | `preexec_fn=os.setsid` était utilisé depuis un thread ; arrêt incomplet, stderr masqué, watchdog perdu si l'exécutable manquait | `start_new_session=True`, SIGTERM puis SIGKILL, stderr conservé, reprise après échec de création, finalisation du serveur | Test de libération en cas d'échec du bind ; cycle rpicam réel non exécuté |
| P2 | Le PID était vérifié puis écrit sans exclusion atomique ; une instance pouvait supprimer le verrou d'une autre | Verrou Linux `flock` maintenu par descripteur ; inode conservé | Test multiprocessus ajouté pour Linux, ignoré sous Windows |
| P2 | Importer `app` démarrait encore le chargeur IA et modifiait la priorité Windows | Chargeur et optimisation Windows déplacés au démarrage explicite ; priorité ABOVE_NORMAL au lieu de REALTIME | Construction de l'UI réelle sans chargeur ni workers caméra |
| P2 | Le générateur MJPEG synchrone pouvait rester en attente dans un worker après déconnexion ; lectures vidéo sans délai explicite | Générateur asynchrone annulable, événement d'arrêt et délais FFmpeg de 5 s | Test d'encodage/décodage JPEG et fermeture ; reconnexion Pi à qualifier |
| P1 | Gradio 5.45 installé exigeait Hub `<1`, incompatible avec Transformers installé qui exige Hub `>=1.5` | Migration d'API vers Gradio 6.30, thème à `launch`, retrait de `ChatInterface(type=...)` ; manifestes core/dev/Pi séparés | Installation neuve, construction de l'UI, tests HTTP et `pip check` réussis |
| P2 | Seulement 12 tests et aucune CI | Suite étendue et workflow Ubuntu 3.11 / Windows 3.13, droits lecture, sans téléchargement de modèles | 70 tests passent localement ; CI distante non exécutée |

P1 signifie priorité élevée, P2 défaut de fiabilité ou de fonctionnement.
Ces niveaux ne sont pas des scores CVSS ni une mesure de l'exploitabilité réelle.

## Sécurité des dépendances : reste à résoudre

Un environnement CPU de test neuf a été installé dans `.venv-test`.
Après mise à jour de son `pip`, `pip-audit` signale **35 avis sur Pillow 12.3.0**.
Les versions corrigées indiquées sont `13.0.0`, mais Gradio 6.30.0 exige
`Pillow<13`. Il n'existe donc pas de résolution satisfaisant ces deux contraintes
dans le profil qualifié ici. Aucune alerte n'a été masquée et aucune installation
forcée avec `--no-deps` n'a été faite.

Le relevé compact est dans [audit/dependencies-2026-10-23.json](audit/dependencies-2026-10-23.json).
Le résultat complet est conservé localement dans `.audit-tmp/dependency-audit-final.json`.
Une correspondance de version à un avis ne prouve pas que chaque chemin est
atteignable dans cette application. Cela interdit néanmoins de qualifier les
dépendances de sécurisées. Le correctif demande une version de Gradio compatible
Pillow 13, puis une nouvelle validation des uploads, images et du VLM.

Le scan n'inclut pas l'ensemble PyTorch/CUDA/TensorRT ni les deux dépendances Git
de la pile ML. Le Python global comporte en outre des conflits avec des paquets
hors projet (`sentence-transformers`, `insightface`) ; il a été préservé.
`.venv-audit` réutilise ces paquets globaux pour les essais YOLO ; ce n'est pas
l'environnement neuf utilisé pour le verdict `pip check`.

## Vérifications réalisées

| Vérification | Résultat |
| --- | --- |
| État initial : `python -m pytest -q` | 12 tests réussis |
| État initial : Ruff | Réussi, sans détecter les défauts fonctionnels ci-dessus |
| Installation neuve : `python -m pip install -r requirements-dev.txt` | Réussie sous Windows / Python 3.13.5 |
| Environnement neuf : `python -m pip check` | Aucun conflit |
| Environnement neuf : `python -m ruff check .` | Réussi |
| Environnement neuf : `python -m pytest -q` | 70 réussis, 1 ignoré : verrou Linux |
| HTTP Gradio réel / WebSocket local | Login, cookies, logout, acquittements et entrées invalides vérifiés |
| Import et construction de `app.demo` sous Gradio 6 | Réussis, aucun worker IA/capture démarré |
| YOLO local sur image noire 640×480 | Détection CPU réussie, forme de sortie conservée, aucune détection |
| Tracking YOLO local puis changement de tracker | `BOTSORT` puis `BYTETracker`, confirmé sur les objets réels |
| `pip-audit` environnement CPU neuf | Échec : 35 avis Pillow non résolus |
| `git diff --check` | Réussi |

Les temps d'inférence de ces essais uniques ne constituent pas un benchmark.
Les comptes rendus XML, versions de paquets et JSON complet restent dans
`.audit-tmp`. Aucun courrier de test, téléchargement de modèles, export TensorRT,
reconstruction des galeries personnelles ou redémarrage du Pi n'a été effectué.

## Actions de mise à jour

1. Utiliser un environnement dédié et réinstaller depuis les manifestes mis à jour.
   Le Python global est resté dans son état initial.
2. Après mise à jour, reconstruire explicitement la galerie **faciale** depuis les
   photos originales : les embeddings précédents reposaient sur le traitement erroné.
   Les photos et l'ancienne galerie ont été conservées durant l'audit.
3. Déployer le serveur et son `config.py` ensemble ; installer
   `requirements-server.txt`, ajuster l'interpréteur du service au virtualenv,
   puis tester arrêt, reconnexion et changement de résolution sur le Pi.
4. Traiter le blocage Pillow avant de qualifier un déploiement exposé.

## Améliorations proposées, par ordre d'utilité

1. **Terminer la qualification sécurité et verrouiller les environnements.**
   Résoudre le plafond Pillow, auditer toute la pile ML et produire des locks
   séparés CPU, GPU Windows et Pi, avec hashes. Épingler les actions CI à leurs
   SHA. Réexécuter les tests sur chaque profil avant de modifier les révisions ML.
2. **Qualifier le matériel et la qualité de reconnaissance.** Sur le Pi : arrêt
   SIGTERM, rpicam absent/planté, port occupé, plusieurs clients, perte réseau et
   changement de résolution. Sur GPU : cycles YOLO/SAM3/VLM, export et chargement
   TensorRT, mémoire après permutations. Sur un jeu d'images autorisé et séparé
   de l'apprentissage : faux positifs, faux négatifs et stabilité des identités.
   Les seuils et gains de précision ne peuvent pas être déduits des tests internes.
3. **Mieux isoler le cycle de vie applicatif.** Introduire progressivement une
   fabrique d'application et un service d'état injecté, puis des workers avec
   files bornées et arrêt attendu. Le serveur a une finalisation explicite ; le
   client conserve des threads daemon dont l'arrêt complet n'est pas encore garanti.
4. **Fiabiliser les reconstructions concurrentes.** Les sauvegardes NPZ sont
   atomiques mais deux reconstructions simultanées peuvent publier des états dans
   un ordre différent de leur lancement. Sérialiser les reconstructions et
   versionner le modèle, le prétraitement et le domaine des crops dans les galeries.
5. **Améliorer l'observabilité et le coût du streaming.** État caméra connecté/perdu,
   état de chaque modèle et erreurs de livraison structurés ; encodage JPEG unique
   partagé entre spectateurs, limites de clients et mesures de latence/mémoire.
6. **Protéger le transport et les données.** Le flux TCP Pi est toujours sans
   authentification ni chiffrement ; réserver les ports au réseau privé/VPN.
   Définir une rétention et une sauvegarde des galeries biométriques, contrôler les
   ACL Windows des secrets et vérifier l'intégrité des poids téléchargés.

## Références de vérification

- [Implémentation officielle InsightFace du détecteur](https://github.com/deepinsight/insightface/blob/master/python-package/insightface/model_zoo/retinaface.py) : ordre RGB, centres et remise à l'échelle.
- [Valeurs officielles systemd KillMode](https://www.freedesktop.org/software/systemd/man/latest/systemd.kill.html) : `control-group` est valide, `process-group` ne l'est pas.
- [Gradio 6.30.0 sur PyPI](https://pypi.org/project/gradio/6.30.0/) et métadonnées de ses dépendances consultées le jour de l'audit.
- [Notes Pillow 13.0.0](https://pillow.readthedocs.io/en/stable/releasenotes/13.0.0.html) et [avis GHSA-78g4-xqwv-q3g8](https://github.com/advisories/GHSA-78g4-xqwv-q3g8).
