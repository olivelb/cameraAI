# Smart Camera pour Windows

## Lancer l’application

Sur le PC configuré, ouvrir **SmartCamera.exe** dans le dossier du projet.
Le lanceur ouvre une fenêtre native Qt, sans console et sans navigateur.
Il empêche l’ouverture d’une seconde instance par ce même lanceur.

Pour installer sur un autre PC Windows :

1. Installer Python 3.11 ou plus et créer un environnement `.venv` dans le projet.
2. Installer PyTorch avec la version CUDA adaptée au PC, puis
   `python -m pip install -r requirements-desktop.txt` et `python -m pip check`.
3. Copier `config.example.yaml` vers `config.yaml`, puis configurer le Pi et les
   identifiants. Conserver les jetons, galeries et poids de modèles en local.
4. Construire le lanceur avec
   `powershell -ExecutionPolicy Bypass -File tools/build_windows_launcher.ps1 -Shortcut`.

Le lanceur utilise d’abord `.venv/Scripts/python.exe`, puis `.venv-audit` s’il n’y
a pas de `.venv`. Il doit rester à côté de `desktop.py` : déplacer seulement
l’exécutable ne transfère pas Python, les bibliothèques CUDA ni les modèles.
Le raccourci Bureau peut être déplacé normalement.

Le contrôle par un LLM est disponible via le [CLI JSON](cli.md). Les commandes
pilotent cette même fenêtre, avec un canal local authentifié. La case de suivi
s’applique immédiatement ; l’état affiché reflète le moteur actif. L’assistant
et le CLI partagent la même conversation et n’y ajoutent pas les erreurs du modèle.

Alternative : `python desktop.py`. Pour examiner seulement l’interface, sans
caméra, téléchargement ou chargement IA : `python desktop.py --no-camera`.

## Commandes disponibles

| Zone | Fonctions |
| --- | --- |
| Aperçu | Vidéo directe, annotations IA ou image brute, pause d’affichage, capture PNG, plein écran F11 / retour Échap |
| Caméra | Adresse, ports vidéo et commandes, jeton masqué, délais réseau, lecture des réglages réels, résolution H.264, cadence demandée, débit |
| IA | Activation du traitement, modèles YOLO/SAM2/SAM3, fichier `.pt` ou `.engine`, détection/segmentation, concepts SAM3, confiance, export TensorRT, suivi BoT-SORT/ByteTrack |
| Visages | Activation, seuil de similarité, nom, import multiple de photos, reconstruction de galerie, apprentissage de 10 vues |
| Animaux | Activation, seuil de similarité, nom, import multiple de photos, reconstruction de galerie, apprentissage de 10 vues |
| Alertes | Activation explicite, expéditeur Gmail, mot de passe d’application, destinataire, sauvegarde chiffrée, email de test |
| Assistant | Question sur l’image courante, historique limité aux 20 derniers messages, effacement de la conversation |
| Journal | Messages du moteur, état de connexion, ouverture du dossier de l’application |

Modifier les champs puis cliquer sur **Appliquer** ou **Enregistrer** dans la
zone correspondante. Les seuils des galeries sont appliqués avec le bouton
de reconnaissance. La confiance de détection s’applique immédiatement.
Changer résolution/cadence/débit redémarre le flux du Pi ; le client se reconnecte.
Ces réglages du Pi restent valables pour la session serveur courante.
Si les commandes répondent mais que l’image reste inaccessible, **Relancer le
flux du Pi** force une relance de l’encodeur. La cadence est modifiée pendant
deux secondes puis les réglages initiaux sont restaurés, y compris en cas
d’erreur pendant l’attente. L’image est momentanément interrompue.

Les modèles et fonctions de reconnaissance utilisent les mêmes fichiers et
galeries que l’interface web. Leur présence dans l’interface ne garantit pas que
les poids, droits d’accès Hugging Face ou runtimes GPU nécessaires sont installés.
Les erreurs apparaissent dans la barre d’état et le journal. SAM3 et Gemma peuvent
se concurrencer pour la mémoire GPU ; le comportement de libération de SAM3 est
celui du moteur commun. Les alertes sont désactivées à chaque démarrage.

## Profils réseau : local, VPN et Tailscale

Configurer l’adresse accessible depuis le réseau utilisé :

- directement sur le réseau de la caméra : son adresse ou son nom d’hôte ;
- via un VPN : l’adresse accessible dans le tunnel, selon son routage et les
  éventuelles redirections de ports de l’installation ;
- les ports par défaut sont **8888** pour la vidéo et **8889** pour les commandes ;
- le profil **VPN / réseau mobile** propose 12 s pour ouvrir et 10 s pour lire ;
- le profil **Tailscale** propose 12 s pour ouvrir et 10 s pour lire ;
- le profil **Réseau local** propose 10 s pour ouvrir et 3 s pour lire.

Chaque profil mémorise son adresse, ses ports et ses délais. Sélectionner un
profil remplit les champs à partir de sa sauvegarde locale ; un nouveau profil
demande une adresse au lieu de reprendre celle d’un autre réseau.
**Enregistrer et reconnecter** valide et sauvegarde la connexion et le profil
dans `config.yaml`, puis relance la réception. Le dernier profil enregistré
est restauré au démarrage. Le jeton de contrôle reste commun à la caméra et
n’est pas dupliqué dans chaque profil.
Un délai plus long tolère davantage de pauses mais retarde la détection d’une
coupure ; il ne garantit ni débit supérieur ni faible latence vidéo.
Les profils ne configurent ni ne démarrent WireGuard ou Tailscale : activer le
client VPN séparément.

Pour Tailscale :

1. Connecter le PC et la caméra à Tailscale, avec les droits réseau nécessaires.
2. Choisir **Tailscale**, puis **Détecter les appareils Tailscale…** et sélectionner
   la caméra. La liste vient de `tailscale status --json`, lu localement en
   arrière-plan. L’adresse IPv4 est privilégiée pour ne pas dépendre de MagicDNS.
3. Une adresse ou un nom MagicDNS peut aussi être saisi manuellement.
4. Cliquer sur **Enregistrer et reconnecter**.

Les appareils détectés ne sont pas journalisés. Les noms, adresses, profils et
jetons restent dans la configuration locale ignorée par Git ; ne jamais les
recopier dans les sources ou dans `config.example.yaml`. Les tests de confidentialité
contrôlent aussi les adresses partagées utilisées par Tailscale et les domaines
MagicDNS. La détection ne change ni le compte Tailscale, ni les routes, ni les ACL.
Si aucun appareil n’est trouvé, vérifier la connexion Tailscale et les permissions,
ou saisir la cible manuellement.

Références : [connexion entre appareils Tailscale](https://tailscale.com/docs/how-to/connect-to-devices)
et [commande status](https://tailscale.com/kb/1080/cli).

Option de décodage à essayer : `SmartCamera.exe --low-latency`. Elle transmet
au décodeur des limites de sondage et les options `nobuffer` / `low_delay` avant
le démarrage de Python. Une comparaison sur le réseau mobile a donné une première
image plus rapide, mais avec encore des pauses ; cette option reste facultative.
Une modification de variable d’environnement après le chargement d’OpenCV peut
être inefficace sous Windows. Le lancement normal conserve le décodage standard.

Fermer les autres lecteurs du flux avant d’utiliser l’application : le serveur
`rpicam-vid --listen` accepte un seul lecteur vidéo à la fois. Les deux interfaces
peuvent être installées ensemble mais ne doivent pas ouvrir chacune le flux du Pi.

## Réactivité et limites

La réception, l’inférence et les commandes longues sont séparées de l’interface.
L’aperçu lit uniquement la dernière image disponible ; aucune file d’images ne
s’accumule dans Qt, et aucune conversion JPEG/MJPEG n’est utilisée pour l’affichage.
Les commandes longues sont exécutées une par une : les boutons correspondants
se désactivent pendant un chargement, une reconstruction ou une analyse.
La fenêtre attend la fin d’une commande en cours et la libération des workers de
réception/inférence avant de se fermer, pour préserver les écritures et libérer
proprement l’unique connexion vidéo.

**Réception** mesure les images décodées récemment. **Affichage** mesure les images
distinctes présentées par le timer d’aperçu. Le texte FPS du moteur IA décrit
l’inférence. Ces valeurs ne sont pas une mesure de latence de bout en bout ni
une certification de la cadence physique de la caméra.
En cas de silence réseau, la dernière image reste visible avec un avertissement.
Les erreurs de lecture provoquent une reconnexion dès le premier échec.

`desktop.log` et `launcher.log` restent locaux et ignorés par Git. Ils peuvent
contenir des informations du réseau et ne doivent pas être publiés sans revue.
Les identifiants de connexion restent dans `config.yaml`, également ignoré.

## Vérifier et diagnostiquer

```powershell
python -m pytest -q
python -m ruff check .
python -m tools.qualify_desktop --seconds 60 --output .audit-tmp/native.json
```

La dernière commande ouvre réellement l’application, la caméra et les modèles
configurés. Elle enregistre des compteurs et intervalles de la boucle Qt ; aucune
image n’est sauvegardée. Ne pas la lancer en même temps qu’un autre lecteur.
Les tests automatisés utilisent Qt hors écran et des images synthétiques ; ils
ne remplacent pas la qualification de la caméra/GPU/réseau réel.
Sur Linux, les tests Qt hors écran nécessitent aussi les bibliothèques système
`libegl1` et `libopengl0` ; la CI les installe avant l’exécution des tests.

En cas d’erreur au lancement, consulter `launcher.log`. Si l’image est figée,
regarder l’état de connexion et le compteur Réception, vérifier le VPN, fermer
les autres lecteurs puis utiliser **Enregistrer et reconnecter**.
Pour un modèle indisponible, choisir un poids installé et consulter le journal.

Références de conception : [images Qt](https://doc.qt.io/qtforpython-6/PySide6/QtGui/QImage.html)
et [déploiement Qt pour Python](https://doc.qt.io/qtforpython-6/deployment/index.html).
