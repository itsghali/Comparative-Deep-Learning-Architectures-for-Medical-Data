import os
import urllib.request
import zipfile
import tarfile
from pathlib import Path

# Définir le chemin absolu vers data/raw (à la racine du projet)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"

def download_and_extract(url, extract_to=DATA_RAW_DIR):
    """
    Télécharge un fichier depuis une URL et le décompresse (zip ou tar.gz) si nécessaire.
    """
    # S'assurer que les chemins sont des strings si on utilise os
    extract_to = str(extract_to)
    
    # Créer le dossier de destination s'il n'existe pas
    os.makedirs(extract_to, exist_ok=True)
    
    # Déduire le nom du fichier à partir de l'URL
    filename = url.split("/")[-1]
    filepath = os.path.join(extract_to, filename)
    
    # 1. Téléchargement
    if not os.path.exists(filepath):
        print(f"[*] Téléchargement de {filename} depuis {url}...")
        try:
            urllib.request.urlretrieve(url, filepath)
            print("[+] Téléchargement terminé.")
        except Exception as e:
            print(f"[!] Erreur lors du téléchargement : {e}")
            return
    else:
        print(f"[+] Le fichier {filename} existe déjà dans {extract_to}.")
        
    # 2. Extraction
    extract_local_archive(filepath, extract_to)

def extract_local_archive(filepath, extract_to=DATA_RAW_DIR):
    """
    Extrait une archive locale (ZIP ou TAR.GZ) dans le dossier spécifié.
    """
    filepath = str(filepath)
    extract_to = str(extract_to)
    
    # Créer le dossier s'il n'existe pas
    os.makedirs(extract_to, exist_ok=True)
    
    if filepath.endswith(".zip"):
        print(f"[*] Extraction de {os.path.basename(filepath)} (ZIP) vers {extract_to}...")
        try:
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            print("[+] Extraction terminée.")
        except zipfile.BadZipFile:
            print("[!] Erreur : Le fichier ZIP est corrompu.")
            
    elif filepath.endswith(".tar.gz") or filepath.endswith(".tgz"):
        print(f"[*] Extraction de {os.path.basename(filepath)} (TAR.GZ) vers {extract_to}...")
        try:
            with tarfile.open(filepath, 'r:gz') as tar_ref:
                tar_ref.extractall(extract_to)
            print("[+] Extraction terminée.")
        except tarfile.TarError:
            print("[!] Erreur : Le fichier TAR est corrompu.")
    else:
        print(f"[-] Le fichier {os.path.basename(filepath)} n'est ni un .zip ni un .tar.gz, aucune extraction nécessaire.")

if __name__ == "__main__":
    print("="*50)
    print("[*] SCRIPT DE TÉLÉCHARGEMENT ET D'EXTRACTION")
    print(f"[*] Destination absolue définie sur : {DATA_RAW_DIR}")
    print("="*50)
    
    # Parcourir tous les fichiers dans le dossier data/raw et les extraire
    if DATA_RAW_DIR.exists():
        found_archives = False
        for file_path in DATA_RAW_DIR.iterdir():
            if file_path.is_file() and (file_path.name.endswith(".zip") or file_path.name.endswith(".tar.gz") or file_path.name.endswith(".tgz")):
                found_archives = True
                extract_local_archive(file_path)
        
        if not found_archives:
            print("[-] Aucune archive (.zip, .tar.gz) trouvée dans le dossier data/raw.")
    else:
        print(f"[!] Le dossier {DATA_RAW_DIR} n'existe pas encore.")
