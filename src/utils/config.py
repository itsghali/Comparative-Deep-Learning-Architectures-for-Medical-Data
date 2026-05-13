import torch

class Config:
    # ---------------------------------------------------------
    # Configuration du Device (CPU / GPU)
    # ---------------------------------------------------------
    # Vérifie si un GPU NVIDIA est disponible, sinon utilise le CPU
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ---------------------------------------------------------
    # Autres paramètres globaux (à enrichir plus tard si besoin)
    # ---------------------------------------------------------
    SEED = 42

def print_config():
    """Affiche la configuration actuelle du projet."""
    print("="*40)
    print("🔧 CONFIGURATION DU PROJET")
    print("="*40)
    print(f"🚀 Device actif : {Config.DEVICE}")
    if Config.DEVICE.type == 'cuda':
        print(f"💻 Nom du GPU : {torch.cuda.get_device_name(0)}")
        print(f"💾 Mémoire GPU allouée : {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    print(f"🌱 Seed globale  : {Config.SEED}")
    print("="*40)

if __name__ == "__main__":
    print_config()
