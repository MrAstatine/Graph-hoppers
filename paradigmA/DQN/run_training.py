import modal
import os

image = (
    modal.Image.debian_slim()
    .pip_install_from_requirements("requirements.txt")
    .run_commands("pip install -e gym-environments/")
)

app = modal.App("dqn-training", image=image)

project_mount = modal.Mount.from_local_dir(".", remote_path="/root/DQN")


@app.function(
    gpu="L4",  # ← change GPU here
    timeout=3600 * 5,
    mounts=[project_mount],
)
def train():
    import subprocess

    os.chdir("/root/DQN")

    # Run training
    subprocess.run(["python", "train_DQN.py"])

    # ---- CHECKPOINT SAVING CODE STARTS HERE ----
    checkpoint_files = {}
    checkpoint_dir = "./modelssample_DQN_agent"

    for fname in os.listdir(checkpoint_dir):
        fpath = os.path.join(checkpoint_dir, fname)
        with open(fpath, "rb") as f:
            checkpoint_files[fname] = f.read()

    return checkpoint_files
    # ---- CHECKPOINT SAVING CODE ENDS HERE ----


@app.local_entrypoint()
def main():
    # Receive checkpoints from the cloud container
    checkpoint_files = train.remote()

    # Save them to your local DQN folder
    save_dir = "./modelssample_DQN_agent"
    os.makedirs(save_dir, exist_ok=True)

    for fname, data in checkpoint_files.items():
        with open(os.path.join(save_dir, fname), "wb") as f:
            f.write(data)

    print("✅ Checkpoints saved locally to", save_dir)
