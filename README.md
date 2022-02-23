# TrackmanAI

A purely **pixel based** Deep Reinforcement Learning Agent for the the racing game Trackmania.
### Features
- Soft Actor Critic Architecture
- Variational Autoencoder (the latent respresentation of the VAE is used for the state representation)
- OCR (Reading on-screen values like speed and checkpoints is
done via simple OCR. I implemented this from scratch since it was 10x faster than using libraries)
- Agent is trained from scratch and in real-time (about 15 steps per second)
- No cheat engines or other tools were used



Disclaimer: There are about a million things that can (and maybe will) be improved. This is a work in progress.
Future focus will be the inclusion of more biomes and tracks, and proper reward penalties. If you have any improvements or questions
feel free to shoot me a message.