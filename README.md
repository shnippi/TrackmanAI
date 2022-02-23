# TrackmanAI

A purely **pixel based** Deep Reinforcement Learning Agent for the the racing game Trackmania.
### Features
- Soft Actor Critic Architecture
- Variational Autoencoder (the latent respresentation of the VAE is used for the state representation)
- OCR (Reading on-screen values like speed and checkpoints is
done via simple OCR. I implemented this from scratch since it was 10x faster than using libraries)
- Agent is trained from scratch and in real-time (about 15 steps per second)
- No cheat engines or other tools were used

## Training Video
Left Side of the screen is the actual game that the Agent is controlling, the right side shows the reconstructed image from the VAE's latent representation of the current frame. This is "what the agent sees".

https://user-images.githubusercontent.com/45588658/155300990-b7b19dfd-5a87-4399-88d6-fc2ecdfa7c70.mp4


Disclaimer: There are about a million things that can (and maybe will) be improved. This is a work in progress.
Future focus will be the inclusion of more biomes and tracks, and proper reward penalties. If you have any improvements or questions
feel free to shoot me a message.

