# 🎙️ Voice Cloning with Emotion Control

This project implements a **deep learning–based voice cloning system** that synthesizes speech in a target speaker’s voice. Built using **Tacotron 2** (text → mel spectrogram) and **WaveGlow** (mel → waveform), it converts text into natural-sounding audio.  

A key feature is **emotion control**, allowing the cloned voice to speak with different styles such as *neutral, happy, sad, and angry*.  

---

## 🚀 Features
- Text-to-Speech (TTS) in a cloned voice  
- Emotion control via labels or reference audio  
- Tacotron 2 backbone with attention mechanism  
- WaveGlow vocoder for high-quality speech synthesis  
- Preprocessing, training, and inference scripts  
- Gradio/Streamlit interface for live demo  

---

## 📂 Project Structure
voice-cloning-project/
│── data/ # Datasets (LJSpeech, EmoV-DB, RAVDESS, etc.)
│── preprocessing/ # Audio cleaning + mel extraction scripts
│── models/ # Tacotron2 + WaveGlow checkpoints
│── notebooks/ # Colab/Jupyter notebooks
│── app/ # Gradio/Streamlit demo app
│── README.md # Project description

---

## 🛠️ Tech Stack
- Python 3.10+  
- PyTorch (Tacotron 2, WaveGlow)  
- Librosa / Torchaudio (audio processing)  
- Gradio / Streamlit (demo UI)  
- Google Colab / CUDA GPUs (training + inference)  

---

## 📊 Datasets
- [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) – base dataset (neutral speech)  
- [EmoV-DB](https://www.openslr.org/115/) / [RAVDESS](https://zenodo.org/record/1188976) – emotion-labeled speech  
- Few-shot recordings of target speaker for cloning  

---

## 🔬 Training Pipeline
1. Preprocess data → resample, trim silence, normalize  
2. Convert audio to mel spectrograms  
3. Train Tacotron 2 with emotion conditioning  
4. Train / fine-tune WaveGlow vocoder  
5. Run inference: input text + emotion → output speech  

---

## 📈 Results
- Naturalness measured via **MOS (Mean Opinion Score)**  
- Emotion accuracy via pretrained emotion classifier  
- Speaker similarity via embedding cosine similarity  

---

## 📌 Applications
- Personalized voice assistants  
- Audiobook/dubbing in cloned voices  
- Accessibility tools (expressive TTS)  
- Creative media & entertainment  

---

## ⚠️ Ethical Use
This project is for **academic and research purposes only**.  
Any use of cloned voices must have **explicit consent** of the speaker.  

---

✨ *Developed as a 7th Semester B.Tech CSE Project*  


