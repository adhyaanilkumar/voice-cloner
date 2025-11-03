// Clone Voice Page JavaScript

class CloneVoiceApp {
    constructor() {
        this.apiBaseUrl = window.location.origin;
        this.selectedVoiceId = null;
        this.generatedAudioUrl = null;
        this.loadVoices();
        this.initializeEventListeners();
    }
    
    initializeEventListeners() {
        // Voice cloning form
        document.getElementById('voiceCloneForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.handleVoiceCloning();
        });
        
        // Download button
        document.getElementById('downloadBtn').addEventListener('click', () => {
            this.downloadAudio();
        });
        
        // Voice select change
        document.getElementById('voiceSelect').addEventListener('change', (e) => {
            this.selectedVoiceId = e.target.value;
        });
        
        // Check if voice_id is in URL query params
        const urlParams = new URLSearchParams(window.location.search);
        const voiceIdFromUrl = urlParams.get('voice_id');
        if (voiceIdFromUrl) {
            setTimeout(() => {
                const selectEl = document.getElementById('voiceSelect');
                if (selectEl) {
                    selectEl.value = voiceIdFromUrl;
                    this.selectedVoiceId = voiceIdFromUrl;
                }
            }, 500); // Wait for voices to load
        }
    }
    
    async loadVoices() {
        const selectEl = document.getElementById('voiceSelect');
        selectEl.innerHTML = '<option value="">Loading voices...</option>';
        selectEl.disabled = true;
        
        try {
            const response = await fetch(`${this.apiBaseUrl}/elevenlabs/voices`);
            
            if (!response.ok) {
                throw new Error(`Failed to load voices: ${response.status}`);
            }
            
            const data = await response.json();
            const voices = data.voices?.voices || data.voices || [];
            
            // Also get voices from localStorage
            const localVoices = JSON.parse(localStorage.getItem('voicesList') || '[]');
            
            // Merge and deduplicate
            const allVoices = [...voices];
            localVoices.forEach(localVoice => {
                if (!allVoices.find(v => v.voice_id === localVoice.voice_id)) {
                    allVoices.push(localVoice);
                }
            });
            
            if (allVoices.length === 0) {
                selectEl.innerHTML = '<option value="">No voices found. Create one first.</option>';
                return;
            }
            
            selectEl.innerHTML = '<option value="">Select a voice...</option>' +
                allVoices.map(voice => {
                    const name = voice.name || voice.voice_name || 'Unnamed Voice';
                    return `<option value="${voice.voice_id}">${this.escapeHtml(name)} (${voice.voice_id})</option>`;
                }).join('');
            
            selectEl.disabled = false;
            
            // Check if there's a selected voice in localStorage
            const selectedVoiceId = localStorage.getItem('selectedVoiceId');
            if (selectedVoiceId && allVoices.find(v => v.voice_id === selectedVoiceId)) {
                selectEl.value = selectedVoiceId;
                this.selectedVoiceId = selectedVoiceId;
            }
            
        } catch (error) {
            console.error('Error loading voices:', error);
            selectEl.innerHTML = '<option value="">Error loading voices</option>';
        }
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    async handleVoiceCloning() {
        const text = document.getElementById('textInput').value.trim();
        if (!text) {
            this.showError('Please enter text to synthesize.');
            return;
        }
        
        const voiceId = document.getElementById('voiceSelect').value;
        if (!voiceId) {
            this.showError('Please select a voice.');
            return;
        }
        
        const stability = document.getElementById('stabilitySelect').value;
        const useEnhancement = document.getElementById('enhancementCheck').checked;
        
        this.showLoading('cloneLoading', 'cloneProgress');
        
        try {
            const formData = new FormData();
            formData.append('text', text);
            formData.append('voice_id', voiceId);
            formData.append('stability', stability);
            formData.append('use_enhancement', useEnhancement ? 'true' : 'false');
            
            console.log('Cloning voice with:', { voiceId, stability, useEnhancement });
            
            const response = await fetch(`${this.apiBaseUrl}/clone-voice`, {
                method: 'POST',
                body: formData
            });
            
            console.log('Response status:', response.status);
            
            if (!response.ok) {
                let errorMsg = `HTTP error! status: ${response.status}`;
                try {
                    const errorData = await response.json();
                    errorMsg = errorData.detail || errorData.message || errorMsg;
                } catch (e) {
                    const errorText = await response.text().catch(() => 'Unable to read error response');
                    errorMsg = errorText || errorMsg;
                }
                throw new Error(errorMsg);
            }
            
            const blob = await response.blob();
            const audioUrl = URL.createObjectURL(blob);
            this.generatedAudioUrl = audioUrl;
            
            // Display results
            document.getElementById('usedVoiceId').textContent = voiceId;
            document.getElementById('generatedAudio').src = audioUrl;
            document.getElementById('results').style.display = 'block';
            
            // Scroll to results
            document.getElementById('results').scrollIntoView({ behavior: 'smooth' });
            
            this.hideLoading('cloneLoading', 'cloneProgress');
            this.showSuccess('Voice cloned successfully!');
            
        } catch (error) {
            console.error('Error cloning voice:', error);
            this.hideLoading('cloneLoading', 'cloneProgress');
            this.showError(`Failed to clone voice: ${error.message}`);
        }
    }
    
    downloadAudio() {
        if (this.generatedAudioUrl) {
            const a = document.createElement('a');
            a.href = this.generatedAudioUrl;
            a.download = `cloned_voice_${Date.now()}.mp3`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
        }
    }
    
    showLoading(loadingId, progressId) {
        if (loadingId) document.getElementById(loadingId).style.display = 'block';
        if (progressId) document.getElementById(progressId).style.display = 'block';
    }
    
    hideLoading(loadingId, progressId) {
        if (loadingId) document.getElementById(loadingId).style.display = 'none';
        if (progressId) document.getElementById(progressId).style.display = 'none';
    }
    
    showError(message) {
        const errorEl = document.getElementById('errorMessage');
        errorEl.textContent = message;
        errorEl.style.display = 'block';
        setTimeout(() => {
            errorEl.style.display = 'none';
        }, 5000);
    }
    
    showSuccess(message) {
        const successEl = document.getElementById('successMessage');
        successEl.textContent = message;
        successEl.style.display = 'block';
        setTimeout(() => {
            successEl.style.display = 'none';
        }, 5000);
    }
}

// Initialize app when page loads
document.addEventListener('DOMContentLoaded', () => {
    new CloneVoiceApp();
});

