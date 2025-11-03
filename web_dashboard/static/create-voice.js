// Create Voice Page JavaScript

class CreateVoiceApp {
    constructor() {
        this.apiBaseUrl = window.location.origin;
        this.currentAudioFile = null;
        this.currentUploadType = 'file'; // 'file' or 'url'
        this.initializeEventListeners();
    }
    
    initializeEventListeners() {
        // Toggle between file and URL upload
        document.getElementById('fileToggle').addEventListener('click', () => this.setUploadType('file'));
        document.getElementById('urlToggle').addEventListener('click', () => this.setUploadType('url'));
        
        // Voice create form
        document.getElementById('voiceCreateForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.handleVoiceCreate();
        });
        
        // File input change
        document.getElementById('voiceFile').addEventListener('change', (e) => {
            this.handleFileSelection(e);
        });
    }
    
    setUploadType(type) {
        this.currentUploadType = type;
        
        // Update toggle buttons
        document.getElementById('fileToggle').classList.toggle('active', type === 'file');
        document.getElementById('urlToggle').classList.toggle('active', type === 'url');
        
        // Update form options
        document.getElementById('fileUploadOption').classList.toggle('active', type === 'file');
        document.getElementById('urlUploadOption').classList.toggle('active', type === 'url');
        
        // Clear file input if switching
        if (type === 'url') {
            document.getElementById('voiceFile').value = '';
            this.currentAudioFile = null;
        } else {
            document.getElementById('audioUrl').value = '';
        }
    }
    
    handleFileSelection(event) {
        const file = event.target.files[0];
        if (file) {
            this.currentAudioFile = file;
            this.updateFileLabel(file.name);
        }
    }
    
    updateFileLabel(filename) {
        const label = document.querySelector('.file-upload-label span');
        if (label) {
            label.textContent = filename;
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
    
    async handleVoiceCreate() {
        const voiceName = document.getElementById('voiceName').value.trim();
        if (!voiceName) {
            this.showError('Please enter a voice name.');
            return;
        }
        
        const voiceLanguage = document.getElementById('voiceLanguage').value;
        if (!voiceLanguage) {
            this.showError('Please select a language for the voice.');
            return;
        }
        
        // Validate based on upload type
        if (this.currentUploadType === 'file') {
            if (!this.currentAudioFile) {
                this.showError('Please select an audio file first.');
                return;
            }
            
            // Validate file size (10MB limit)
            const maxSize = 10 * 1024 * 1024;
            if (this.currentAudioFile.size > maxSize) {
                this.showError(`File size must be less than 10MB. Your file is ${(this.currentAudioFile.size / (1024 * 1024)).toFixed(2)}MB`);
                return;
            }
        } else {
            const audioUrl = document.getElementById('audioUrl').value.trim();
            if (!audioUrl) {
                this.showError('Please enter an audio file URL.');
                return;
            }
            
            // Basic URL validation
            try {
                new URL(audioUrl);
            } catch (e) {
                this.showError('Please enter a valid URL.');
                return;
            }
        }
        
        this.showLoading('uploadLoading', 'uploadProgress');
        
        try {
            const formData = new FormData();
            
            if (this.currentUploadType === 'file') {
                formData.append('audio_file', this.currentAudioFile);
            } else {
                const audioUrl = document.getElementById('audioUrl').value.trim();
                formData.append('audio_url', audioUrl);
            }
            
            formData.append('voice_name', voiceName);
            formData.append('voice_language', voiceLanguage);
            
            console.log('Sending request to create voice...');
            if (this.currentUploadType === 'file') {
                console.log('File:', this.currentAudioFile.name, 'Size:', this.currentAudioFile.size, 'Type:', this.currentAudioFile.type);
            } else {
                console.log('URL:', document.getElementById('audioUrl').value);
            }
            console.log('Voice name:', voiceName, 'Language:', voiceLanguage);
            
            const response = await fetch(`${this.apiBaseUrl}/create-voice`, {
                method: 'POST',
                body: formData
            });
            
            console.log('Response status:', response.status, response.statusText);
            
            if (!response.ok) {
                let errorMsg = `HTTP error! status: ${response.status}`;
                try {
                    const errorData = await response.json();
                    console.error('Error response data:', errorData);
                    errorMsg = errorData.detail || errorData.message || errorMsg;
                } catch (e) {
                    const errorText = await response.text().catch(() => 'Unable to read error response');
                    console.error('Error response text:', errorText);
                    errorMsg = errorText || errorMsg;
                }
                throw new Error(errorMsg);
            }
            
            const result = await response.json();
            
            // Store voice ID and name in localStorage
            localStorage.setItem('lastCreatedVoiceId', result.voice_id);
            localStorage.setItem('lastCreatedVoiceName', result.voice_name || voiceName);
            
            // Also store in a voices list
            let voicesList = JSON.parse(localStorage.getItem('voicesList') || '[]');
            const voiceEntry = {
                voice_id: result.voice_id,
                voice_name: result.voice_name || voiceName,
                created_at: new Date().toISOString()
            };
            
            // Check if voice already exists in list
            const existingIndex = voicesList.findIndex(v => v.voice_id === result.voice_id);
            if (existingIndex >= 0) {
                voicesList[existingIndex] = voiceEntry;
            } else {
                voicesList.push(voiceEntry);
            }
            localStorage.setItem('voicesList', JSON.stringify(voicesList));
            
            // Display voice creation info
            document.getElementById('createdVoiceId').textContent = result.voice_id;
            document.getElementById('createdVoiceName').textContent = result.voice_name || voiceName;
            document.getElementById('voiceCreatedInfo').style.display = 'block';
            
            console.log('Voice created successfully! Voice ID stored:', result.voice_id);
            
            this.hideLoading('uploadLoading', 'uploadProgress');
            this.showSuccess('Voice created successfully in ElevenLabs!');
            
        } catch (error) {
            console.error('Error creating voice:', error);
            this.hideLoading('uploadLoading', 'uploadProgress');
            
            let errorMessage = 'Failed to create voice. ';
            if (error.message) {
                errorMessage += error.message;
            } else {
                errorMessage += 'Please check that your ElevenLabs API key is set correctly and the audio file is under 10MB.';
            }
            
            console.error('Detailed error:', errorMessage);
            this.showError(errorMessage);
        }
    }
}

// Initialize app when page loads
let createVoiceApp;

document.addEventListener('DOMContentLoaded', () => {
    createVoiceApp = new CreateVoiceApp();
});

