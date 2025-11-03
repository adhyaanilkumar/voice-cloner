// Voice Cloner Web Dashboard JavaScript

class VoiceClonerApp {
    constructor() {
        // Use the same origin/port the dashboard was served from
        this.apiBaseUrl = window.location.origin;
        this.currentAudioFile = null;
        this.createdVoiceId = null;  // Store the created voice ID
        this.createdVoiceName = null;
        this.stats = {
            totalProcessed: 0,
            successful: 0,
            totalProcessingTime: 0,
            voicesCreated: 0
        };
        
        this.initializeEventListeners();
        this.loadStats();
        this.loadLastCreatedVoice();  // Load previously created voice if available
    }
    
    loadLastCreatedVoice() {
        // Load last created voice from localStorage
        const lastVoiceId = localStorage.getItem('lastCreatedVoiceId');
        const lastVoiceName = localStorage.getItem('lastCreatedVoiceName');
        
        if (lastVoiceId && lastVoiceName) {
            this.createdVoiceId = lastVoiceId;
            this.createdVoiceName = lastVoiceName;
            
            // Display voice info
            document.getElementById('createdVoiceId').textContent = lastVoiceId;
            document.getElementById('createdVoiceName').textContent = lastVoiceName;
            document.getElementById('voiceCreatedInfo').style.display = 'block';
            
            // Update voice ID input in cloning form
            document.getElementById('voiceIdInput').value = lastVoiceId;
            document.getElementById('voiceIdGroup').style.display = 'block';
            
            // Enable clone button
            document.getElementById('cloneBtn').disabled = false;
            
            console.log('Loaded previously created voice:', lastVoiceId);
        }
    }
    
    initializeEventListeners() {
        // Voice create form
        document.getElementById('voiceCreateForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.handleVoiceCreate();
        });
        
        // Voice cloning form
        document.getElementById('voiceCloneForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.handleVoiceCloning();
        });
        
        // File input change
        document.getElementById('voiceFile').addEventListener('change', (e) => {
            this.handleFileSelection(e);
        });
        
        // Download button
        document.getElementById('downloadBtn').addEventListener('click', () => {
            this.downloadAudio();
        });
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
        label.textContent = filename;
    }
    
    async handleVoiceCreate() {
        if (!this.currentAudioFile) {
            this.showError('Please select an audio file first.');
            return;
        }
        
        const voiceName = document.getElementById('voiceName').value.trim();
        if (!voiceName) {
            this.showError('Please enter a voice name.');
            return;
        }
        
        this.showLoading('uploadLoading', 'uploadProgress');
        
        const voiceLanguage = document.getElementById('voiceLanguage').value;
        if (!voiceLanguage) {
            this.showError('Please select a language for the voice.');
            return;
        }
        
        try {
            const formData = new FormData();
            formData.append('audio_file', this.currentAudioFile);
            formData.append('voice_name', voiceName);
            formData.append('voice_language', voiceLanguage);
            
            console.log('Sending request to create voice...');
            console.log('File:', this.currentAudioFile.name, 'Size:', this.currentAudioFile.size, 'Type:', this.currentAudioFile.type);
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
            
            // Store voice ID and name
            this.createdVoiceId = result.voice_id;
            this.createdVoiceName = result.voice_name || voiceName;
            
            // Store in localStorage for persistence across page reloads
            localStorage.setItem('lastCreatedVoiceId', this.createdVoiceId);
            localStorage.setItem('lastCreatedVoiceName', this.createdVoiceName);
            
            // Display voice creation info
            document.getElementById('createdVoiceId').textContent = this.createdVoiceId;
            document.getElementById('createdVoiceName').textContent = this.createdVoiceName;
            document.getElementById('voiceCreatedInfo').style.display = 'block';
            
            // Update voice ID input in cloning form
            document.getElementById('voiceIdInput').value = this.createdVoiceId;
            document.getElementById('voiceIdGroup').style.display = 'block';
            
            // Enable clone button
            document.getElementById('cloneBtn').disabled = false;
            
            console.log('Voice created successfully! Voice ID stored:', this.createdVoiceId);
            
            this.hideLoading('uploadLoading', 'uploadProgress');
            this.showSuccess('Voice created successfully in ElevenLabs! You can now use it for cloning.');
            
            // Update stats
            this.stats.voicesCreated++;
            document.getElementById('voicesCreated').textContent = this.stats.voicesCreated;
            this.saveStats();
            
        } catch (error) {
            console.error('Error creating voice:', error);
            this.hideLoading('uploadLoading', 'uploadProgress');
            
            // Extract detailed error message from response
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
    
    async handleVoiceCloning() {
        if (!this.createdVoiceId) {
            this.showError('Please create a voice first using the "Create Voice" section above.');
            return;
        }
        
        const text = document.getElementById('textInput').value.trim();
        if (!text) {
            this.showError('Please enter text to synthesize.');
            return;
        }
        
        this.showLoading('cloneLoading', 'cloneProgress');
        
        try {
            const formData = new FormData();
            formData.append('text', text);
            formData.append('voice_id', this.createdVoiceId);  // Use created voice_id
            
            // Add voice cloning parameters for Eleven v3 (Alpha)
            const stability = document.getElementById('stabilitySelect').value;
            const use_enhancement = document.getElementById('enhancementCheck').checked;
            
            formData.append('stability', stability);
            formData.append('use_enhancement', use_enhancement ? 'true' : 'false');
            
            const response = await fetch(`${this.apiBaseUrl}/clone-voice`, {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                const errorMsg = errorData.detail || `HTTP error! status: ${response.status}`;
                throw new Error(errorMsg);
            }
            
            const result = await response.json();
            this.displayCloningResults(result);
            this.hideLoading('cloneLoading', 'cloneProgress');
            this.showSuccess('Voice cloning completed successfully!');
            
            // Update statistics
            this.updateStats(result);
            
        } catch (error) {
            console.error('Error cloning voice:', error);
            this.hideLoading('cloneLoading', 'cloneProgress');
            let errorMessage = 'Failed to clone voice using ElevenLabs API. ';
            if (error.message) {
                errorMessage = errorMessage + error.message;
            } else {
                errorMessage = errorMessage + 'Please check that your ElevenLabs API key is set correctly.';
            }
            this.showError(errorMessage);
        }
    }
    
    displayCloningResults(result) {
        // Show results section
        document.getElementById('results').style.display = 'block';
        
        // Update result display
        document.getElementById('usedVoiceId').textContent = this.createdVoiceId || '-';
        document.getElementById('processingTime').textContent = `${result.processing_time.toFixed(2)}s`;
        
        // Show audio player
        const audioElement = document.getElementById('generatedAudio');
        
        // Normalize path separators and extract filename
        const normalizedPath = result.audio_file_path.replace(/\\/g, '/');
        const filename = normalizedPath.split('/').pop();
        
        audioElement.src = `${this.apiBaseUrl}/download/${filename}`;
        
        // Store audio file path for download
        this.currentGeneratedAudio = result.audio_file_path;
        
        // Add error handling for audio loading
        audioElement.onerror = () => {
            console.error('Failed to load audio file:', filename);
            this.showError('Failed to load audio file. The generated audio may be corrupted.');
        };
        
        audioElement.onloadeddata = () => {
            console.log('Audio file loaded successfully:', filename);
        };
    }
    
    downloadAudio() {
        if (this.currentGeneratedAudio) {
            // Normalize path separators
            const normalizedPath = this.currentGeneratedAudio.replace(/\\/g, '/');
            const filename = normalizedPath.split('/').pop();
            const downloadUrl = `${this.apiBaseUrl}/download/${filename}`;
            
            // Create temporary link and trigger download
            const link = document.createElement('a');
            link.href = downloadUrl;
            link.download = `cloned_voice_${Date.now()}.wav`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }
    }
    
    showLoading(loadingId, progressId) {
        document.getElementById(loadingId).style.display = 'block';
        document.getElementById(progressId).style.display = 'block';
        
        // Simulate progress
        let progress = 0;
        const interval = setInterval(() => {
            progress += Math.random() * 15;
            if (progress > 90) progress = 90;
            
            document.getElementById(progressId + 'Fill').style.width = progress + '%';
            
            if (progress >= 90) {
                clearInterval(interval);
            }
        }, 200);
        
        this.progressInterval = interval;
    }
    
    hideLoading(loadingId, progressId) {
        document.getElementById(loadingId).style.display = 'none';
        
        // Complete progress bar
        if (this.progressInterval) {
            clearInterval(this.progressInterval);
        }
        document.getElementById(progressId + 'Fill').style.width = '100%';
        
        setTimeout(() => {
            document.getElementById(progressId).style.display = 'none';
            document.getElementById(progressId + 'Fill').style.width = '0%';
        }, 1000);
    }
    
    showError(message) {
        const errorDiv = document.getElementById('errorMessage');
        errorDiv.textContent = message;
        errorDiv.style.display = 'block';
        
        // Hide success message
        document.getElementById('successMessage').style.display = 'none';
        
        // Auto-hide after 5 seconds
        setTimeout(() => {
            errorDiv.style.display = 'none';
        }, 5000);
    }
    
    showSuccess(message) {
        const successDiv = document.getElementById('successMessage');
        successDiv.textContent = message;
        successDiv.style.display = 'block';
        
        // Hide error message
        document.getElementById('errorMessage').style.display = 'none';
        
        // Auto-hide after 3 seconds
        setTimeout(() => {
            successDiv.style.display = 'none';
        }, 3000);
    }
    
    updateStats(result) {
        this.stats.totalProcessed++;
        this.stats.successful++;
        this.stats.totalProcessingTime += result.processing_time;
        
        // Update display
        document.getElementById('totalProcessed').textContent = this.stats.totalProcessed;
        document.getElementById('successRate').textContent = 
            `${((this.stats.successful / this.stats.totalProcessed) * 100).toFixed(1)}%`;
        document.getElementById('avgProcessingTime').textContent = 
            `${(this.stats.totalProcessingTime / this.stats.totalProcessed).toFixed(1)}s`;
        
        // Save to localStorage
        this.saveStats();
    }
    
    loadStats() {
        const savedStats = localStorage.getItem('voiceClonerStats');
        if (savedStats) {
            this.stats = { ...this.stats, ...JSON.parse(savedStats) };
            
            // Update display
            document.getElementById('totalProcessed').textContent = this.stats.totalProcessed;
            document.getElementById('successRate').textContent = 
                this.stats.totalProcessed > 0 ? 
                `${((this.stats.successful / this.stats.totalProcessed) * 100).toFixed(1)}%` : '0%';
            document.getElementById('avgProcessingTime').textContent = 
                this.stats.totalProcessed > 0 ? 
                `${(this.stats.totalProcessingTime / this.stats.totalProcessed).toFixed(1)}s` : '0s';
            document.getElementById('voicesCreated').textContent = this.stats.voicesCreated || 0;
        }
    }
    
    saveStats() {
        localStorage.setItem('voiceClonerStats', JSON.stringify(this.stats));
    }
    
    // Utility methods
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
    
    validateAudioFile(file) {
        const allowedTypes = ['audio/wav', 'audio/mp3', 'audio/flac', 'audio/mpeg', 'audio/m4a', 'audio/ogg'];
        const maxSize = 10 * 1024 * 1024; // 10MB (ElevenLabs limit)
        
        if (!allowedTypes.includes(file.type)) {
            throw new Error('Please select a valid audio file (WAV, MP3, FLAC, M4A, OGG).');
        }
        
        if (file.size > maxSize) {
            const fileSizeMB = (file.size / (1024 * 1024)).toFixed(2);
            throw new Error(`File size must be less than 10MB. Your file is ${fileSizeMB}MB.`);
        }
        
        return true;
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new VoiceClonerApp();
});

// Add drag and drop functionality
document.addEventListener('DOMContentLoaded', () => {
    const fileUpload = document.querySelector('.file-upload-label');
    
    fileUpload.addEventListener('dragover', (e) => {
        e.preventDefault();
        fileUpload.style.background = '#667eea';
        fileUpload.style.color = 'white';
    });
    
    fileUpload.addEventListener('dragleave', (e) => {
        e.preventDefault();
        fileUpload.style.background = '#f8f9ff';
        fileUpload.style.color = '#667eea';
    });
    
    fileUpload.addEventListener('drop', (e) => {
        e.preventDefault();
        fileUpload.style.background = '#f8f9ff';
        fileUpload.style.color = '#667eea';
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            const fileInput = document.getElementById('voiceFile');
            fileInput.files = files;
            fileInput.dispatchEvent(new Event('change'));
        }
    });
});
