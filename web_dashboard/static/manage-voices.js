// Manage Voices Page JavaScript

class ManageVoicesApp {
    constructor() {
        this.apiBaseUrl = window.location.origin;
        this.voices = [];
        this.loadVoices();
    }
    
    async loadVoices() {
        const container = document.getElementById('voiceListContainer');
        container.innerHTML = '<div class="loading-container"><i class="fas fa-spinner fa-spin" style="font-size: 2rem;"></i><p>Loading voices...</p></div>';
        
        try {
            const response = await fetch(`${this.apiBaseUrl}/elevenlabs/voices`);
            
            if (!response.ok) {
                throw new Error(`Failed to load voices: ${response.status}`);
            }
            
            const data = await response.json();
            
            // Extract voices from response
            // ElevenLabs API returns: { "voices": [...] }
            const voices = data.voices?.voices || data.voices || [];
            
            this.voices = voices;
            
            // Also get voices from localStorage
            const localVoices = JSON.parse(localStorage.getItem('voicesList') || '[]');
            
            // Merge and deduplicate
            const allVoices = [...voices];
            localVoices.forEach(localVoice => {
                if (!allVoices.find(v => v.voice_id === localVoice.voice_id)) {
                    allVoices.push(localVoice);
                }
            });
            
            this.displayVoices(allVoices);
            
        } catch (error) {
            console.error('Error loading voices:', error);
            container.innerHTML = `
                <div class="empty-state">
                    <i class="fas fa-exclamation-triangle"></i>
                    <p>Failed to load voices. Please check your ElevenLabs API key.</p>
                    <p style="margin-top: 10px; font-size: 0.9rem;">${error.message}</p>
                </div>
            `;
        }
    }
    
    displayVoices(voices) {
        const container = document.getElementById('voiceListContainer');
        
        if (voices.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <i class="fas fa-folder-open"></i>
                    <h3>No Voices Found</h3>
                    <p>You haven't created any voices yet.</p>
                    <a href="/create-voice" style="display: inline-block; margin-top: 20px; padding: 10px 20px; background: #667eea; color: white; text-decoration: none; border-radius: 5px;">
                        <i class="fas fa-plus-circle"></i> Create Your First Voice
                    </a>
                </div>
            `;
            return;
        }
        
        container.innerHTML = `
            <div class="voice-list">
                ${voices.map(voice => this.createVoiceCard(voice)).join('')}
            </div>
        `;
        
        // Add event listeners to buttons
        voices.forEach(voice => {
            const selectBtn = document.getElementById(`select-${voice.voice_id}`);
            const deleteBtn = document.getElementById(`delete-${voice.voice_id}`);
            
            if (selectBtn) {
                selectBtn.addEventListener('click', () => this.selectVoice(voice));
            }
            
            if (deleteBtn) {
                deleteBtn.addEventListener('click', () => this.deleteVoice(voice.voice_id));
            }
        });
    }
    
    createVoiceCard(voice) {
        const voiceId = voice.voice_id;
        const voiceName = voice.name || voice.voice_name || 'Unnamed Voice';
        const description = voice.description || voice.labels?.description || '';
        const category = voice.category || '';
        
        return `
            <div class="voice-card">
                <h3>${this.escapeHtml(voiceName)}</h3>
                ${category ? `<p style="color: #666; font-size: 0.9rem; margin-bottom: 10px;">${this.escapeHtml(category)}</p>` : ''}
                ${description ? `<p style="color: #666; font-size: 0.9rem; margin-bottom: 10px;">${this.escapeHtml(description)}</p>` : ''}
                <div class="voice-id">
                    <strong>Voice ID:</strong><br>
                    <code>${voiceId}</code>
                </div>
                <div class="voice-actions">
                    <button class="btn-select" id="select-${voiceId}">
                        <i class="fas fa-check"></i> Use for Cloning
                    </button>
                    <button class="btn-delete" id="delete-${voiceId}">
                        <i class="fas fa-trash"></i> Delete
                    </button>
                </div>
            </div>
        `;
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    selectVoice(voice) {
        // Store selected voice in localStorage
        localStorage.setItem('selectedVoiceId', voice.voice_id);
        localStorage.setItem('selectedVoiceName', voice.name || voice.voice_name || 'Unnamed Voice');
        
        // Redirect to clone page
        window.location.href = `/clone-voice?voice_id=${voice.voice_id}`;
    }
    
    async deleteVoice(voiceId) {
        if (!confirm(`Are you sure you want to delete this voice? This action cannot be undone.`)) {
            return;
        }
        
        try {
            const response = await fetch(`${this.apiBaseUrl}/elevenlabs/voices/${voiceId}`, {
                method: 'DELETE'
            });
            
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail || `Failed to delete voice: ${response.status}`);
            }
            
            // Remove from localStorage
            let voicesList = JSON.parse(localStorage.getItem('voicesList') || '[]');
            voicesList = voicesList.filter(v => v.voice_id !== voiceId);
            localStorage.setItem('voicesList', JSON.stringify(voicesList));
            
            // Reload voices
            this.loadVoices();
            
            this.showSuccess('Voice deleted successfully');
            
        } catch (error) {
            console.error('Error deleting voice:', error);
            this.showError(`Failed to delete voice: ${error.message}`);
        }
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

// Make loadVoices available globally for the refresh button
let manageVoicesApp;

// Initialize app when page loads
document.addEventListener('DOMContentLoaded', () => {
    manageVoicesApp = new ManageVoicesApp();
});

// Global function for refresh button
function loadVoices() {
    if (manageVoicesApp) {
        manageVoicesApp.loadVoices();
    }
}

