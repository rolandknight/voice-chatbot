// ui/script.js
// Client-side JavaScript for the Chatterbox TTS Server web interface.
// Handles UI interactions, API communication, audio playback, and settings management.

document.addEventListener('DOMContentLoaded', async function () {
    // --- Global Flags & State ---
    let uiReady = false;
    let listenersAttached = false;
    let isGenerating = false;
    let saveStateTimeout = null;
    let currentPresetName = null;

    let currentConfig = {};
    let currentUiState = {};
    let appPresets = [];
    let initialReferenceFiles = [];
    let initialPredefinedVoices = [];

    // Model information state
    let currentModelInfo = null;
    let selectedModelSelector = 'chatterbox-turbo';
    let modelChangesPending = false;

    let hideChunkWarning = false;
    let hideGenerationWarning = false;
    let currentVoiceMode = 'predefined';

    const IS_LOCAL_FILE = window.location.protocol === 'file:';
    // If you always access the server via localhost
    const API_BASE_URL = IS_LOCAL_FILE ? 'http://localhost:8004' : '';

    const DEBOUNCE_DELAY_MS = 750;

    // --- DOM Element Selectors ---
    const appTitleLink = document.getElementById('app-title-link');
    const themeToggleButton = document.getElementById('theme-toggle-btn');
    const themeSwitchThumb = themeToggleButton ? themeToggleButton.querySelector('.theme-switch-thumb') : null;
    const notificationArea = document.getElementById('notification-area');
    const ttsForm = document.getElementById('tts-form');
    const ttsFormHeader = document.getElementById('tts-form-header');
    const textArea = document.getElementById('text');
    const charCount = document.getElementById('char-count');
    const generateBtn = document.getElementById('generate-btn');
    const splitTextToggle = document.getElementById('split-text-toggle');
    const chunkSizeControls = document.getElementById('chunk-size-controls');
    const chunkSizeSlider = document.getElementById('chunk-size-slider');
    const chunkSizeValue = document.getElementById('chunk-size-value');
    const chunkExplanation = document.getElementById('chunk-explanation');
    const voiceModeRadios = document.querySelectorAll('input[name="voice_mode"]');
    const predefinedVoiceOptionsDiv = document.getElementById('predefined-voice-options');
    const predefinedVoiceSelect = document.getElementById('predefined-voice-select');
    const predefinedVoiceImportButton = document.getElementById('predefined-voice-import-button');
    const predefinedVoiceRefreshButton = document.getElementById('predefined-voice-refresh-button');
    const predefinedVoiceFileInput = document.getElementById('predefined-voice-file-input');
    const cloneOptionsDiv = document.getElementById('clone-options');
    const cloneReferenceSelect = document.getElementById('clone-reference-select');
    const cloneImportButton = document.getElementById('clone-import-button');
    const cloneRefreshButton = document.getElementById('clone-refresh-button');
    const cloneFileInput = document.getElementById('clone-file-input');
    const presetsContainer = document.getElementById('presets-container');
    const presetsPlaceholder = document.getElementById('presets-placeholder');
    const temperatureSlider = document.getElementById('temperature');
    const temperatureValueDisplay = document.getElementById('temperature-value');
    const exaggerationSlider = document.getElementById('exaggeration');
    const exaggerationValueDisplay = document.getElementById('exaggeration-value');
    const cfgWeightSlider = document.getElementById('cfg-weight');
    const cfgWeightValueDisplay = document.getElementById('cfg-weight-value');
    const numStepsSlider = document.getElementById('num-steps');
    const cfmTimestepsSlider = document.getElementById('cfm-timesteps');
    const saveGenDefaultsBtn = document.getElementById('save-gen-defaults-btn');
    const genDefaultsStatus = document.getElementById('gen-defaults-status');
    const serverConfigForm = document.getElementById('server-config-form');
    const saveConfigBtn = document.getElementById('save-config-btn');
    const restartServerBtn = document.getElementById('restart-server-btn');
    const configStatus = document.getElementById('config-status');
    const resetSettingsBtn = document.getElementById('reset-settings-btn');
    const loadingOverlay = document.getElementById('loading-overlay');
    const loadingMessage = document.getElementById('loading-message');
    const loadingStatusText = document.getElementById('loading-status');
    const loadingCancelBtn = document.getElementById('loading-cancel-btn');
    const chunkWarningModal = document.getElementById('chunk-warning-modal');
    const chunkWarningOkBtn = document.getElementById('chunk-warning-ok');
    const chunkWarningCancelBtn = document.getElementById('chunk-warning-cancel');
    const hideChunkWarningCheckbox = document.getElementById('hide-chunk-warning-checkbox');
    const generationWarningModal = document.getElementById('generation-warning-modal');
    const generationWarningAcknowledgeBtn = document.getElementById('generation-warning-acknowledge');
    const hideGenerationWarningCheckbox = document.getElementById('hide-generation-warning-checkbox');

    // Model-related elements
    const modelIndicator = document.getElementById('model-indicator');
    const modelBadge = document.getElementById('model-badge');
    const modelBadgeIcon = document.getElementById('model-badge-icon');
    const modelBadgeText = document.getElementById('model-badge-text');
    const modelSelect = document.getElementById('model-select');
    const modelStatusIndicator = document.getElementById('model-status-indicator');
    const modelStatusText = document.getElementById('model-status-text');
    const applyModelBtn = document.getElementById('apply-model-btn');
    const paralinguisticTagsSection = document.getElementById('paralinguistic-tags-section');
    const tagButtons = document.querySelectorAll('.tag-btn');


    // Handle voice mode selection visual feedback
    const voiceModeOptions = document.querySelectorAll('.voice-mode__option');

    voiceModeRadios.forEach(radio => {
        radio.addEventListener('change', function () {
            // Remove selected class from all options
            voiceModeOptions.forEach(option => {
                option.classList.remove('selected');
            });

            // Add selected class to the parent of the checked radio
            // CORRECTED: Selector updated to match HTML
            const selectedOption = this.closest('.voice-mode__option');
            if (selectedOption) {
                selectedOption.classList.add('selected');
            }
        });
    });

    // Set initial state
    const checkedRadio = document.querySelector('input[name="voice_mode"]:checked');
    if (checkedRadio) {
        // CORRECTED: Selector updated to match HTML
        const selectedOption = checkedRadio.closest('.voice-mode__option');
        if (selectedOption) {
            selectedOption.classList.add('selected');
        }
    }

    // --- Utility Functions ---
    function formatErrorDetail(detail) {
        if (typeof detail === 'string') return detail;
        if (Array.isArray(detail)) return detail.map(e => e.msg || JSON.stringify(e)).join('; ');
        if (detail && typeof detail === 'object') return JSON.stringify(detail);
        return String(detail);
    }

    function showNotification(message, type = 'info', duration = 5000) {
        if (!notificationArea) return null;

        const icons = {
            success: '<svg class="notification__icon" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd" /></svg>',
            error: '<svg class="notification__icon" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd" /></svg>',
            warning: '<svg class="notification__icon" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M8.485 2.495c.673-1.167 2.357-1.167 3.03 0l6.28 10.875c.673 1.167-.17 2.625-1.516 2.625H3.72c-1.347 0-2.189-1.458-1.515-2.625L8.485 2.495zM10 5a.75.75 0 01.75.75v3.5a.75.75 0 01-1.5 0v-3.5A.75.75 0 0110 5zm0 9a1 1 0 100-2 1 1 0 000 2z" clip-rule="evenodd" /></svg>',
            info: '<svg class="notification__icon" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a.75.75 0 000 1.5h.253a.25.25 0 01.244.304l-.459 2.066A1.75 1.75 0 0010.747 15H11a.75.75 0 000-1.5h-.253a.25.25 0 01-.244-.304l.459-2.066A1.75 1.75 0 009.253 9H9z" clip-rule="evenodd" /></svg>'
        };

        const notificationDiv = document.createElement('div');
        notificationDiv.className = `notification ${type}`;
        notificationDiv.setAttribute('role', 'alert');

        // Build notification structure
        notificationDiv.innerHTML = `
            ${icons[type] || icons['info']}
            <div class="notification__content"><span>${message}</span></div>
        `;

        // Create close button
        const closeButton = document.createElement('button');
        closeButton.type = 'button';
        closeButton.className = 'notification__close';
        closeButton.innerHTML = `
            <span class="sr-only">Close</span>
            <svg class="notification__close-icon" fill="currentColor" viewBox="0 0 20 20">
                <path fill-rule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clip-rule="evenodd"></path>
            </svg>
        `;
        closeButton.onclick = () => {
            notificationDiv.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
            notificationDiv.style.opacity = '0';
            notificationDiv.style.transform = 'translateY(-20px)';
            setTimeout(() => notificationDiv.remove(), 300);
        };

        notificationDiv.appendChild(closeButton);
        notificationArea.appendChild(notificationDiv);

        if (duration > 0) {
            setTimeout(() => closeButton.click(), duration);
        }

        return notificationDiv;
    }

    // --- Theme Management ---
    function applyTheme(theme) {
        const isDark = theme === 'dark';
        document.documentElement.classList.toggle('dark', isDark);
        localStorage.setItem('uiTheme', theme);
    }

    if (themeToggleButton) {
        themeToggleButton.addEventListener('click', () => {
            const newTheme = document.documentElement.classList.contains('dark') ? 'light' : 'dark';
            applyTheme(newTheme);
            debouncedSaveState();
        });
    }

    // --- UI State Persistence ---
    async function saveCurrentUiState() {
        const stateToSave = {
            last_text: textArea ? textArea.value : '',
            last_voice_mode: currentVoiceMode,
            last_predefined_voice: predefinedVoiceSelect ? predefinedVoiceSelect.value : null,
            last_reference_file: cloneReferenceSelect ? cloneReferenceSelect.value : null,
            last_chunk_size: chunkSizeSlider ? parseInt(chunkSizeSlider.value, 10) : 120,
            last_split_text_enabled: splitTextToggle ? splitTextToggle.checked : true,
            hide_chunk_warning: hideChunkWarning,
            hide_generation_warning: hideGenerationWarning,
            theme: localStorage.getItem('uiTheme') || 'dark',
            last_preset_name: currentPresetName,
        };

        try {
            const response = await fetch(`${API_BASE_URL}/save_settings`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ ui_state: stateToSave })
            });
            if (!response.ok) {
                const errorResult = await response.json();
                throw new Error(formatErrorDetail(errorResult.detail) || `Failed to save UI state (status ${response.status})`);
            }
        } catch (error) {
            console.error("Error saving UI state via API:", error);
            showNotification(`Error saving settings: ${error.message}. Some changes may not persist.`, 'error', 0);
        }
    }

    function debouncedSaveState() {
        // Do not save anything until the entire UI has finished its initial setup.
        if (!uiReady || !listenersAttached) { return; }
        clearTimeout(saveStateTimeout);
        saveStateTimeout = setTimeout(saveCurrentUiState, DEBOUNCE_DELAY_MS);
    }

    // --- Model Management Functions (New Features) ---

    function updateModelUI(modelInfo) {
        if (!modelInfo) {
            console.warn('updateModelUI called with null modelInfo');
            return;
        }

        currentModelInfo = modelInfo;

        // Update model indicator badge
        if (modelIndicator && modelBadge) {
            modelIndicator.classList.remove('hidden');

            // Use simplified modifier classes
            if (modelInfo.type === 'turbo') {
                modelBadge.className = 'model-badge turbo';
                modelBadgeText.textContent = '⚡ Turbo';
            } else if (modelInfo.type === 'multilingual') {
                modelBadge.className = 'model-badge multilingual';
                modelBadgeText.textContent = '🌍 Multilingual';
            } else {
                modelBadge.className = 'model-badge original';
                modelBadgeText.textContent = 'Original';
            }
        }

        // Update model status indicator
        if (modelStatusIndicator && modelStatusText) {
            if (modelInfo.loaded) {
                modelStatusIndicator.className = 'status-dot success';
                modelStatusText.textContent = `${modelInfo.class_name} loaded on ${modelInfo.device}`;
                modelStatusText.className = 'model-status__text success';
            } else {
                modelStatusIndicator.className = 'status-dot error';
                modelStatusText.textContent = 'Model not loaded';
                modelStatusText.className = 'model-status__text error';
            }
        }

        // Update model selector dropdown to match loaded model
        if (modelSelect && !modelChangesPending) {
            let selectorValue = 'chatterbox';
            if (modelInfo.type === 'turbo') {
                selectorValue = 'chatterbox-turbo';
            } else if (modelInfo.type === 'multilingual') {
                selectorValue = 'chatterbox-multilingual';
            }
            modelSelect.value = selectorValue;
            selectedModelSelector = selectorValue;
        }

        // Show/hide model-specific UI sections
        const exaggerationGroup = document.getElementById('exaggeration-group');
        const cfgWeightGroup = document.getElementById('cfg-weight-group');
        const flashKnobsGroup = document.getElementById('flash-knobs-group');

        // Show/hide paralinguistic tags section (Turbo only)
        if (paralinguisticTagsSection) {
            if (modelInfo.type === 'turbo' && modelInfo.supports_paralinguistic_tags) {
                paralinguisticTagsSection.classList.remove('hidden');
            } else {
                paralinguisticTagsSection.classList.add('hidden');
            }
        }

        // Hide exaggeration and CFG for turbo model
        if (modelInfo.type === 'turbo') {
            exaggerationGroup?.classList.add('hidden');
            cfgWeightGroup?.classList.add('hidden');
        } else {
            exaggerationGroup?.classList.remove('hidden');
            cfgWeightGroup?.classList.remove('hidden');
        }

        // Show diffusion/CFM step controls only for Flash
        flashKnobsGroup?.classList.toggle('hidden', modelInfo.type !== 'flash');

        // Refresh presets to filter based on current model type
        populatePresets();

        console.log('Model UI updated:', modelInfo);
    }

    function insertTagAtCursor(tag) {
        if (!textArea) return;

        const startPos = textArea.selectionStart;
        const endPos = textArea.selectionEnd;
        const textBefore = textArea.value.substring(0, startPos);
        const textAfter = textArea.value.substring(endPos);

        // Insert tag with a space after if not at end and next char isn't a space
        let insertText = tag;
        if (textAfter.length > 0 && textAfter[0] !== ' ') {
            insertText = tag + ' ';
        }

        textArea.value = textBefore + insertText + textAfter;

        // Update cursor position to after the inserted tag
        const newCursorPos = startPos + insertText.length;
        textArea.setSelectionRange(newCursorPos, newCursorPos);
        textArea.focus();

        // Update character count
        if (charCount) {
            charCount.textContent = textArea.value.length;
        }

        // Trigger state save
        debouncedSaveState();
    }

    function handleModelSelectChange() {
        if (!modelSelect) return;

        const newSelector = modelSelect.value;
        let currentSelector = 'chatterbox';
        if (currentModelInfo?.type === 'turbo') {
            currentSelector = 'chatterbox-turbo';
        } else if (currentModelInfo?.type === 'multilingual') {
            currentSelector = 'chatterbox-multilingual';
        }

        if (newSelector !== currentSelector) {
            modelChangesPending = true;

            // Show the apply button
            if (applyModelBtn) {
                applyModelBtn.classList.remove('hidden');
            }

            // Update status indicator and text to show pending state
            if (modelStatusIndicator) {
                modelStatusIndicator.className = 'status-dot warning';
            }
            if (modelStatusText) {
                modelStatusText.textContent = 'Model change pending - click Apply & Restart';
                modelStatusText.className = 'model-status__text warning';
            }
        } else {
            modelChangesPending = false;

            // Hide the apply button
            if (applyModelBtn) {
                applyModelBtn.classList.add('hidden');
            }

            // Restore status from current model info
            updateModelUI(currentModelInfo);
        }
    }


    async function applyModelChange() {
        if (!modelSelect) return;

        const newSelector = modelSelect.value;

        // Update status
        if (modelStatusText) {
            modelStatusText.textContent = 'Saving configuration...';
        }
        if (applyModelBtn) {
            applyModelBtn.disabled = true;
            applyModelBtn.innerHTML = `
                <svg class="btn__icon animate-spin" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                    <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Saving...
            `;
        }

        try {
            // Save the model selector to config
            const response = await fetch(`${API_BASE_URL}/save_settings`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    model: {
                        repo_id: newSelector
                    }
                })
            });

            if (!response.ok) {
                const errorResult = await response.json().catch(() => ({ detail: 'Failed to save' }));
                throw new Error(formatErrorDetail(errorResult.detail) || 'Failed to save model configuration');
            }

            showNotification('Model configuration saved. Initiating server restart...', 'info');

            // Trigger server restart
            const restartResponse = await fetch(`${API_BASE_URL}/restart_server`, {
                method: 'POST'
            });

            if (restartResponse.ok) {
                showNotification(
                    'Server restart initiated. The page will reload automatically in a few seconds...',
                    'success',
                    10000
                );

                // Attempt to reload after delay
                setTimeout(() => {
                    window.location.reload();
                }, 5000);
            } else {
                showNotification(
                    'Configuration saved. Please restart the server manually for changes to take effect.',
                    'warning',
                    0
                );
            }

        } catch (error) {
            console.error('Error applying model change:', error);
            showNotification(`Error: ${error.message}`, 'error');

            // Re-enable button
            if (applyModelBtn) {
                applyModelBtn.disabled = false;
                applyModelBtn.innerHTML = `
                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-4 h-4 mr-1">
                        <path stroke-linecap="round" stroke-linejoin="round" d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0 3.181 3.183a8.25 8.25 0 0 0 13.803-3.7M4.031 9.865a8.25 8.25 0 0 1 13.803-3.7l3.181 3.182m0-4.991v4.99" />
                    </svg>
                    Apply & Restart
                `;
            }
        }
    }


    // --- Initial Application Setup ---
    function initializeApplication() {
        const preferredTheme = localStorage.getItem('uiTheme') || currentUiState.theme || 'dark';
        applyTheme(preferredTheme);
        const pageTitle = currentConfig?.ui?.title || "Chatterbox TTS Server";
        document.title = pageTitle;
        if (appTitleLink) appTitleLink.textContent = pageTitle;
        if (ttsFormHeader) ttsFormHeader.textContent = `Generate Speech`;
        loadInitialUiState();
        populatePredefinedVoices();
        populateReferenceFiles();
        populatePresets();
        displayServerConfiguration();
    }

    async function fetchInitialData() {
        try {
            const response = await fetch(`${API_BASE_URL}/api/ui/initial-data`);
            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`Failed to fetch initial UI data: ${response.status} ${response.statusText}. Server response: ${errorText}`);
            }
            const data = await response.json();
            currentConfig = data.config || {};
            currentUiState = currentConfig.ui_state || {};
            appPresets = data.presets || [];
            initialReferenceFiles = data.reference_files || [];
            initialPredefinedVoices = data.predefined_voices || [];
            hideChunkWarning = currentUiState.hide_chunk_warning || false;
            hideGenerationWarning = currentUiState.hide_generation_warning || false;
            currentVoiceMode = currentUiState.last_voice_mode || 'predefined';

            // NEW: Handle model info from initial data
            if (data.model_info) {
                updateModelUI(data.model_info);
            }

            initializeApplication();

        } catch (error) {
            console.error("Error fetching initial data:", error);
            showNotification(`Could not load essential application data: ${error.message}. Please try refreshing.`, 'error', 0);
            if (Object.keys(currentConfig).length === 0) {
                currentConfig = { ui: { title: "Chatterbox TTS Server (Error Mode)" }, generation_defaults: {}, ui_state: {} };
                currentUiState = currentConfig.ui_state;
            }
            initializeApplication(); // Attempt to init in a degraded state
        } finally {
            // --- PHASE 2: Attach listeners and enable UI readiness ---
            // This pushes the listener attachment to the end of the event queue,
            // ensuring all initialization events have fired harmlessly before we start listening.
            setTimeout(() => {
                attachStateSavingListeners();
                listenersAttached = true;
                uiReady = true;
            }, 50); // A 50ms delay is more robust than 0ms for complex UIs.
        }
    }

    function loadInitialUiState() {
        // Restore any SAVED text, including a deliberately emptied box. Testing
        // truthiness here meant an explicit clear was indistinguishable from
        // "never saved", so the default preset was re-applied on every reload
        // and could never be dismissed.
        if (textArea && typeof currentUiState.last_text === 'string') {
            textArea.value = currentUiState.last_text;
            if (charCount) charCount.textContent = textArea.value.length;
        }

        // Handle Voice Mode Selection
        const modeRadioToSelect = document.querySelector(`input[name="voice_mode"][value="${currentVoiceMode}"]`);

        if (modeRadioToSelect) {
            modeRadioToSelect.checked = true;
            // FIX: Manually fire the change event so the .selected class updates visually
            modeRadioToSelect.dispatchEvent(new Event('change'));
        } else {
            const defaultRadio = document.querySelector('input[name="voice_mode"][value="predefined"]');
            if (defaultRadio) {
                defaultRadio.checked = true;
                currentVoiceMode = 'predefined';
                defaultRadio.dispatchEvent(new Event('change'));
            }
        }

        toggleVoiceOptionsDisplay();

        if (splitTextToggle) splitTextToggle.checked = currentUiState.last_split_text_enabled !== undefined ? currentUiState.last_split_text_enabled : true;

        if (chunkSizeSlider && currentUiState.last_chunk_size !== undefined) chunkSizeSlider.value = currentUiState.last_chunk_size;
        if (chunkSizeValue) chunkSizeValue.textContent = chunkSizeSlider ? chunkSizeSlider.value : '120';
        toggleChunkControlsVisibility();

        const genDefaults = currentConfig.generation_defaults || {};
        if (temperatureSlider) temperatureSlider.value = genDefaults.temperature !== undefined ? genDefaults.temperature : 0.8;
        if (temperatureValueDisplay) temperatureValueDisplay.textContent = temperatureSlider.value;
        if (exaggerationSlider) exaggerationSlider.value = genDefaults.exaggeration !== undefined ? genDefaults.exaggeration : 0.5;
        if (exaggerationValueDisplay) exaggerationValueDisplay.textContent = exaggerationSlider.value;
        if (cfgWeightSlider) cfgWeightSlider.value = genDefaults.cfg_weight !== undefined ? genDefaults.cfg_weight : 0.5;
        if (cfgWeightValueDisplay) cfgWeightValueDisplay.textContent = cfgWeightSlider.value;

        if (hideChunkWarningCheckbox) hideChunkWarningCheckbox.checked = hideChunkWarning;
        if (hideGenerationWarningCheckbox) hideGenerationWarningCheckbox.checked = hideGenerationWarning;

        // --- PRESET RESTORATION LOGIC ---

        // 1. Restore the name from state variable
        if (currentUiState.last_preset_name) {
            currentPresetName = currentUiState.last_preset_name;
        }

        // 2. Logic to apply preset (if empty) OR just highlight button (if text exists)
        const hasSavedText = typeof currentUiState.last_text === 'string';
        if (textArea && !textArea.value && !hasSavedText && appPresets && appPresets.length > 0) {
            // Case A: No text entered. We want to load a preset fully.
            // Priority: Saved preset > "Standard Narration" > First available
            const savedPreset = appPresets.find(p => p.name === currentPresetName);
            const defaultPreset = savedPreset || appPresets.find(p => p.name === "Standard Narration") || appPresets[0];

            if (defaultPreset) {
                // Apply values AND visuals, no notification, no save
                applyPreset(defaultPreset, false, false);
            }
        } else if (currentPresetName) {
            // Case B: Text already exists (restored from last_text). 
            // We don't want to overwrite parameters, but we want to show which preset button was active.
            updatePresetVisuals(currentPresetName);
        }
    }

    function attachStateSavingListeners() {
        voiceModeRadios.forEach(radio => {
            radio.addEventListener('change', debouncedSaveState);
        });

        if (textArea) textArea.addEventListener('input', () => { if (charCount) charCount.textContent = textArea.value.length; debouncedSaveState(); });
        if (predefinedVoiceSelect) predefinedVoiceSelect.addEventListener('change', debouncedSaveState);
        if (cloneReferenceSelect) cloneReferenceSelect.addEventListener('change', debouncedSaveState);
        if (splitTextToggle) splitTextToggle.addEventListener('change', () => { toggleChunkControlsVisibility(); debouncedSaveState(); });
        if (chunkSizeSlider) {
            chunkSizeSlider.addEventListener('input', () => { if (chunkSizeValue) chunkSizeValue.textContent = chunkSizeSlider.value; });
            chunkSizeSlider.addEventListener('change', debouncedSaveState);
        }
        const genParamSliders = [temperatureSlider, exaggerationSlider, cfgWeightSlider, numStepsSlider, cfmTimestepsSlider];
        genParamSliders.forEach(slider => {
            if (slider) {
                const valueDisplayId = slider.id + '-value';
                const valueDisplay = document.getElementById(valueDisplayId);
                slider.addEventListener('input', () => {
                    if (valueDisplay) valueDisplay.textContent = slider.value;
                });
                slider.addEventListener('change', debouncedSaveState);
            }
        });

        // NEW: Model management listeners
        if (modelSelect) {
            modelSelect.addEventListener('change', handleModelSelectChange);
        }

        if (applyModelBtn) {
            applyModelBtn.addEventListener('click', applyModelChange);
        }

        // NEW: Tag button listeners
        tagButtons.forEach(button => {
            button.addEventListener('click', (e) => {
                const tag = e.currentTarget.getAttribute('data-tag');
                if (tag) {
                    insertTagAtCursor(tag);
                }
            });
        });
    }

    // --- Dynamic UI Population ---
    function populatePredefinedVoices(voicesData = initialPredefinedVoices) {
        if (!predefinedVoiceSelect) return;
        const currentSelectedValue = predefinedVoiceSelect.value;
        predefinedVoiceSelect.innerHTML = '<option value="none">-- Select Voice --</option>';
        voicesData.forEach(voice => {
            const option = document.createElement('option');
            option.value = voice.filename;
            option.textContent = voice.display_name || voice.filename;
            predefinedVoiceSelect.appendChild(option);
        });
        const lastSelected = currentUiState.last_predefined_voice;
        const defaultFromConfig = currentConfig?.tts_engine?.default_voice_id;
        if (currentSelectedValue !== 'none' && voicesData.some(v => v.filename === currentSelectedValue)) {
            predefinedVoiceSelect.value = currentSelectedValue;
        } else if (lastSelected && voicesData.some(v => v.filename === lastSelected)) {
            predefinedVoiceSelect.value = lastSelected;
        } else if (defaultFromConfig && voicesData.some(v => v.filename === defaultFromConfig)) {
            predefinedVoiceSelect.value = defaultFromConfig;
        } else {
            predefinedVoiceSelect.value = 'none';
        }
    }

    function populateReferenceFiles(filesData = initialReferenceFiles) {
        if (!cloneReferenceSelect) return;
        const currentSelectedValue = cloneReferenceSelect.value;
        cloneReferenceSelect.innerHTML = '<option value="none">-- Select Reference File --</option>';
        filesData.forEach(filename => {
            const option = document.createElement('option');
            option.value = filename;
            option.textContent = filename;
            cloneReferenceSelect.appendChild(option);
        });
        const lastSelected = currentUiState.last_reference_file;
        if (currentSelectedValue !== 'none' && filesData.includes(currentSelectedValue)) {
            cloneReferenceSelect.value = currentSelectedValue;
        } else if (lastSelected && filesData.includes(lastSelected)) {
            cloneReferenceSelect.value = lastSelected;
        } else {
            cloneReferenceSelect.value = 'none';
        }
    }

    function updatePresetVisuals(name) {
        currentPresetName = name;

        // Find all preset buttons
        const buttons = document.querySelectorAll('.preset-btn');
        buttons.forEach(btn => {
            // We will add data-name to buttons in the next step
            if (btn.dataset.name === name) {
                btn.classList.add('selected');
            } else {
                btn.classList.remove('selected');
            }
        });
    }

    function populatePresets() {
        if (!presetsContainer || !appPresets) return;

        // Hide presets the loaded model cannot actually render.
        //
        // The original check was name-based -- startsWith('turbo') -- but every
        // such preset is named "\u26a1 Turbo: ...", so the leading emoji meant it
        // never matched and the tag-heavy presets stayed on screen for models
        // that ignore the tags. Filter on the thing that matters instead: does
        // the preset's TEXT use paralinguistic tags this model supports?
        const PARALINGUISTIC_TAG = /\[(laugh|chuckle|sigh|gasp|cough|clear throat|sniff|groan|shush)\]/i;
        let filteredPresets = appPresets;
        if (currentModelInfo && currentModelInfo.supports_paralinguistic_tags === false) {
            filteredPresets = appPresets.filter(preset => !PARALINGUISTIC_TAG.test(preset.text || ''));
        }

        // Clear container
        presetsContainer.innerHTML = '';

        if (filteredPresets.length === 0) {
            const placeholder = document.createElement('p');
            placeholder.className = 'form-hint';
            placeholder.textContent = 'No presets available for this model.';
            presetsContainer.appendChild(placeholder);
            return;
        }

        filteredPresets.forEach((preset, index) => {
            const button = document.createElement('button');
            button.type = 'button';
            button.id = `preset-btn-${index}`;
            button.className = 'preset-btn';
            button.dataset.name = preset.name;
            button.title = `Load '${preset.name}' preset`;
            button.textContent = preset.name;
            button.addEventListener('click', () => applyPreset(preset));
            presetsContainer.appendChild(button);
        });

        if (currentPresetName) {
            updatePresetVisuals(currentPresetName);
        }
    }

    function applyPreset(presetData, showNotif = true, isUserInteraction = true) {
        if (!presetData) return;
        if (textArea && presetData.text !== undefined) {
            textArea.value = presetData.text;
            if (charCount) charCount.textContent = textArea.value.length;
        }
        const genParams = presetData.params || presetData;
        if (temperatureSlider && genParams.temperature !== undefined) temperatureSlider.value = genParams.temperature;
        if (exaggerationSlider && genParams.exaggeration !== undefined) exaggerationSlider.value = genParams.exaggeration;
        if (cfgWeightSlider && genParams.cfg_weight !== undefined) cfgWeightSlider.value = genParams.cfg_weight;
        if (temperatureValueDisplay && temperatureSlider) temperatureValueDisplay.textContent = temperatureSlider.value;
        if (exaggerationValueDisplay && exaggerationSlider) exaggerationValueDisplay.textContent = exaggerationSlider.value;
        if (cfgWeightValueDisplay && cfgWeightSlider) cfgWeightValueDisplay.textContent = cfgWeightSlider.value;

        if (genParams.voice_id && predefinedVoiceSelect) {
            const voiceExists = Array.from(predefinedVoiceSelect.options).some(opt => opt.value === genParams.voice_id);
            if (voiceExists) {
                predefinedVoiceSelect.value = genParams.voice_id;
                const predefinedRadio = document.querySelector('input[name="voice_mode"][value="predefined"]');
                if (predefinedRadio) {
                    predefinedRadio.checked = true;
                    predefinedRadio.dispatchEvent(new Event('change', { bubbles: true }));
                }
                toggleVoiceOptionsDisplay();
            }
        } else if (genParams.reference_audio_filename && cloneReferenceSelect) {
            const refExists = Array.from(cloneReferenceSelect.options).some(opt => opt.value === genParams.reference_audio_filename);
            if (refExists) {
                cloneReferenceSelect.value = genParams.reference_audio_filename;
                const cloneRadio = document.querySelector('input[name="voice_mode"][value="clone"]');
                if (cloneRadio) {
                    cloneRadio.checked = true;
                    cloneRadio.dispatchEvent(new Event('change', { bubbles: true }));
                }
                toggleVoiceOptionsDisplay();
            }
        }

        if (presetData.name) {
            updatePresetVisuals(presetData.name);
        }

        if (showNotif) showNotification(`Preset "${presetData.name}" loaded.`, 'info', 3000);
        if (isUserInteraction) {
            debouncedSaveState();
        }
    }

    // --- Voice Mode and Options Visibility ---
    function toggleVoiceOptionsDisplay() {
        const selectedMode = document.querySelector('input[name="voice_mode"]:checked')?.value;
        currentVoiceMode = selectedMode;
        if (predefinedVoiceOptionsDiv) predefinedVoiceOptionsDiv.classList.toggle('hidden', selectedMode !== 'predefined');
        if (cloneOptionsDiv) cloneOptionsDiv.classList.toggle('hidden', selectedMode !== 'clone');
        if (predefinedVoiceSelect) predefinedVoiceSelect.required = (selectedMode === 'predefined');
        if (cloneReferenceSelect) cloneReferenceSelect.required = (selectedMode === 'clone');
    }
    voiceModeRadios.forEach(radio => radio.addEventListener('change', toggleVoiceOptionsDisplay));

    function toggleChunkControlsVisibility() {
        const isChecked = splitTextToggle ? splitTextToggle.checked : false;
        if (chunkSizeControls) chunkSizeControls.classList.toggle('hidden', !isChecked);
        if (chunkExplanation) chunkExplanation.classList.toggle('hidden', !isChecked);
    }
    if (splitTextToggle) toggleChunkControlsVisibility();

    // --- TTS Generation Logic ---
    function getTTSFormData() {
        const jsonData = {
            text: textArea.value,
            temperature: parseFloat(temperatureSlider.value),
            exaggeration: parseFloat(exaggerationSlider.value),
            cfg_weight: parseFloat(cfgWeightSlider.value),
            num_steps: parseInt(numStepsSlider.value, 10),
            n_cfm_timesteps: parseInt(cfmTimestepsSlider.value, 10),
            voice_mode: currentVoiceMode,
            split_text: splitTextToggle.checked,
            chunk_size: parseInt(chunkSizeSlider.value, 10),
        };
        if (currentVoiceMode === 'predefined' && predefinedVoiceSelect.value !== 'none') {
            jsonData.predefined_voice_id = predefinedVoiceSelect.value;
        } else if (currentVoiceMode === 'clone' && cloneReferenceSelect.value !== 'none') {
            jsonData.reference_audio_filename = cloneReferenceSelect.value;
        }
        return jsonData;
    }

    // --- Realtime streaming ---
    let rt = null;                      // RealtimeTtsClient
    let analyser = null, meterRaf = null, meterCtx = null;
    let metrics = null;                 // per-response timing
    let lastDownloadUrl = null;         // blob URL for the last recorded utterance
    const $ = (id) => document.getElementById(id);

    function logEvent(kind, ev) {
        const log = $('events-log');
        const line = document.createElement('div');
        line.className = kind;
        const ts = new Date().toISOString().slice(11, 23);
        line.textContent = `${ts} ${kind === 'out' ? '→' : '←'} ${JSON.stringify(ev)}`;
        log.appendChild(line);
        log.scrollTop = log.scrollHeight;
    }

    function setPill(id, label, cls) { const el = $(id); el.textContent = label; el.className = 'pill' + (cls ? ' ' + cls : ''); }

    function sessionPatchFromControls() {
        const data = getTTSFormData();
        const voice = currentVoiceMode === 'predefined' ? data.predefined_voice_id : data.reference_audio_filename;
        return {
            audio: { output: { voice } },
            x_chatterbox: {
                temperature: data.temperature, exaggeration: data.exaggeration, cfg_scale: data.cfg_weight,
                num_steps: data.num_steps, n_cfm_timesteps: data.n_cfm_timesteps,
                chunk_size: data.chunk_size, split_text: data.split_text, split_on_clauses: true,
            },
        };
    }

    async function ensureConnected() {
        if (rt && rt.state.dc === 'open') return rt;
        rt = new RealtimeTtsClient({ baseUrl: API_BASE_URL, session: sessionPatchFromControls(),
                                     iceServers: [{ urls: 'stun:stun.l.google.com:19302' }] });
        rt.on('*', (ev) => logEvent(ev.type === 'error' ? 'err' : 'in', ev));
        rt.on('client-event', (ev) => logEvent('out', ev));
        rt.on('state', (s) => {
            setPill('pill-pc', `pc: ${s.pc}`, s.pc === 'connected' ? 'ok' : s.pc === 'failed' ? 'err' : '');
            setPill('pill-ice', `ice: ${s.ice}`, /connected|completed/.test(s.ice) ? 'ok' : s.ice === 'failed' ? 'err' : '');
            setPill('pill-dc', `dc: ${s.dc}`, s.dc === 'open' ? 'ok' : '');
            $('disconnect-btn').disabled = s.dc !== 'open';
        });
        rt.on('track', (stream) => { $('remote-audio').srcObject = stream; startMeter(stream); });
        rt.on('response.created', (ev) => { metrics = { id: ev.response.id, created: performance.now(), chunks: 0, firstAudio: null, serverStarted: null }; renderMetrics(); });
        rt.on('response.output_audio_transcript.delta', () => { if (metrics) { metrics.chunks++; renderMetrics(); } });
        rt.on('output_audio_buffer.started', () => { if (metrics) { metrics.serverStarted = performance.now(); renderMetrics(); } hideLoadingOverlay(); });
        rt.on('response.done', (ev) => { if (metrics) { metrics.done = performance.now(); metrics.status = ev.response.status; renderMetrics(); } $('stop-btn').disabled = true; isGenerating = false; hideLoadingOverlay(); });
        rt.on('output_audio_buffer.stopped', () => { if (metrics) { metrics.stopped = performance.now(); renderMetrics(); } });
        rt.on('error', (ev) => showNotification(`${ev.error.code}: ${ev.error.message}`, 'error'));

        // --- MediaRecorder capture of the remote stream, for offline A/B download ---
        let recorder = null, recorded = [];
        rt.on('response.created', () => {
            if (!rt.remoteStream) return;
            recorded = [];
            const mimeType = (window.MediaRecorder && MediaRecorder.isTypeSupported('audio/webm;codecs=opus'))
                ? 'audio/webm;codecs=opus' : null;
            recorder = mimeType ? new MediaRecorder(rt.remoteStream, { mimeType }) : new MediaRecorder(rt.remoteStream);
            recorder.ondataavailable = (e) => { if (e.data.size) recorded.push(e.data); };
            recorder.onstop = () => {
                if (lastDownloadUrl) URL.revokeObjectURL(lastDownloadUrl);
                lastDownloadUrl = URL.createObjectURL(new Blob(recorded, { type: 'audio/webm' }));
                const a = $('download-last'); a.href = lastDownloadUrl; a.hidden = false;
            };
            recorder.start(250);
        });
        rt.on('output_audio_buffer.stopped', () => { if (recorder && recorder.state === 'recording') recorder.stop(); });
        rt.on('output_audio_buffer.cleared', () => { if (recorder && recorder.state === 'recording') recorder.stop(); });

        await rt.connect();
        return rt;
    }

    function startMeter(stream) {
        if (meterCtx) { try { meterCtx.close(); } catch {} }
        meterCtx = new (window.AudioContext || window.webkitAudioContext)();
        if (meterCtx.state === 'suspended') meterCtx.resume();
        const src = meterCtx.createMediaStreamSource(stream);
        analyser = meterCtx.createAnalyser(); analyser.fftSize = 512;
        src.connect(analyser);
        const buf = new Float32Array(analyser.fftSize);
        const tick = () => {
            analyser.getFloatTimeDomainData(buf);
            let peak = 0; for (const v of buf) peak = Math.max(peak, Math.abs(v));
            $('level-bar').style.width = `${Math.min(100, peak * 300)}%`;
            if (metrics && metrics.firstAudio === null && peak > 0.01) { metrics.firstAudio = performance.now(); renderMetrics(); }
            meterRaf = requestAnimationFrame(tick);
        };
        if (meterRaf) cancelAnimationFrame(meterRaf);
        tick();
    }

    function renderMetrics() {
        if (!metrics) return;
        const s = (a, b) => (a != null && b != null) ? `${((b - a) / 1000).toFixed(3)} s` : '–';
        $('m-ttfa').textContent = s(metrics.created, metrics.firstAudio);
        $('m-ttfa-server').textContent = s(metrics.created, metrics.serverStarted);
        $('m-total').textContent = s(metrics.created, metrics.done) + (metrics.status ? ` (${metrics.status})` : '');
        $('m-audio').textContent = s(metrics.serverStarted, metrics.stopped);
        $('m-chunks').textContent = String(metrics.chunks);
    }

    async function submitTTSRequest() {
        isGenerating = true;
        showLoadingOverlay();
        try {
            const client = await ensureConnected();
            await client.updateSession(sessionPatchFromControls());
            $('stop-btn').disabled = false;
            await client.speak(textArea.value);
        } catch (error) {
            console.error('Realtime error:', error);
            showNotification(error.message || 'Streaming failed.', 'error');
            isGenerating = false;
            hideLoadingOverlay();
        }
    }

    $('stop-btn').addEventListener('click', async () => {
        if (!rt) return;
        try { await rt.cancel(); await rt.clear(); } catch (e) { showNotification(e.message, 'error'); }
    });
    $('disconnect-btn').addEventListener('click', async () => {
        if (rt) { await rt.disconnect(); rt = null; }
        if (meterRaf) { cancelAnimationFrame(meterRaf); meterRaf = null; }
        if (meterCtx) { try { meterCtx.close(); } catch {} meterCtx = null; }
    });

    function proceedWithSubmissionChecks() {
        const textContent = textArea.value.trim();
        const isSplittingEnabled = splitTextToggle.checked;
        const currentChunkSz = parseInt(chunkSizeSlider.value, 10);
        const needsChunkWarn = isSplittingEnabled && textContent.length >= currentChunkSz * 1.5 &&
            currentVoiceMode !== 'predefined' && currentVoiceMode !== 'clone' && !hideChunkWarning;
        if (needsChunkWarn) { showChunkWarningModal(); return; }
        submitTTSRequest();
    }

    // --- Attach main generation event to the button's CLICK, not the form's SUBMIT ---
    // This is a more robust method that prevents accidental submissions during page load.
    if (generateBtn) {
        generateBtn.addEventListener('click', function (event) {

            console.log('Generate button clicked!');
            console.log('Current voice mode:', currentVoiceMode);
            console.log('Is generating:', isGenerating);
            console.log('Text content:', textArea ? textArea.value.trim() : 'NO TEXTAREA');

            // We still prevent default in case the button has any default browser actions.
            event.preventDefault();

            if (isGenerating) {
                showNotification("Generation is already in progress.", "warning");
                return;
            }
            const textContent = textArea.value.trim();
            if (!textContent) {
                showNotification("Please enter some text to generate speech.", 'error');
                return;
            }
            if (currentVoiceMode === 'predefined' && (!predefinedVoiceSelect || predefinedVoiceSelect.value === 'none')) {
                showNotification("Please select a predefined voice.", 'error');
                return;
            }
            if (currentVoiceMode === 'clone' && (!cloneReferenceSelect || cloneReferenceSelect.value === 'none')) {
                showNotification("Please select a reference audio file for Voice Cloning.", 'error');
                return;
            }

            // Check for the generation quality warning.
            if (!hideGenerationWarning) {
                showGenerationWarningModal();
                return; // Stop here and let the modal handler take over.
            }

            // If the warning is hidden, proceed to the final checks.
            proceedWithSubmissionChecks();
        });
    } else {
        console.log('Generate button not found!');
    }

    // --- Modal Handling ---
    function showChunkWarningModal() {
        if (chunkWarningModal) {
            chunkWarningModal.style.display = 'flex';
            chunkWarningModal.classList.remove('hidden', 'opacity-0');
            chunkWarningModal.dataset.state = 'open';
        }
    }
    function hideChunkWarningModal() {
        if (chunkWarningModal) {
            chunkWarningModal.classList.add('opacity-0');
            setTimeout(() => {
                chunkWarningModal.style.display = 'none';
                chunkWarningModal.dataset.state = 'closed';
            }, 300);
        }
    }
    function showGenerationWarningModal() {
        if (generationWarningModal) {
            generationWarningModal.style.display = 'flex';
            generationWarningModal.classList.remove('hidden', 'opacity-0');
            generationWarningModal.dataset.state = 'open';
        }
    }
    function hideGenerationWarningModal() {
        if (generationWarningModal) {
            generationWarningModal.classList.add('opacity-0');
            setTimeout(() => {
                generationWarningModal.style.display = 'none';
                generationWarningModal.dataset.state = 'closed';
            }, 300);
        }
    }
    if (chunkWarningOkBtn) chunkWarningOkBtn.addEventListener('click', () => {
        if (hideChunkWarningCheckbox && hideChunkWarningCheckbox.checked) hideChunkWarning = true;
        hideChunkWarningModal(); debouncedSaveState(); submitTTSRequest();
    });
    if (chunkWarningCancelBtn) chunkWarningCancelBtn.addEventListener('click', hideChunkWarningModal);
    if (generationWarningAcknowledgeBtn) generationWarningAcknowledgeBtn.addEventListener('click', () => {
        if (hideGenerationWarningCheckbox && hideGenerationWarningCheckbox.checked) hideGenerationWarning = true;
        hideGenerationWarningModal(); debouncedSaveState(); proceedWithSubmissionChecks();
    });
    if (loadingCancelBtn) loadingCancelBtn.addEventListener('click', () => {
        if (isGenerating) {
            isGenerating = false;
            hideLoadingOverlay();
            showNotification("Generation UI cancelled by user.", "info");
            if (rt) rt.cancel().catch(() => {});
        }
    });
    function showLoadingOverlay() {
        if (loadingOverlay && generateBtn && loadingCancelBtn) {
            loadingMessage.textContent = 'Generating audio...';
            loadingStatusText.textContent = 'Please wait. This may take some time.';
            loadingOverlay.style.display = 'flex';
            loadingOverlay.classList.remove('hidden', 'opacity-0'); loadingOverlay.dataset.state = 'open';
            generateBtn.disabled = true; loadingCancelBtn.disabled = false;
        }
    }
    function hideLoadingOverlay() {
        if (loadingOverlay && generateBtn) {
            loadingOverlay.classList.add('opacity-0');
            setTimeout(() => {
                loadingOverlay.style.display = 'none';
                loadingOverlay.dataset.state = 'closed';
            }, 300);
            generateBtn.disabled = false;
        }
    }

    // --- Configuration Management ---
    function displayServerConfiguration() {
        if (!serverConfigForm || !currentConfig || Object.keys(currentConfig).length === 0) return;
        const fieldsToDisplay = {
            "server.host": currentConfig.server?.host, "server.port": currentConfig.server?.port,
            "tts_engine.device": currentConfig.tts_engine?.device, "tts_engine.default_voice_id": currentConfig.tts_engine?.default_voice_id,
            "paths.model_cache": currentConfig.paths?.model_cache, "tts_engine.predefined_voices_path": currentConfig.tts_engine?.predefined_voices_path,
            "tts_engine.reference_audio_path": currentConfig.tts_engine?.reference_audio_path, "paths.output": currentConfig.paths?.output,
            "audio_output.format": currentConfig.audio_output?.format, "audio_output.sample_rate": currentConfig.audio_output?.sample_rate
        };
        const checkboxFields = {
            "audio_output.save_to_disk": currentConfig.audio_output?.save_to_disk
        };
        for (const name in fieldsToDisplay) {
            const input = serverConfigForm.querySelector(`input[name="${name}"]`);
            if (input) {
                input.value = fieldsToDisplay[name] !== undefined ? fieldsToDisplay[name] : '';
                if (name.includes('.host') || name.includes('.port') || name.includes('.device') || name.includes('paths.')) input.readOnly = true;
                else input.readOnly = false;
            }
        }
        for (const name in checkboxFields) {
            const input = serverConfigForm.querySelector(`input[name="${name}"]`);
            if (input) input.checked = !!checkboxFields[name];
        }
    }
    async function updateConfigStatus(button, statusElem, message, type = 'info', duration = 5000, enableButtonAfter = true) {
        const statusClasses = { success: 'text-green-600 dark:text-green-400', error: 'text-red-600 dark:text-red-400', warning: 'text-yellow-600 dark:text-yellow-400', info: 'text-indigo-600 dark:text-indigo-400', processing: 'text-yellow-600 dark:text-yellow-400 animate-pulse' };
        const isProcessing = message.toLowerCase().includes('saving') || message.toLowerCase().includes('restarting') || message.toLowerCase().includes('resetting');
        const messageType = isProcessing ? 'processing' : type;
        if (statusElem) {
            statusElem.textContent = message;
            statusElem.className = `text-xs ml-2 ${statusClasses[messageType] || statusClasses['info']}`;
            statusElem.classList.remove('hidden');
        }
        if (button) button.disabled = isProcessing || (type === 'error' && !enableButtonAfter) || (type === 'success' && !enableButtonAfter);
        if (duration > 0) setTimeout(() => { if (statusElem) statusElem.classList.add('hidden'); if (button && enableButtonAfter) button.disabled = false; }, duration);
        else if (button && enableButtonAfter && !isProcessing) button.disabled = false;
    }

    if (saveConfigBtn && configStatus) {
        saveConfigBtn.addEventListener('click', async () => {
            const configDataToSave = {};
            const inputs = serverConfigForm.querySelectorAll('input[name]:not([readonly]), select[name]:not([readonly])');
            inputs.forEach(input => {
                const keys = input.name.split('.'); let currentLevel = configDataToSave;
                keys.forEach((key, index) => {
                    if (index === keys.length - 1) {
                        let value = input.value;
                        if (input.type === 'number') value = parseFloat(value) || 0;
                        else if (input.type === 'checkbox') value = input.checked;
                        currentLevel[key] = value;
                    } else { currentLevel[key] = currentLevel[key] || {}; currentLevel = currentLevel[key]; }
                });
            });
            if (Object.keys(configDataToSave).length === 0) { showNotification("No editable configuration values to save.", "info"); return; }
            updateConfigStatus(saveConfigBtn, configStatus, 'Saving configuration...', 'info', 0, false);
            try {
                const response = await fetch(`${API_BASE_URL}/save_settings`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(configDataToSave)
                });
                const result = await response.json();
                if (!response.ok) throw new Error(result.detail || 'Failed to save configuration');
                updateConfigStatus(saveConfigBtn, configStatus, result.message || 'Configuration saved.', 'success', 5000);
                if (result.restart_needed && restartServerBtn) restartServerBtn.classList.remove('hidden');
                await fetchInitialData();
                showNotification("Configuration saved. Some changes may require a server restart if prompted.", "success");
            } catch (error) {
                console.error('Error saving server config:', error);
                updateConfigStatus(saveConfigBtn, configStatus, `Error: ${error.message}`, 'error', 0);
            }
        });
    }

    if (saveGenDefaultsBtn && genDefaultsStatus) {
        saveGenDefaultsBtn.addEventListener('click', async () => {
            const genParams = {
                temperature: parseFloat(temperatureSlider.value), exaggeration: parseFloat(exaggerationSlider.value),
                cfg_weight: parseFloat(cfgWeightSlider.value),
            };
            updateConfigStatus(saveGenDefaultsBtn, genDefaultsStatus, 'Saving generation defaults...', 'info', 0, false);
            try {
                const response = await fetch(`${API_BASE_URL}/save_settings`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ generation_defaults: genParams })
                });
                const result = await response.json();
                if (!response.ok) throw new Error(result.detail || 'Failed to save generation defaults');
                updateConfigStatus(saveGenDefaultsBtn, genDefaultsStatus, result.message || 'Generation defaults saved.', 'success', 5000);
                if (currentConfig.generation_defaults) Object.assign(currentConfig.generation_defaults, genParams);
            } catch (error) {
                console.error('Error saving generation defaults:', error);
                updateConfigStatus(saveGenDefaultsBtn, genDefaultsStatus, `Error: ${error.message}`, 'error', 0);
            }
        });
    }

    if (resetSettingsBtn) {
        resetSettingsBtn.addEventListener('click', async () => {
            if (!confirm("Are you sure you want to reset ALL settings to their initial defaults? This will affect config.yaml and UI preferences. This action cannot be undone.")) return;
            updateConfigStatus(resetSettingsBtn, configStatus, 'Resetting settings...', 'info', 0, false);
            try {
                const response = await fetch(`${API_BASE_URL}/reset_settings`, {
                    method: 'POST'
                });
                if (!response.ok) {
                    const errorResult = await response.json().catch(() => ({ detail: 'Failed to reset settings on server.' }));
                    throw new Error(formatErrorDetail(errorResult.detail));
                }
                const result = await response.json();
                updateConfigStatus(resetSettingsBtn, configStatus, result.message + " Reloading page...", 'success', 0, false);
                setTimeout(() => window.location.reload(true), 2000);
            } catch (error) {
                console.error('Error resetting settings:', error);
                updateConfigStatus(resetSettingsBtn, configStatus, `Reset Error: ${error.message}`, 'error', 0);
                showNotification(`Error resetting settings: ${error.message}`, 'error');
            }
        });
    }

    if (restartServerBtn) {
        restartServerBtn.addEventListener('click', async () => {
            if (!confirm("Are you sure you want to restart the server?")) return;
            updateConfigStatus(restartServerBtn, configStatus, 'Attempting server restart...', 'processing', 0, false);
            try {
                const response = await fetch(`${API_BASE_URL}/restart_server`, {
                    method: 'POST'
                });
                const result = await response.json();
                if (!response.ok) throw new Error(result.detail || 'Server responded with error on restart command');
                showNotification("Server restart initiated. Please wait a moment for the server to come back online, then refresh the page.", "info", 10000);
            } catch (error) {
                showNotification(`Server restart command failed: ${error.message}`, "error");
                updateConfigStatus(restartServerBtn, configStatus, `Restart failed.`, 'error', 5000, true);
            }
        });
    }

    // --- File Upload & Refresh ---
    async function handleFileUpload(fileInput, endpoint, successCallback, buttonToAnimate) {
        const files = fileInput.files;
        if (!files || files.length === 0) return;
        const originalButtonHTML = buttonToAnimate ? buttonToAnimate.innerHTML : '';
        if (buttonToAnimate) {
            buttonToAnimate.innerHTML = `<svg class="animate-spin h-5 w-5 mr-1.5 inline-block" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>Uploading...`;
            buttonToAnimate.disabled = true;
        }
        const uploadNotification = showNotification(`Uploading ${files.length} file(s)...`, 'info', 0);
        const formData = new FormData();
        for (const file of files) formData.append('files', file);
        try {
            const response = await fetch(`${API_BASE_URL}${endpoint}`, {
                method: 'POST',
                body: formData
            });
            const result = await response.json();
            if (uploadNotification) uploadNotification.remove();
            if (!response.ok) throw new Error(result.message || result.detail || `Upload failed with status ${response.status}`);
            if (result.errors && result.errors.length > 0) {
                result.errors.forEach(err => showNotification(`Upload Warning: ${err.filename || 'File'} - ${err.error}`, 'warning', 10000));
            }
            const successfulUploads = result.uploaded_files || [];
            if (successfulUploads.length > 0) {
                showNotification(`Successfully uploaded: ${successfulUploads.join(', ')}`, 'success');
            } else if (!result.errors || result.errors.length === 0) {
                showNotification("Files processed. No new valid files were added or an issue occurred.", 'info');
            }
            successCallback(result);
            debouncedSaveState();
        } catch (error) {
            console.error(`Error uploading to ${endpoint}:`, error);
            if (uploadNotification) uploadNotification.remove();
            showNotification(`Upload Error: ${error.message}`, 'error');
        } finally {
            if (buttonToAnimate) {
                buttonToAnimate.disabled = false;
                buttonToAnimate.innerHTML = originalButtonHTML;
            }
            fileInput.value = '';
        }
    }

    if (cloneImportButton && cloneFileInput) {
        cloneImportButton.addEventListener('click', () => cloneFileInput.click());
        cloneFileInput.addEventListener('change', () => handleFileUpload(cloneFileInput, '/upload_reference', (result) => {
            initialReferenceFiles = result.all_reference_files || [];
            populateReferenceFiles();
            const firstUploaded = result.uploaded_files?.[0];
            if (firstUploaded && cloneReferenceSelect && Array.from(cloneReferenceSelect.options).some(opt => opt.value === firstUploaded)) {
                cloneReferenceSelect.value = firstUploaded;
            }
        }, cloneImportButton));
    }

    if (predefinedVoiceImportButton && predefinedVoiceFileInput) {
        predefinedVoiceImportButton.addEventListener('click', () => predefinedVoiceFileInput.click());
        predefinedVoiceFileInput.addEventListener('change', () => handleFileUpload(predefinedVoiceFileInput, '/upload_predefined_voice', (result) => {
            initialPredefinedVoices = result.all_predefined_voices || [];
            populatePredefinedVoices();
            const firstUploadedFilename = result.uploaded_files?.[0];
            if (firstUploadedFilename && predefinedVoiceSelect && initialPredefinedVoices.some(v => v.filename === firstUploadedFilename)) {
                predefinedVoiceSelect.value = firstUploadedFilename;
            }
        }, predefinedVoiceImportButton));
    }

    if (cloneRefreshButton && cloneReferenceSelect) {
        cloneRefreshButton.addEventListener('click', async () => {
            const originalButtonIcon = cloneRefreshButton.innerHTML;
            cloneRefreshButton.innerHTML = `<svg class="animate-spin h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>`;
            cloneRefreshButton.disabled = true;
            try {
                const response = await fetch(`${API_BASE_URL}/get_reference_files`);
                if (!response.ok) throw new Error('Failed to fetch reference files list');
                const files = await response.json();
                initialReferenceFiles = files;
                populateReferenceFiles();
                showNotification("Reference file list refreshed.", 'info', 2000);
                debouncedSaveState();
            } catch (error) {
                console.error("Error refreshing reference files:", error);
                showNotification(`Error refreshing list: ${error.message}`, 'error');
            } finally {
                cloneRefreshButton.disabled = false;
                cloneRefreshButton.innerHTML = originalButtonIcon;
            }
        });
    }

    if (predefinedVoiceRefreshButton && predefinedVoiceSelect) {
        predefinedVoiceRefreshButton.addEventListener('click', async () => {
            const originalButtonIcon = predefinedVoiceRefreshButton.innerHTML;
            predefinedVoiceRefreshButton.innerHTML = `<svg class="animate-spin h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>`;
            predefinedVoiceRefreshButton.disabled = true;
            try {
                const response = await fetch(`${API_BASE_URL}/get_predefined_voices`);
                if (!response.ok) throw new Error('Failed to fetch predefined voices list');
                const voices = await response.json();
                initialPredefinedVoices = voices;
                populatePredefinedVoices();
                showNotification("Predefined voices list refreshed.", 'info', 2000);
                debouncedSaveState();
            } catch (error) {
                console.error("Error refreshing predefined voices:", error);
                showNotification(`Error refreshing list: ${error.message}`, 'error');
            } finally {
                predefinedVoiceRefreshButton.disabled = false;
                predefinedVoiceRefreshButton.innerHTML = originalButtonIcon;
            }
        });
    }

    // Call fetchInitialData at the end of setup to kick everything off.
    // Note: This calls initializeApplication internally.
    await fetchInitialData();
});
