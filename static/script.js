document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');
    const imagePreviewContainer = document.getElementById('image-preview-container');
    const imagePreview = document.getElementById('image-preview');
    const predictButton = document.getElementById('predict-button');
    const resultContainer = document.getElementById('result-container');
    const resultLoading = document.getElementById('result-loading');
    const resultContent = document.getElementById('result-content');
    const resultIcon = document.getElementById('result-icon');
    const resultText = document.getElementById('result-text');
    const resultConfidence = document.getElementById('result-confidence');
    const chatForm = document.getElementById('chat-form');
    const chatInput = document.getElementById('chat-input');
    const chatMessages = document.getElementById('chat-messages');

    // File Upload Functionality
    uploadArea.addEventListener('click', () => {
      fileInput.click();
    });

    uploadArea.addEventListener('dragover', (e) => {
      e.preventDefault();
      uploadArea.style.borderColor = '#38BDF8';
    });

    uploadArea.addEventListener('dragleave', () => {
      uploadArea.style.borderColor = '#CBD5E1';
    });

    uploadArea.addEventListener('drop', (e) => {
      e.preventDefault();
      uploadArea.style.borderColor = '#CBD5E1';

      if (e.dataTransfer.files.length) {
        handleFile(e.dataTransfer.files[0]);
      }
    });

    fileInput.addEventListener('change', () => {
      if (fileInput.files.length) {
        handleFile(fileInput.files[0]);
      }
    });

    function handleFile(file) {
      // Check if file is an image
      if (!file.type.match('image.*')) {
        alert('Please upload an image file');
        return;
      }

      const reader = new FileReader();

      reader.onload = (e) => {
        imagePreview.src = e.target.result;
        imagePreviewContainer.classList.remove('hidden');
        resultContainer.classList.add('hidden'); // Hide any previous results
      };

      reader.readAsDataURL(file);
    }

    // Prediction Functionality
    predictButton.addEventListener('click', () => {
      // Show the result container and loading state
      resultContainer.classList.remove('hidden');
      resultLoading.classList.remove('hidden');
      resultContent.classList.add('hidden');

      // Prepare form data for API call
      const formData = new FormData();
      formData.append('image', fileInput.files[0]);

      // Make API call to /predict
      fetch('/predict', {
        method: 'POST',
        body: formData
      })
      .then(response => response.json())
      .then(data => {
        // Hide loading state and show results
        resultLoading.classList.add('hidden');
        resultContent.classList.remove('hidden');

        if (data.error) {
          resultIcon.className = 'result-icon error';
          resultIcon.innerHTML = `
            <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" width="24" height="24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
            </svg>
          `;
          resultText.textContent = 'Error';
          resultConfidence.textContent = data.error;
          return;
        }

        const hasTumor = data.predicted_class === 'yes';
        const confidence = data.probability_of_tumor_yes;

        if (hasTumor) {
          resultIcon.className = 'result-icon warning';
          resultIcon.innerHTML = `
            <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" width="24" height="24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
          `;
          resultText.textContent = 'Tumor Detected';
        } else {
          resultIcon.className = 'result-icon success';
          resultIcon.innerHTML = `
            <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" width="24" height="24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
            </svg>
          `;
          resultText.textContent = 'No Tumor Detected';
        }

        resultConfidence.textContent = `Confidence: ${confidence}%`;
      })
      .catch(error => {
        resultLoading.classList.add('hidden');
        resultContent.classList.remove('hidden');
        resultIcon.className = 'result-icon error';
        resultIcon.innerHTML = `
          <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" width="24" height="24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
          </svg>
        `;
        resultText.textContent = 'Error';
        resultConfidence.textContent = 'Failed to process the image';
      });
    });

    // Chatbot Functionality
    chatForm.addEventListener('submit', (e) => {
      e.preventDefault();

      const message = chatInput.value.trim();
      if (!message) return;

      // Add user message to chat
      addChatMessage(message, 'user');

      // Clear input
      chatInput.value = '';

      // Make API call to /chat
      fetch('/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ message })
      })
      .then(response => response.json())
      .then(data => {
        addChatMessage(data.response, 'bot');
      })
      .catch(error => {
        addChatMessage('Sorry, something went wrong. Please try again.', 'bot');
      });
    });

    function addChatMessage(message, sender) {
      const messageDiv = document.createElement('div');
      messageDiv.className = `chat-message ${sender}`;

      const avatar = sender === 'bot' ? 'AI' : 'You';
      const avatarClass = sender === 'bot' ? 'bot-avatar' : '';

      messageDiv.innerHTML = `
        <div class="message-content">
          <div class="avatar ${avatarClass}">${avatar}</div>
          <div class="message-bubble">
            <p>${message}</p>
          </div>
        </div>
      `;

      chatMessages.appendChild(messageDiv);

      // Auto-scroll to bottom
      chatMessages.scrollTop = chatMessages.scrollHeight;
    }
});