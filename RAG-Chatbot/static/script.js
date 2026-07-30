document.addEventListener('DOMContentLoaded', () => {
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const chatHistory = document.getElementById('chat-history');
    const sendButton = document.getElementById('send-button');

    function scrollToBottom() {
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }

    function addMessage(content, role) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}-message`;
        
        const bubble = document.createElement('div');
        bubble.className = 'message-bubble';
        
        // Simple markdown formatting (bold, newlines)
        let formattedContent = content
            .replace(/\n/g, '<br>')
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>');
            
        bubble.innerHTML = formattedContent;
        
        messageDiv.appendChild(bubble);
        chatHistory.appendChild(messageDiv);
        scrollToBottom();
    }

    function addTypingIndicator() {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message assistant-message typing';
        messageDiv.id = 'typing-indicator';
        
        const indicator = document.createElement('div');
        indicator.className = 'typing-indicator';
        indicator.innerHTML = '<div class="dot"></div><div class="dot"></div><div class="dot"></div>';
        
        messageDiv.appendChild(indicator);
        chatHistory.appendChild(messageDiv);
        scrollToBottom();
    }

    function removeTypingIndicator() {
        const indicator = document.getElementById('typing-indicator');
        if (indicator) {
            indicator.remove();
        }
    }

    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const query = userInput.value.trim();
        if (!query) return;
        
        // Disable input and add user message
        userInput.value = '';
        userInput.disabled = true;
        sendButton.disabled = true;
        
        addMessage(query, 'user');
        addTypingIndicator();
        
        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ query: query })
            });
            
            const data = await response.json();
            
            removeTypingIndicator();
            
            if (response.ok) {
                addMessage(data.response, 'assistant');
            } else {
                addMessage(`Error: ${data.error || 'Something went wrong.'}`, 'assistant');
            }
            
        } catch (error) {
            removeTypingIndicator();
            addMessage(`Connection error: ${error.message}`, 'assistant');
        } finally {
            userInput.disabled = false;
            sendButton.disabled = false;
            userInput.focus();
            scrollToBottom();
        }
    });
});
