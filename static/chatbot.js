const chatbotToggle = document.getElementById('chatbot_toggle');
const chatbot = document.getElementById('chatbot');
const chatArea = document.getElementById('chat_area');
const inputMessage = document.getElementById('input_message');
const sendButton = document.getElementById('send_button');

chatbotToggle.addEventListener('click', () => {
  if (chatbot.classList.contains('collapsed')) {
    chatbot.classList.remove('collapsed');
  } else {
    chatbot.classList.add('collapsed');
  }
});

sendButton.addEventListener('click', sendMessage);
inputMessage.addEventListener('keydown', e => {
  if (e.key === 'Enter') {
    sendMessage();
  }
});

function addMessage(text, sender) {
  const messageDiv = document.createElement('div');
  messageDiv.classList.add('chat-message-div');

  const messageContent = document.createElement('div');
  if (sender === 'user') {
    messageContent.classList.add('chat-message-sent');
  } else {
    messageContent.classList.add('chat-message-received');
  }
  messageContent.textContent = text;

  messageDiv.appendChild(messageContent);
  chatArea.appendChild(messageDiv);
  chatArea.scrollTop = chatArea.scrollHeight;
}

async function sendMessage() {
  const msg = inputMessage.value.trim();
  if (!msg) return;

  addMessage(msg, 'user');
  inputMessage.value = '';
  inputMessage.focus();

  try {
    const response = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: msg })
    });
    const data = await response.json();

    addMessage(data.reply, 'bot');
  } catch (error) {
    addMessage('Oops! Something went wrong.', 'bot');
    console.error(error);
  }
}
